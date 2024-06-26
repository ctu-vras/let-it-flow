import argparse
import enum
import itertools
from collections import namedtuple
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

from bucketed_scene_flow_eval.datasets import construct_dataset
from bucketed_scene_flow_eval.datastructures import (
    EgoLidarFlow,
    O3DVisualizer,
    PointCloud,
    PointCloudFrame,
    PoseInfo,
    RGBFrame,
    RGBFrameLookup,
    RGBImage,
    TimeSyncedSceneFlowFrame,
)


class Mode(enum.Enum):
    PROJECT_LIDAR = "project_lidar"
    PROJECT_FLOW = "project_flow"


DEFAULT_TRANSITIONS = (15, 6, 4, 11, 13, 6)


def _make_colorwheel(transitions: tuple = DEFAULT_TRANSITIONS) -> np.ndarray:
    """Creates a colorwheel (borrowed/modified from flowpy).
    A colorwheel defines the transitions between the six primary hues:
    Red(255, 0, 0), Yellow(255, 255, 0), Green(0, 255, 0), Cyan(0, 255, 255), Blue(0, 0, 255) and Magenta(255, 0, 255).
    Args:
        transitions: Contains the length of the six transitions, based on human color perception.
    Returns:
        colorwheel: The RGB values of the transitions in the color space.
    Notes:
        For more information, see:
        https://web.archive.org/web/20051107102013/http://members.shaw.ca/quadibloc/other/colint.htm
        http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    """
    colorwheel_length = sum(transitions)
    # The red hue is repeated to make the colorwheel cyclic
    base_hues = map(
        np.array,
        (
            [255, 0, 0],
            [255, 255, 0],
            [0, 255, 0],
            [0, 255, 255],
            [0, 0, 255],
            [255, 0, 255],
            [255, 0, 0],
        ),
    )
    colorwheel = np.zeros((colorwheel_length, 3), dtype="uint8")
    hue_from = next(base_hues)
    start_index = 0
    for hue_to, end_index in zip(base_hues, itertools.accumulate(transitions)):
        transition_length = end_index - start_index
        colorwheel[start_index:end_index] = np.linspace(
            hue_from, hue_to, transition_length, endpoint=False
        )
        hue_from = hue_to
        start_index = end_index
    return colorwheel


def _color_by_distance(distances: np.ndarray, max_distance: float = 10.0, cmap: str = "viridis"):
    # Use distance to color points, normalized to [0, 1].
    colors = distances.copy()

    # Normalize to [0, 1]
    colors = colors / max_distance
    colors[colors > 1] = 1.0

    colormap = plt.get_cmap(cmap)
    colors = colormap(colors)[:, :3]
    return colors


def _flow_to_rgb(
    flow: np.ndarray,
    flow_max_radius: Optional[float] = 2.0,
    background: Optional[str] = "bright",
) -> np.ndarray:
    """Creates a RGB representation of an optical flow (borrowed/modified from flowpy).
    Args:
        flow: scene flow.
            flow[..., 0] should be the x-displacement
            flow[..., 1] should be the y-displacement
            flow[..., 2] should be the z-displacement
        flow_max_radius: Set the radius that gives the maximum color intensity, useful for comparing different flows.
            Default: The normalization is based on the input flow maximum radius.
        background: States if zero-valued flow should look 'bright' or 'dark'.
    Returns: An array of RGB colors.
    """
    valid_backgrounds = ("bright", "dark")
    if background not in valid_backgrounds:
        raise ValueError(
            f"background should be one the following: {valid_backgrounds}, not {background}."
        )
    wheel = _make_colorwheel()
    # For scene flow, it's reasonable to assume displacements in x and y directions only for visualization pursposes.
    complex_flow = flow[..., 0] + 1j * flow[..., 1]
    radius, angle = np.abs(complex_flow), np.angle(complex_flow)
    if flow_max_radius is None:
        flow_max_radius = np.max(radius)
    if flow_max_radius > 0:
        radius /= flow_max_radius
    ncols = len(wheel)
    # Map the angles from (-pi, pi] to [0, 2pi) to [0, ncols - 1)
    angle[angle < 0] += 2 * np.pi
    angle = angle * ((ncols - 1) / (2 * np.pi))
    # Make the wheel cyclic for interpolation
    wheel = np.vstack((wheel, wheel[0]))
    # Interpolate the hues
    (angle_fractional, angle_floor), angle_ceil = np.modf(angle), np.ceil(angle)
    angle_fractional = angle_fractional.reshape((angle_fractional.shape) + (1,))
    float_hue = (
        wheel[angle_floor.astype(np.int32)] * (1 - angle_fractional)
        + wheel[angle_ceil.astype(np.int32)] * angle_fractional
    )
    ColorizationArgs = namedtuple(
        "ColorizationArgs", ["move_hue_valid_radius", "move_hue_oversized_radius", "invalid_color"]
    )

    def move_hue_on_V_axis(hues, factors):
        return hues * np.expand_dims(factors, -1)

    def move_hue_on_S_axis(hues, factors):
        return 255.0 - np.expand_dims(factors, -1) * (255.0 - hues)

    if background == "dark":
        parameters = ColorizationArgs(
            move_hue_on_V_axis, move_hue_on_S_axis, np.array([255, 255, 255], dtype=np.float32)
        )
    else:
        parameters = ColorizationArgs(
            move_hue_on_S_axis, move_hue_on_V_axis, np.array([0, 0, 0], dtype=np.float32)
        )
    colors = parameters.move_hue_valid_radius(float_hue, radius)
    oversized_radius_mask = radius > 1
    colors[oversized_radius_mask] = parameters.move_hue_oversized_radius(
        float_hue[oversized_radius_mask], 1 / radius[oversized_radius_mask]
    )
    return colors.astype(np.uint8)


def _insert_into_image(
    rgb_image: RGBImage, projected_points: np.ndarray, colors: np.ndarray, reduction_factor: int
) -> RGBImage:
    # Suppress RuntimeWarning: invalid value encountered in cast
    with np.errstate(invalid="ignore"):
        projected_points = projected_points.astype(np.int32)

    valid_points_mask = (
        (projected_points[:, 0] >= 0)
        & (projected_points[:, 0] < rgb_image.shape[1])
        & (projected_points[:, 1] >= 0)
        & (projected_points[:, 1] < rgb_image.shape[0])
    )

    projected_points = projected_points[valid_points_mask]
    colors = colors[valid_points_mask]

    scaled_rgb = rgb_image.rescale(reduction_factor)
    scaled_projected_points = projected_points // reduction_factor
    projected_rgb_image = scaled_rgb.full_image
    projected_rgb_image[scaled_projected_points[:, 1], scaled_projected_points[:, 0], :] = colors
    return RGBImage(projected_rgb_image)


def project_lidar_into_rgb(
    pc_frame: PointCloudFrame, rgb_frame: RGBFrame, reduction_factor: int
) -> RGBImage:
    pc_into_cam_frame_se3 = pc_frame.pose.sensor_to_ego.compose(
        rgb_frame.pose.sensor_to_ego.inverse()
    )
    cam_frame_pc = pc_frame.full_pc.transform(pc_into_cam_frame_se3)

    # cam_frame_pc = PointCloud(cam_frame_pc.points[cam_frame_pc.points[:, 0] >= 0])

    projected_points = rgb_frame.camera_projection.camera_frame_to_pixels(cam_frame_pc.points)
    # print(projected_points, projected_points.shape, cam_frame_pc.points.shape)
    # Use distance to color points, normalized to [0, 1].
    # colors = _color_by_distance(cam_frame_pc.points[:, 0], max_distance=30)
    # breakpoint()

    return projected_points
    # return _insert_into_image(rgb_frame.rgb, projected_points, colors, reduction_factor)


def project_flow_into_rgb(
    pc_frame: PointCloudFrame,
    flowed_pc_frame: PointCloudFrame,
    rgb_frame: RGBFrame,
    color_pose: PoseInfo,  # Pose used to compute color of flow
    reduction_factor: int,
) -> RGBImage:
    assert len(pc_frame.full_pc) == len(
        flowed_pc_frame.full_pc
    ), f"Pointclouds must be the same size, got {len(pc_frame.full_pc)} and {len(flowed_pc_frame.full_pc)}"

    # Ensure that all valid flowed_pc_frame points are valid in pc_frame
    assert np.all(
        pc_frame.mask & flowed_pc_frame.mask == flowed_pc_frame.mask
    ), f"Flow mask must be subset of pc mask but it's not"

    # Set the pc_frame mask to be the same as the flowed_pc_frame mask
    pc_frame.mask = flowed_pc_frame.mask

    assert len(pc_frame.pc) == len(
        flowed_pc_frame.pc
    ), f"Pointclouds must be the same size, got {len(pc_frame.pc)} and {len(flowed_pc_frame.pc)}"

    assert (
        pc_frame.pose == flowed_pc_frame.pose
    ), f"Poses must be the same, got {pc_frame.pose} and {flowed_pc_frame.pose}"

    pc_into_cam_frame_se3 = pc_frame.pose.sensor_to_ego.compose(
        rgb_frame.pose.sensor_to_ego.inverse()
    )

    cam_frame_pc = pc_frame.pc.transform(pc_into_cam_frame_se3)
    cam_frame_flowed_pc = flowed_pc_frame.pc.transform(pc_into_cam_frame_se3)

    in_front_of_cam_mask = cam_frame_pc.points[:, 0] >= 0

    # Don't use points behind the camera to describe flow.
    cam_frame_pc = PointCloud(cam_frame_pc.points[in_front_of_cam_mask])
    cam_frame_flowed_pc = PointCloud(cam_frame_flowed_pc.points[in_front_of_cam_mask])

    projected_points = rgb_frame.camera_projection.camera_frame_to_pixels(cam_frame_pc.points)
    

    # Convert the cam_frame_pc and cam_frame_flowed_pc to the color_pose frame
    cam_frame_to_color_pose_se3 = rgb_frame.pose.sensor_to_ego.compose(
        color_pose.sensor_to_ego.inverse()
    )
    cam_frame_pc = cam_frame_pc.transform(cam_frame_to_color_pose_se3)
    cam_frame_flowed_pc = cam_frame_flowed_pc.transform(cam_frame_to_color_pose_se3)
    flow_vectors = cam_frame_flowed_pc.points - cam_frame_pc.points

    flow_colors = _flow_to_rgb(flow_vectors) / 255.0
    blank_image = RGBImage.white_image_like(rgb_frame.rgb)
    return _insert_into_image(blank_image, projected_points, flow_colors, reduction_factor)


def visualize(
    frame_idx: int,
    frame: TimeSyncedSceneFlowFrame,
    save_dir: Optional[Path],
    mode: Mode,
    reduction_factor: int,
):
    pc_frame = frame.pc
    rgb_frames = frame.rgbs
    flow = frame.flow

    if mode == Mode.PROJECT_LIDAR:
        rgb_images = [
            project_lidar_into_rgb(pc_frame, rgb_frame, reduction_factor)
            for rgb_frame in rgb_frames.values()
        ]
    elif mode == Mode.PROJECT_FLOW:
        middle_frame = rgb_frames.values()[len(rgb_frames) // 2]
        rgb_images = [
            project_flow_into_rgb(
                pc_frame, pc_frame.flow(flow), rgb_frame, middle_frame.pose, reduction_factor
            )
            for rgb_frame in rgb_frames.values()
        ]

    for plot_idx, rgb_image in enumerate(rgb_images):
        plt.subplot(1, len(rgb_images), plot_idx + 1)
        plt.imshow(rgb_image.full_image)
        # Disable axis ticks
        plt.xticks([])
        plt.yticks([])
        # Set padding between subplots to 0
        plt.tight_layout(pad=0)
        # Get rid of black border
        # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        # Get rid of white space
        plt.margins(0)
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        # Set the background to be black
    fig = plt.gcf()
    fig.set_facecolor("black")

    if save_dir is None:
        plt.show()
    else:
        save_location = save_dir / f"{frame.log_id}" / f"{frame_idx:010d}.png"
        save_location.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_location, bbox_inches="tight", pad_inches=0, dpi=200)
        plt.clf()


if __name__ == "__main__":
    # Take arguments to specify dataset and root directory
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Argoverse2CausalSceneFlow")
    parser.add_argument("--root_dir", type=Path, default="/mnt/personal/vacekpa2/data/argoverse2/sensor/val")
    parser.add_argument("--flow_dir", type=Path, default="/mnt/personal/vacekpa2/data/argoverse2/sensor/val_sceneflow_feather")
    # use modes from the Mode enum
    parser.add_argument(
        "--mode", type=str, choices=[mode.value for mode in Mode], default=Mode.PROJECT_LIDAR.value
    )
    parser.add_argument("--save_dir", type=Path, default=None)
    parser.add_argument("--sequence_length", type=int, default=5)   # set number of frames
    parser.add_argument("--reduction_factor", type=int, default=1)
    args = parser.parse_args()

    dataset = construct_dataset(
        args.dataset,
        dict(
            root_dir=args.root_dir,
            with_rgb=True,
            flow_data_path=args.flow_dir,
            subsequence_length=args.sequence_length,
            use_gt_flow=False,
        ),
    )

    print("Dataset contains", len(dataset), "samples")
    mode = Mode(args.mode)

    if args.save_dir is not None:
        save_dir = (
            args.save_dir
            / f"{args.dataset}"
            / f"{args.root_dir.stem}"
            / f"{mode.value}"
            / f"{args.flow_dir.stem}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None

    assert len(dataset) > 0, "Dataset is empty"
    print("Dataset contains", len(dataset), "samples")
    
    # SEEM works only on gpu nodes (not amdgpu)!!!
    import sys
    from PIL import Image
    sys.path.append("/home/vacekpa2/models/SEEM/")
    from build_SEEM import SEEM
    Segmentor = SEEM(0)
    
    for vis_index in tqdm(range(len(dataset)),desc="Processing RGB images for SEEM masking"):
        
        # print("Loading sequence idx", vis_index)
        frame_list = dataset[vis_index] # frame, iterate
        # print(len(frame_list))
        if vis_index == 2: break

        for idx, frame in enumerate(frame_list):
            # print("IDX", idx, "SAVE_DIR", save_dir)
            sequence = frame.log_id
            ts = frame.log_timestamp

            pc_frame = frame.pc
            rgb_frames = frame.rgbs
            flow = frame.flow

            # add SAM
            # connect ids

            for rgb_frame in rgb_frames.values():
            #  make overall projection from all frame
                # rgb_image = (rgb_frame.rgb.full_image * 255).astype('uint8')
                rgb_image = rgb_frame.rgb.full_image
                
                x = Image.fromarray((rgb_image * 255).astype('uint8'))
                    
                pano_seg, instances, pano_seg_info = Segmentor(x)
                pano_seg = pano_seg.detach().cpu().numpy()
                # output_vis = Segmentor.visualize_image_predictions(x, pano_seg, pano_seg_info)

                projected_points = project_lidar_into_rgb(pc_frame, rgb_frame, args.reduction_factor)    

                # Suppress RuntimeWarning: invalid value encountered in cast
                with np.errstate(invalid="ignore"):
                    projected_points = projected_points.astype(np.int32)

                valid_points_mask = (
                    (projected_points[:, 0] >= 0)
                    & (projected_points[:, 0] < rgb_image.shape[1])
                    & (projected_points[:, 1] >= 0)
                    & (projected_points[:, 1] < rgb_image.shape[0])
                    # Here you can add another logic for filtration
                )   
                #
                
                projected_points = projected_points[valid_points_mask]
                # colors = colors[valid_points_mask]

                # scaled_rgb = rgb_image.rescale(reduction_factor)
                # scaled_projected_points = projected_points // reduction_factor
                projected_rgb_image = rgb_image.copy()
                
                valid_color = projected_rgb_image[projected_points[:, 1], projected_points[:, 0], :]
                seem_pano_class = pano_seg[projected_points[:, 1], projected_points[:, 0]]

                
                valid_points = pc_frame.full_pc.points[valid_points_mask]

        
            tmp_store = np.concatenate((valid_points, seem_pano_class[:, None]), axis=1)
            np.save('tmp_store.npy', tmp_store)
            

    breakpoint()
    