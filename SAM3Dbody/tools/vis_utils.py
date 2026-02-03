# Copyright (c) Meta Platforms, Inc. and affiliates.
import cv2
import matplotlib.pyplot as plt
import numpy as np
from ..sam_3d_body.visualization.renderer import Renderer
from ..sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
from ..sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

visualizer = SkeletonVisualizer(line_width=2, radius=5)
visualizer.set_pose_meta(mhr70_pose_info)


def visualize_2d_results(img_cv2, outputs, visualizer: SkeletonVisualizer):
    """Visualize 2D keypoints and bounding boxes."""
    results = []

    for pid, person_output in enumerate(outputs):
        img_vis = img_cv2.copy()
        keypoints_2d = person_output["pred_keypoints_2d"]
        keypoints_2d_vis = np.concatenate(
            [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
        )
        img_vis = visualizer.draw_skeleton(img_vis, keypoints_2d_vis)

        bbox = person_output["bbox"]
        img_vis = cv2.rectangle(
            img_vis,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            img_vis,
            f"Person {pid}",
            (int(bbox[0]), int(bbox[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        results.append(img_vis)

    return results


def visualize_3d_mesh(img_cv2, outputs, faces: np.ndarray):
    """Visualize 3D mesh overlaid on image and side view."""
    results = []

    for person_output in outputs:
        renderer = Renderer(focal_length=person_output["focal_length"], faces=faces)
        img_orig = img_cv2.copy()

        img_mesh_overlay = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                img_cv2.copy(),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        ).astype(np.uint8)

        white_img = np.ones_like(img_cv2) * 255
        img_mesh_white = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                white_img,
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        ).astype(np.uint8)

        img_mesh_side = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                white_img.copy(),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                side_view=True,
            )
            * 255
        ).astype(np.uint8)

        combined = np.concatenate(
            [img_orig, img_mesh_overlay, img_mesh_white, img_mesh_side], axis=1
        )
        results.append(combined)

    return results


def visualize_3d_skeleton(img_cv2, outputs, visualizer: SkeletonVisualizer):
    """3D skeleton plotting; returns RGB image array."""
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    kpt_colors = (
        np.array(visualizer.kpt_color, dtype=np.float32) / 255.0
        if visualizer.kpt_color is not None
        else None
    )
    link_colors = (
        np.array(visualizer.link_color, dtype=np.float32) / 255.0
        if visualizer.link_color is not None
        else None
    )

    has_data = False
    all_points = []
    for target in outputs:
        pts = target.get("pred_keypoints_3d")
        if pts is not None:
            all_points.append(pts.reshape(-1, 3))

    if all_points:
        has_data = True
        all_points_np = np.concatenate(all_points, axis=0)
        max_range = (all_points_np.max(axis=0) - all_points_np.min(axis=0)).max() / 2.0
        mid = (all_points_np.max(axis=0) + all_points_np.min(axis=0)) / 2.0
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        for target in outputs:
            pts_3d = target.get("pred_keypoints_3d")
            if pts_3d is None:
                continue
            if pts_3d.ndim == 3:
                pts_3d = pts_3d[0]

            ax.scatter(
                pts_3d[:, 0],
                pts_3d[:, 1],
                pts_3d[:, 2],
                c=kpt_colors if kpt_colors is not None else "r",
                s=visualizer.radius * 5,
                alpha=getattr(visualizer, "alpha", 0.8),
            )

            if visualizer.skeleton is not None:
                for j, (p1_idx, p2_idx) in enumerate(visualizer.skeleton):
                    if p1_idx < len(pts_3d) and p2_idx < len(pts_3d):
                        p1, p2 = pts_3d[p1_idx], pts_3d[p2_idx]
                        color = (
                            link_colors[j % len(link_colors)]
                            if link_colors is not None
                            else "b"
                        )
                        ax.plot(
                            [p1[0], p2[0]],
                            [p1[1], p2[1]],
                            [p1[2], p2[2]],
                            color=color,
                            linewidth=visualizer.line_width,
                            alpha=getattr(visualizer, "alpha", 0.8),
                        )

    if not has_data:
        ax.text(0.5, 0.5, 0.5, "No Data", ha="center")

    ax.view_init(elev=-30, azim=270)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img_3d = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img_3d = img_3d.reshape((h, w, 4))[:, :, :3]
    plt.close(fig)
    return img_3d


def visualize_sample_together(img_cv2, outputs, faces, visualizer: SkeletonVisualizer):
    # Render everything together
    img_keypoints = img_cv2.copy()
    img_mesh = img_cv2.copy()

    # First, sort by depth, furthest to closest
    all_depths = np.stack([tmp["pred_cam_t"] for tmp in outputs], axis=0)[:, 2]
    outputs_sorted = [outputs[idx] for idx in np.argsort(-all_depths)]

    # Then, draw all keypoints.
    for pid, person_output in enumerate(outputs_sorted):
        keypoints_2d = person_output["pred_keypoints_2d"]
        keypoints_2d = np.concatenate(
            [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
        )
        img_keypoints = visualizer.draw_skeleton(img_keypoints, keypoints_2d)

    # Then, put all meshes together as one super mesh
    all_pred_vertices = []
    all_faces = []
    for pid, person_output in enumerate(outputs_sorted):
        all_pred_vertices.append(
            person_output["pred_vertices"] + person_output["pred_cam_t"]
        )
        all_faces.append(faces + len(person_output["pred_vertices"]) * pid)
    all_pred_vertices = np.concatenate(all_pred_vertices, axis=0)
    all_faces = np.concatenate(all_faces, axis=0)

    # Pull out a fake translation; take the closest two
    fake_pred_cam_t = (
        np.max(all_pred_vertices[-2 * 18439 :], axis=0)
        + np.min(all_pred_vertices[-2 * 18439 :], axis=0)
    ) / 2
    all_pred_vertices = all_pred_vertices - fake_pred_cam_t

    # Render front view
    renderer = Renderer(focal_length=person_output["focal_length"], faces=all_faces)
    img_mesh = (
        renderer(
            all_pred_vertices,
            fake_pred_cam_t,
            img_mesh,
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
        )
        * 255
    )

    # Render side view
    white_img = np.ones_like(img_cv2) * 255
    img_mesh_side = (
        renderer(
            all_pred_vertices,
            fake_pred_cam_t,
            white_img,
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
            side_view=True,
        )
        * 255
    )

    cur_img = np.concatenate([img_cv2, img_keypoints, img_mesh, img_mesh_side], axis=1)

    return cur_img
