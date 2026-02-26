import os
import argparse
import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt

import plotly.graph_objs as go
import plotly.io as pio

from tqdm import tqdm

from CamVisual.enhancement import (
    rescale_extrinsics_t_auto_numpy,
    densify_extrinsics
)

from CamVisual.utils import (
    cam2world,
    unbind_np,
    merge_wireframes_plotly,
    merge_meshes,
    get_camera_mesh,
    merge_xyz_indicators_plotly,
    get_xyz_indicators,
    get_world_axes_traces
)

matplotlib.use("agg")

def plotly_visualize_pose(
    poses,
    vis_depth=0.5,
    xyz_length=0.5,
    center_size=2,
    xyz_width=5,
    mesh_opacity=0.05,
    save_img_path=None,
    img_width=800,
    img_height=500,
    img_scale=2,
    return_fig=False
):
    N = len(poses)
    centers_cam = np.zeros([N, 1, 3])
    centers_world = cam2world(centers_cam, poses)
    centers_world = centers_world[:, 0]

    vertices, faces, wireframe = get_camera_mesh(poses, depth=vis_depth)
    xyz = get_xyz_indicators(poses, length=xyz_length)

    vertices_merged, faces_merged = merge_meshes(vertices, faces)
    wireframe_merged = merge_wireframes_plotly(wireframe)
    xyz_merged = merge_xyz_indicators_plotly(xyz)

    wireframe_x, wireframe_y, wireframe_z = unbind_np(wireframe_merged, axis=-1)
    xyz_x, xyz_y, xyz_z = unbind_np(xyz_merged, axis=-1)
    centers_x, centers_y, centers_z = unbind_np(centers_world, axis=-1)
    vertices_x, vertices_y, vertices_z = unbind_np(vertices_merged, axis=-1)

    color_map = plt.get_cmap("gist_rainbow")
    center_color, faces_merged_color, wireframe_color, xyz_color = [], [], [], []
    x_color, y_color, z_color = *np.eye(3).T,

    for i in range(N):
        r, g, b, _ = color_map(i / max(N - 1, 1))
        rgb = np.array([r, g, b]) * 0.8
        wireframe_color += [rgb] * 11
        center_color += [rgb]
        faces_merged_color += [rgb] * 6
        xyz_color += [x_color] * 3 + [y_color] * 3 + [z_color] * 3

    plotly_traces = [
        go.Scatter3d(x=wireframe_x, y=wireframe_y, z=wireframe_z, mode="lines",
                     line=dict(color=wireframe_color, width=1)),
        go.Scatter3d(x=xyz_x, y=xyz_y, z=xyz_z, mode="lines",
                     line=dict(color=xyz_color, width=xyz_width)),
        go.Scatter3d(x=centers_x, y=centers_y, z=centers_z, mode="markers",
                     marker=dict(color=center_color, size=center_size, opacity=1)),
        go.Mesh3d(x=vertices_x, y=vertices_y, z=vertices_z,
                  i=[f[0] for f in faces_merged],
                  j=[f[1] for f in faces_merged],
                  k=[f[2] for f in faces_merged],
                  facecolor=faces_merged_color, opacity=mesh_opacity),
    ]
    plotly_traces += get_world_axes_traces(centers_world, axis_len=None, axis_width=6, show_text=True)

    # ===== 保存“默认视角”图像 =====
    fig = None
    if save_img_path is not None or return_fig:
        layout = go.Layout(
            scene=dict(
                xaxis=dict(visible=True, title="X", showgrid=True, zeroline=True, showticklabels=True),
                yaxis=dict(visible=True, title="Y", showgrid=True, zeroline=True, showticklabels=True),
                zaxis=dict(visible=True, title="Z", showgrid=True, zeroline=True, showticklabels=True),
                dragmode="orbit",
                aspectmode="data"
                # Plotly 默认视角：不设置 scene.camera
            ),
            height=img_height,
            width=img_width,
            showlegend=False
        )
        fig = go.Figure(data=plotly_traces, layout=layout)

        if save_img_path is not None:
            try:
                pio.write_image(fig, save_img_path, width=img_width, height=img_height, scale=img_scale)
            except Exception as e:
                raise RuntimeError(
                    "Plotly 导出静态图片需要 Kaleido 或 Orca 作为渲染后端。 原始错误：\n" + str(e)
                )

    return (plotly_traces, fig) if return_fig else plotly_traces


def write_html(poses, file, dset, vis_depth=1, xyz_length=0.2, center_size=0.01, xyz_width=2, saved_img_pth=None):
    """Write camera pose visualization to HTML file."""
    if dset == "lightspeed":
        xyz_length = xyz_length / 3
        xyz_width = xyz_width
        vis_depth = vis_depth / 3
        center_size *= 3

    traces_poses = plotly_visualize_pose(
        poses,
        vis_depth=vis_depth,
        xyz_length=xyz_length,
        center_size=center_size,
        xyz_width=xyz_width,
        mesh_opacity=0.05,
        save_img_path=saved_img_pth,
        img_width=800,
        img_height=500,
        img_scale=2
    )

    traces_all2 = traces_poses
    
    layout2 = go.Layout(
        scene=dict(
            xaxis=dict(visible=True, title="X", showgrid=True, zeroline=True, showticklabels=True),
            yaxis=dict(visible=True, title="Y", showgrid=True, zeroline=True, showticklabels=True),
            zaxis=dict(visible=True, title="Z", showgrid=True, zeroline=True, showticklabels=True),
            dragmode="orbit",
            aspectmode="data"
        ),
        height=500,
        width=800,
        showlegend=False
    )

    fig2 = go.Figure(data=traces_all2, layout=layout2)
    html_str2 = pio.to_html(fig2, full_html=False)

    file.write(html_str2)


def load_extrinsics_from_txt(txt_path):
    extrinsics = []

    with open(txt_path, "r") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            values = list(map(float, line.split()))

            if len(values) < 19:
                raise ValueError(
                    f"Line {line_idx} has insufficient values: {len(values)}"
                )

            # Skip：frame_id + 6 intrinsics
            extrinsic_values = values[7:19]  # 12 numbers
            extrinsic = np.array(extrinsic_values).reshape(3, 4)
            extrinsics.append(extrinsic)

    extrinsics = np.stack(extrinsics, axis=0)  # (N, 3, 4)

    return extrinsics

def viz_poses_using_explict_parameters(i, extrinsic_file, case_name, file, dset, saved_img_pth, extrinsic_enhanced=False):
    file.write(f"<span style='font-size: 18pt;'>{i} {case_name}</span><br>")

    if extrinsic_file.endswith(".pt"):
        device = torch.device("cpu")
        extrinsic = torch.load(extrinsic_file, map_location=device, weights_only=False)
        extrinsic = extrinsic.numpy()
    elif extrinsic_file.endswith(".txt"):
        extrinsic = load_extrinsics_from_txt(extrinsic_file)
        
    if extrinsic_enhanced:
        extrinsic, scale = rescale_extrinsics_t_auto_numpy(extrinsic)
        extrinsic = densify_extrinsics(extrinsic, inserts_per_gap=7)
    
    write_html(extrinsic, file, dset, saved_img_pth=saved_img_pth)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset", type=str, default="self-data", choices=["self-data", "lightspeed"])
    parser.add_argument("--exp_file_root", type=str, default="./assets")
    parser.add_argument("--camera_file_type", type=str, default="txt", help="txt or pt file")
    parser.add_argument("--project_name", type=str, default=None, help="Saved project name.")
    parser.add_argument("--split_size", type=int, default=6, help="to avoid poses disappearing.")
    parser.add_argument("--output_root", type=str, default=None, required=True, help="root path for saving results.")
    args = parser.parse_args()

    split_size = args.split_size
    
    camera_pts = []
    exp_file_root = args.exp_file_root
    exp_dir_name = exp_file_root.split("/")[-1]

    exp_file_dirs = os.listdir(exp_file_root)
    exp_file_dirs.sort()
    for exp_file_dir in exp_file_dirs:
        files = os.listdir(os.path.join(exp_file_root, exp_file_dir))
        for file in files:
            if args.camera_file_type == "txt":
                if ".txt" in file:
                    camera_pts.append(
                        {
                            exp_file_dir: os.path.join(exp_file_root, exp_file_dir, file)
                        }
                    )
            elif args.camera_file_type == "pt":
                if ".pt" in file:
                    camera_pts.append(
                        {
                            exp_file_dir: os.path.join(exp_file_root, exp_file_dir, file)
                        }
                    )
    
    saved_html_root = f"{args.output_root}/saved_html_{args.project_name}"
    saved_img_root = f"{args.output_root}/saved_imgs_{args.project_name}"
    os.makedirs(saved_html_root, exist_ok=True)
    os.makedirs(saved_img_root, exist_ok=True)

    for j in tqdm(range(int(np.ceil(len(camera_pts)/split_size)))):
        with open(f"{saved_html_root}/index_{str(j)}.html", "w") as file:
            for i, seq in enumerate(tqdm(camera_pts[j*split_size:j*split_size+split_size])):
                for key in seq.keys():
                    case_name = key
                    extrinsic_file = seq[key]
                
                saved_img_pth = os.path.join(saved_img_root, f"{case_name}.jpg")

                viz_poses_using_explict_parameters(i, extrinsic_file, case_name, file, args.dset, saved_img_pth)

                with open(f"{args.output_root}/info.txt", "a") as f:
                    f.writelines(f"{case_name}#####{extrinsic_file}#####{j}\n")
