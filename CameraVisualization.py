import os
import argparse
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import plotly.graph_objs as go
import plotly.io as pio
import colorsys

from tqdm import tqdm

matplotlib.use("agg")

from CamVisual.load_camera_parameters import load_camera_parameters

from CamVisual.utils import (
    cam2world,
    unbind_np,
    merge_wireframes_plotly,
    merge_meshes,
    get_camera_mesh,
    merge_xyz_indicators_plotly,
    get_xyz_indicators,
    get_world_axes_traces,
    plotly_animation_to_mp4
)

from CamVisual.enhancement import (
    rescale_extrinsics_t_auto_numpy,
    densify_extrinsics
)


def plotly_visualize_pose_and_img(
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
    return_fig=False,
    animate_cube=True,
    cube_size=6,
    fps=25
):
    # ===== calculate centers_world / traces =====
    N = len(poses)
    centers_cam = np.zeros([N, 1, 3])
    centers_world = cam2world(centers_cam, poses)
    centers_world = centers_world[:, 0]

    # vertices, faces, wireframe = get_camera_mesh(poses, depth=vis_depth)

    centers = centers_world  # (N,3);
    # === Adjust vis_depth with the scale of traj === 
    mins = centers.min(axis=0)
    maxs = centers.max(axis=0)
    span = float((maxs - mins).max())
    span = max(span, 1e-6) # Trajectory scale

    vis_depth_auto = span * 0.08
    vis_depth = vis_depth_auto

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

    base_traces = [
        go.Scatter3d(
            x=wireframe_x, y=wireframe_y, z=wireframe_z,
            mode="lines", line=dict(color=wireframe_color, width=1),
            name="camera_frustum"
        ),
        go.Scatter3d(
            x=xyz_x, y=xyz_y, z=xyz_z,
            mode="lines", line=dict(color=xyz_color, width=xyz_width),
            name="camera_axes"
        ),
        go.Scatter3d(
            x=centers_x, y=centers_y, z=centers_z,
            mode="markers", marker=dict(color=center_color, size=center_size, opacity=1),
            name="centers"
        ),
        go.Mesh3d(
            x=vertices_x, y=vertices_y, z=vertices_z,
            i=[f[0] for f in faces_merged],
            j=[f[1] for f in faces_merged],
            k=[f[2] for f in faces_merged],
            facecolor=faces_merged_color, opacity=mesh_opacity,
            name="mesh"
        ),
    ]

    base_traces += get_world_axes_traces(centers_world, axis_len=None, axis_width=6, show_text=True)

    # ===== 动画小方块（初始位置）+ 走过的轨迹 trail =====
    frames = []
    cube_trace = None
    trail_trace = None

    # ===== 为小方块准备渐变颜色 =====
    cube_colors = []
    for i in range(N):
        r, g, b, _ = color_map(i / max(N - 1, 1))
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
        cube_colors.append(f"rgb({r},{g},{b})")

    frames = []
    if animate_cube:
        frustum_colors = []
        for i in range(N):
            r, g, b, _ = color_map(i / max(N - 1, 1))
            h, s, v = colorsys.rgb_to_hsv(r, g, b) # RGB -> HSV
            s = min(1.0, s * 1.9)
            v = v * 0.5
            r, g, b = colorsys.hsv_to_rgb(h, s, v) # HSV -> RGB
            r, g, b = int(r * 255), int(g * 255), int(b * 255)
            frustum_colors.append(f"rgb({r},{g},{b})")

        # 当前帧的视锥线框
        wf0 = wireframe[0]  # (10,3)
        moving_frustum = go.Scatter3d(
            x=wf0[:, 0], y=wf0[:, 1], z=wf0[:, 2],
            mode="lines",
            line=dict(color=frustum_colors[0], width=6),
            name="moving_frustum"
        )

        # 走过的 trail
        trail = go.Scatter3d(
            x=[centers_world[0, 0]],
            y=[centers_world[0, 1]],
            z=[centers_world[0, 2]],
            mode="lines",
            line=dict(color="rgba(0,0,0,0.18)", width=6),
            name="trail"
        )

        # 帧：只更新 moving_frustum + trail（不重复 base_traces）
        for i in range(N):
            wf = wireframe[i]
            frames.append(
                go.Frame(
                    name=str(i),
                    data=[
                        go.Scatter3d(
                            x=wf[:, 0], y=wf[:, 1], z=wf[:, 2],
                            mode="lines",
                            line=dict(color=frustum_colors[i], width=6),
                        ),
                        go.Scatter3d(
                            x=centers_world[:i+1, 0],
                            y=centers_world[:i+1, 1],
                            z=centers_world[:i+1, 2],
                            mode="lines",
                            line=dict(color="rgba(0,0,0,0.18)", width=6),
                        ),
                    ],
                )
            )


    # ===== fig =====
    layout = go.Layout(
        scene=dict(
            xaxis=dict(visible=True, title="X", showgrid=True, zeroline=True, showticklabels=True),
            yaxis=dict(visible=True, title="Y", showgrid=True, zeroline=True, showticklabels=True),
            zaxis=dict(visible=True, title="Z", showgrid=True, zeroline=True, showticklabels=True),
            dragmode="orbit",
            aspectmode="data",
        ),
        height=img_height,
        width=img_width,
        showlegend=False,
    )

    data = base_traces.copy()
    if animate_cube:
        # data += [cube_trace, trail_trace]
        data += [moving_frustum, trail]

        # slider + play/pause
        frame_duration_ms = int(1000 / max(fps, 1))
        layout.update(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    x=0.02, y=0.98,
                    xanchor="left", yanchor="top",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=frame_duration_ms, redraw=True),
                                    transition=dict(duration=0),
                                    fromcurrent=True,
                                    mode="immediate",
                                ),
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(frame=dict(duration=0, redraw=False), mode="immediate"),
                            ],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    x=0.1, y=0.02, len=0.85,
                    currentvalue=dict(prefix="Frame: "),
                    steps=[
                        dict(
                            method="animate",
                            label=str(i),
                            args=[
                                [str(i)],
                                dict(
                                    frame=dict(duration=0, redraw=True),
                                    transition=dict(duration=0),
                                    mode="immediate",
                                ),
                            ],
                        )
                        for i in range(N)
                    ],
                )
            ],
        )

    fig = go.Figure(data=data, layout=layout, frames=frames)

    if save_img_path is not None:
        try:
            pio.write_image(fig, save_img_path, width=img_width, height=img_height, scale=img_scale)
        except Exception as e:
            raise RuntimeError(
                "Plotly 导出静态图片需要 Kaleido 或 Orca 作为渲染后端。原始错误：\n" + str(e)
            )

    return (data, fig) if return_fig else data


def write_html(
        poses, 
        file, 
        dset, 
        vis_depth=1, 
        xyz_length=0.2, 
        center_size=0.01, 
        xyz_width=2, 
        saved_root=None, 
        saved_img_pth=None,
        output_video=False
    ):

    if dset == "lightspeed":
        xyz_length = xyz_length / 3
        vis_depth = vis_depth / 3
        center_size *= 3

    traces_poses, fig = plotly_visualize_pose_and_img(
        poses,
        vis_depth=vis_depth,
        xyz_length=xyz_length,
        center_size=center_size,
        xyz_width=xyz_width,
        mesh_opacity=0.05,
        save_img_path=saved_img_pth,
        img_width=800,
        img_height=500,
        img_scale=2,
        return_fig=True,
        animate_cube=True,
        cube_size=7,
        fps=25
    )
    
    if output_video:
        saved_video_name = saved_img_pth.split("/")[-1].replace(".jpg", ".mp4")
        saved_video_pth = f"{saved_root}/{saved_video_name}"
        out_mp4 = plotly_animation_to_mp4(fig, saved_video_pth, fps=25, width=800, height=500, scale=2)
        print("saved:", out_mp4)
    
    html_str2 = pio.to_html(fig, full_html=False)  # 或 include_plotlyjs='cdn'
    file.write(html_str2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset", type=str, default="self-data", choices=["self-data", "lightspeed"])
    parser.add_argument("--exp_file_root", type=str, default="./assets")
    parser.add_argument("--camera_file_type", type=str, default="txt", help="txt or pt file")
    parser.add_argument("--project_name", type=str, default=None, help="Saved project name.")
    parser.add_argument("--split_size", type=int, default=6, help="to avoid poses disappearing.")
    parser.add_argument("--extrinsic_enhanced", type=bool, default=True, help="make visualization more stronger.")
    parser.add_argument("--output_root", type=str, default=None, required=True, help="root path for saving results.")
    args = parser.parse_args()

    split_size = args.split_size
    
    camera_pts = []
    exp_file_root = args.exp_file_root
    exp_dir_name = os.path.basename(exp_file_root)

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

    assert len(camera_pts) != 0

    saved_html_root = f"{args.output_root}/saved_html_{args.project_name}"
    saved_img_root = f"{args.output_root}/saved_imgs_{args.project_name}"
    os.makedirs(saved_html_root, exist_ok=True)
    os.makedirs(saved_img_root, exist_ok=True)

    for j in tqdm(range(int(np.ceil(len(camera_pts)/split_size)))):
        with open(f"{saved_html_root}/index_{str(j)}.html", "w") as file:
            for i, seq in enumerate(tqdm(camera_pts[j*split_size:j*split_size+split_size])):
                for key in seq.keys():
                    case_name = seq[key].split("/")[-1].replace(".txt", "")
                    extrinsic_file = seq[key]
                
                saved_img_pth = os.path.join(saved_img_root, f"{case_name}.jpg")

                file.write(f"<span style='font-size: 18pt;'>{i} {case_name}</span><br>")
                extrinsic = load_camera_parameters(extrinsic_file)
                
                if args.extrinsic_enhanced:
                    extrinsic, scale = rescale_extrinsics_t_auto_numpy(extrinsic)
                    extrinsic = densify_extrinsics(extrinsic, inserts_per_gap=3)
                
                write_html(extrinsic, file, args.dset, saved_root=args.output_root, saved_img_pth=saved_img_pth, output_video=False)
                # when set output_video=True, you will get a dynamic trajectory moving video, but it will cost lots of time!

                with open(f"{args.output_root}/info.txt", "a") as f: # help you to locate each case.
                    f.writelines(f"{case_name}#####{j}#####{extrinsic_file}\n")
