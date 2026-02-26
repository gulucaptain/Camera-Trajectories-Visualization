import os
import numpy as np
import copy
import subprocess

import plotly.graph_objs as go
import plotly.io as pio

from CamVisual.CameraPose import Pose

def to_hom(X):
    """Get homogeneous coordinates of the input by appending ones."""
    X_hom = np.concatenate([X, np.ones_like(X[..., :1])], axis=-1)
    return X_hom

def cam2world(X, pose):
    """Transform points from camera to world coordinates."""
    X_hom = to_hom(X)
    pose_inv = Pose().invert(pose)
    return X_hom @ pose_inv.transpose(0, 2, 1)

def unbind_np(array, axis=0):
    """Split numpy array along specified axis into list."""
    if axis == 0:
        return [array[i, :] for i in range(array.shape[0])]
    elif axis == 1 or (len(array.shape) == 2 and axis == -1):
        return [array[:, j] for j in range(array.shape[1])]
    elif axis == 2 or (len(array.shape) == 3 and axis == -1):
        return [array[:, :, j] for j in range(array.shape[2])]
    else:
        raise ValueError("Invalid axis. Use 0 for rows or 1 for columns.")
    
def merge_wireframes_plotly(wireframe):
    """Merge camera wireframes for plotly visualization."""
    wf_dummy = wireframe[:, :1] * np.nan
    wireframe_merged = np.concatenate([wireframe, wf_dummy], axis=1).reshape(-1, 3)
    return wireframe_merged

def merge_meshes(vertices, faces):
    """Merge multiple camera meshes into a single mesh."""
    mesh_N, vertex_N = vertices.shape[:2]
    faces_merged = np.concatenate([faces + i * vertex_N for i in range(mesh_N)], axis=0)
    vertices_merged = vertices.reshape(-1, vertices.shape[-1])
    return vertices_merged, faces_merged

def get_camera_mesh(pose, depth=1):
    """
    Create a camera mesh visualization.
    
    Args:
        pose: Camera pose matrix
        depth: Size of the camera frustum

    Returns:
        vertices: Camera mesh vertices
        faces: Camera mesh faces
        wireframe: Camera wireframe vertices
    """
    vertices = np.array([[-0.5, -0.5, 1],
                         [0.5, -0.5, 1],
                         [0.5, 0.5, 1],
                         [-0.5, 0.5, 1],
                         [0, 0, 0]]) * depth  # [6,3]
    faces = np.array([[0, 1, 2],
                      [0, 2, 3],
                      [0, 1, 4],
                      [1, 2, 4],
                      [2, 3, 4],
                      [3, 0, 4]])  # [6,3]
    vertices = cam2world(vertices[None], pose)  # [N,6,3]
    wireframe = vertices[:, [0, 1, 2, 3, 0, 4, 1, 2, 4, 3]]  # [N,10,3]
    return vertices, faces, wireframe

def merge_xyz_indicators_plotly(xyz):
    """Merge xyz coordinate indicators for plotly visualization."""
    xyz = xyz[:, [[-1, 0], [-1, 1], [-1, 2]]]  # [N,3,2,3]
    xyz_0, xyz_1 = unbind_np(xyz, axis=2)  # [N,3,3]
    xyz_dummy = xyz_0 * np.nan
    xyz_merged = np.stack([xyz_0, xyz_1, xyz_dummy], axis=2)  # [N,3,3,3]
    xyz_merged = xyz_merged.reshape(-1, 3)
    return xyz_merged

def get_xyz_indicators(pose, length=0.1):
    """Get xyz coordinate axis indicators for a camera pose."""
    xyz = np.eye(4, 3)[None] * length
    xyz = cam2world(xyz, pose)
    return xyz


def get_world_axes_traces(centers_world, axis_len=None, axis_width=6, show_text=True):
    """
    centers_world: [N,3] Camera center point (world coordinate system)
    axis_len: World axis length (None will adapt to the path automatically)
    """
    centers_world = np.asarray(centers_world)

    if axis_len is None:
        # Track scale: Estimate of diagonal length of the bounding box
        mn = np.nanmin(centers_world, axis=0)
        mx = np.nanmax(centers_world, axis=0)
        diag = float(np.linalg.norm(mx - mn))
        axis_len = max(diag * 0.35, 1e-3)

    O = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    X = np.array([axis_len, 0.0, 0.0], dtype=np.float32)
    Y = np.array([0.0, axis_len, 0.0], dtype=np.float32)
    Z = np.array([0.0, 0.0, axis_len], dtype=np.float32)

    traces = []

    # Three world axes (red/green/blue)
    traces += [
        go.Scatter3d(x=[O[0], X[0]], y=[O[1], X[1]], z=[O[2], X[2]],
                     mode="lines", name="World X",
                     line=dict(color="rgb(255,0,0)", width=axis_width)),
        go.Scatter3d(x=[O[0], Y[0]], y=[O[1], Y[1]], z=[O[2], Y[2]],
                     mode="lines", name="World Y",
                     line=dict(color="rgb(0,255,0)", width=axis_width)),
        go.Scatter3d(x=[O[0], Z[0]], y=[O[1], Z[1]], z=[O[2], Z[2]],
                     mode="lines", name="World Z",
                     line=dict(color="rgb(0,0,255)", width=axis_width)),
    ]

    # axis_len: adaptive parameter; 
    # larger value means smaller arraw;
    sizeref = axis_len * 0.12  
    traces += [
        go.Cone(x=[X[0]], y=[X[1]], z=[X[2]], u=[1], v=[0], w=[0],
                sizemode="absolute", sizeref=sizeref,
                showscale=False, colorscale=[[0, "rgb(255,0,0)"], [1, "rgb(255,0,0)"]],
                name="X arrow"),
        go.Cone(x=[Y[0]], y=[Y[1]], z=[Y[2]], u=[0], v=[1], w=[0],
                sizemode="absolute", sizeref=sizeref,
                showscale=False, colorscale=[[0, "rgb(0,255,0)"], [1, "rgb(0,255,0)"]],
                name="Y arrow"),
        go.Cone(x=[Z[0]], y=[Z[1]], z=[Z[2]], u=[0], v=[0], w=[1],
                sizemode="absolute", sizeref=sizeref,
                showscale=False, colorscale=[[0, "rgb(0,0,255)"], [1, "rgb(0,0,255)"]],
                name="Z arrow"),
    ]

    if show_text:
        traces += [
            go.Scatter3d(x=[X[0]], y=[X[1]], z=[X[2]], mode="text", text=[""],
                         textfont=dict(color="rgb(255,0,0)", size=16), showlegend=False),
            go.Scatter3d(x=[Y[0]], y=[Y[1]], z=[Y[2]], mode="text", text=[""],
                         textfont=dict(color="rgb(0,255,0)", size=16), showlegend=False),
            go.Scatter3d(x=[Z[0]], y=[Z[1]], z=[Z[2]], mode="text", text=[""],
                         textfont=dict(color="rgb(0,0,255)", size=16), showlegend=False),
            go.Scatter3d(x=[0], y=[0], z=[0], mode="markers+text", text=["O"],
                         textposition="top center",
                         marker=dict(size=4, color="black"),
                         textfont=dict(color="black", size=14),
                         showlegend=False),
        ]

    return traces


def plotly_animation_to_mp4(fig, out_mp4, fps=25, width=800, height=500, scale=2, tmp_dir="./_frames"):
    os.makedirs(tmp_dir, exist_ok=True)

    frames = getattr(fig, "frames", None)
    if not frames or len(frames) == 0:
        raise ValueError("fig.frames 为空：当前 figure 没有可导出的动画帧。")

    # 基准：静态 data + layout
    base_data = [copy.deepcopy(tr) for tr in fig.data]
    base_layout = copy.deepcopy(fig.layout)

    for i, fr in enumerate(frames):
        # 1) 从 base_data 复制一份作为本帧 data
        frame_data = [copy.deepcopy(tr) for tr in base_data]

        # 2) 将 frame.data 写回到正确的 trace index 上
        #    - 如果 fr.traces 存在：按指定索引更新
        #    - 否则：默认从 0..len(fr.data)-1 依次覆盖
        if fr.data is not None:
            if getattr(fr, "traces", None) is not None and len(fr.traces) > 0:
                for j, idx in enumerate(fr.traces):
                    frame_data[idx] = copy.deepcopy(fr.data[j])
            else:
                for j in range(min(len(fr.data), len(frame_data))):
                    frame_data[j] = copy.deepcopy(fr.data[j])

        # 3) layout：base_layout + frame.layout 叠加
        frame_layout = copy.deepcopy(base_layout)
        if fr.layout is not None:
            # 用 update 叠加帧 layout（scene.camera 等通常在这里）
            frame_layout.update(fr.layout)

        # 4) 构造“新 figure”，导出 png
        frame_fig = go.Figure(data=frame_data, layout=frame_layout)
        png_path = os.path.join(tmp_dir, f"frame_{i:05d}.png")
        png_bytes = pio.to_image(frame_fig, format="png", width=width, height=height, scale=scale)
        with open(png_path, "wb") as f:
            f.write(png_bytes)

    # 5) ffmpeg 合成 mp4
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(tmp_dir, "frame_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        out_mp4
    ]
    subprocess.run(cmd, check=True)
    return out_mp4
