import numpy as np
import torch

def load_camera_parameters(extrinsic_file, camera_trajectory_truncated=None):
    if extrinsic_file.endswith(".pt"):
        device = torch.device("cpu")
        extrinsics = torch.load(extrinsic_file, map_location=device, weights_only=False)
        
        assert extrinsics.ndim == 3 and extrinsics.shape[1] == 3 and extrinsics.shape[2] == 4, \
            f"Extrinsic shape must be (F, 3, 4), but got {extrinsics.shape}"
        
        if camera_trajectory_truncated is not None:
            # extrinsic = extrinsic[:13, :, :]
            extrinsics = extrinsics[:camera_trajectory_truncated, ...]
        extrinsics = extrinsics.numpy()
    elif extrinsic_file.endswith(".txt"):
        extrinsics = []
        with open(extrinsic_file, "r") as f:
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
                extrinsic_values = values[7 : 19]  # 12 Number
                extrinsic = np.array(extrinsic_values).reshape(3, 4)
                extrinsics.append(extrinsic)

        extrinsics = np.stack(extrinsics, axis=0)  # (N, 3, 4)
        # extrinsics = extrinsics[:13, :, :] # Truncate
    
    # print(f"Loaded finished! The extrinsic' shape is: {extrinsic.shape}")
    return extrinsics