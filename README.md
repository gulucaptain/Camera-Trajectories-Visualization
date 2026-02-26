# Camera-Trajectories-Visualization
Camera trajectories visualization with dynamic results in 3D mesh.

## How to visualize the camera trajectories?

In "assets" folder, we put several camera parameters with ".txt" files, using our code, you can visualize these camera parameters in 3D mesh and observe how the trajectories moving.

You can run this script to obtain dynamic results:

```bash
python CameraVisualization.py \
    --dset="self-data" \
    --exp_file_root="./assets" \
    --camera_file_type="txt" \
    --project_name="test_dynamic" \
    --split_size=6 \
    --output_root="./outputs"
```

Or you can use the following script to obtain static results:

```bash
python CameraVisualization_static.py \
    --dset="self-data" \
    --exp_file_root="./assets" \
    --camera_file_type="txt" \
    --project_name="test_static" \
    --split_size=6 \
    --output_root="./outputs2"
```

Then, you will get the results that shown in "outputs" or "outputs2" folder.
