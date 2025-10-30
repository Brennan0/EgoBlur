# EgoBlur Demo
This repository contains demo of [EgoBlur models](https://www.projectaria.com/tools/egoblur) with visualizations.


## Installation

This code requires `conda>=23.1.0` to install dependencies and create a virtual environment to execute the code in. Please follow the instructions [here](https://docs.anaconda.com/free/anaconda/install/index.html) to install Anaconda for your machine.

We list our dependencies in `environment.yaml` file. To install the dependencies and create the env run:
```
conda env create --file=environment.yaml

# After installation, check pytorch.
conda activate ego_blur
python
>>> import torch
>>> torch.__version__
'1.12.1'
>>> torch.cuda.is_available()
True
```

Please note that this code can run on both CPU and GPU but installing both PyTorch and TorchVision with CUDA support is strongly recommended.

## Quick start

Below are copy-paste commands to get running locally. Adjust paths to where you store the models and outputs.

### Windows (NVIDIA GPU)

1) Create the environment

```powershell
conda env create --file=environment.yaml
conda activate ego_blur
```

2) Check your driver’s CUDA support and install a matching PyTorch CUDA build

```powershell
nvidia-smi   # Note the "CUDA Version: X.Y" at top-right

# Choose ONE that is <= your driver’s supported CUDA version
conda install -n ego_blur -c pytorch -c nvidia pytorch=1.12.1 torchvision=0.13.1 cudatoolkit=11.3 -y
# OR
conda install -n ego_blur -c pytorch -c nvidia pytorch=1.12.1 torchvision=0.13.1 cudatoolkit=11.6 -y
```

Alternative (pip wheels, reliable on Windows):

```powershell
python -m pip uninstall -y torch torchvision
python -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

3) Verify GPU

```powershell
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

4) Download the models from the links below and place them somewhere convenient, e.g. `C:\ego_blur_assets\`.

5) Run the demo on the sample image

```powershell
$env:CUDA_VISIBLE_DEVICES = "0"  # optional
python .\script\demo_ego_blur.py `
      --face_model_path "C:\ego_blur_assets\ego_blur_face.jit" `
      --input_image_path ".\demo_assets\test_image.jpg" `
      --output_image_path "C:\ego_blur_assets\test_image_output.jpg"
```

Notes:
- If `nvidia-smi` isn’t found, install or update your NVIDIA GPU driver.
- On some Windows setups, `torchvision` may resolve to a CPU build; the demo will still work because it runs NMS on CPU while keeping the model on GPU.

### Linux/macOS (CPU or GPU)

1) Create the environment

```bash
conda env create --file=environment.yaml
conda activate ego_blur
```

2) For GPU (Linux): install a CUDA-enabled PyTorch that matches your driver (e.g., cu113 or cu116)

```bash
conda install -n ego_blur -c pytorch -c nvidia pytorch=1.12.1 torchvision=0.13.1 cudatoolkit=11.3 -y
# or cudatoolkit=11.6
```

3) Download models and run the demo

```bash
python script/demo_ego_blur.py \
      --face_model_path /home/$USER/ego_blur_assets/ego_blur_face.jit \
      --input_image_path demo_assets/test_image.jpg \
      --output_image_path /home/$USER/ego_blur_assets/test_image_output.jpg
```

## Getting Started
First download the zipped models from given links. Then the models can be used as input/s to CLI.

| Model | Download link |
| -------- | -------- |
| ego_blur_face | [ego_blur_website](https://www.projectaria.com/tools/egoblur) |
| ego_blur_lp | [ego_blur_website](https://www.projectaria.com/tools/egoblur) |


### CLI options

A brief description of CLI args:

`--face_model_path` use this argument to provide absolute EgoBlur face model file path. You MUST provide either `--face_model_path` or `--lp_model_path` or both. If none is provided code will throw a `ValueError`.

`--face_model_score_threshold` use this argument to provide face model score threshold to filter out low confidence face detections. The values must be between 0.0 and 1.0, if not provided this defaults to 0.1.

`--lp_model_path` use this argument to provide absolute EgoBlur license plate file path. You MUST provide either `--face_model_path` or `--lp_model_path` or both. If none is provided code will throw a `ValueError`.

`--lp_model_score_threshold` use this argument to provide license plate model score threshold to filter out low confidence license plate detections. The values must be between 0.0 and 1.0, if not provided this defaults to 0.1.

`--nms_iou_threshold` use this argument to provide NMS iou threshold to filter out low confidence overlapping boxes. The values must be between 0.0 and 1.0, if not provided this defaults to 0.3.

`--scale_factor_detections` use this argument to provide scale detections by the given factor to allow blurring more area. The values can only be positive real numbers eg: 0.9(values < 1) would mean scaling DOWN the predicted blurred region by 10%, whereas as 1.1(values > 1) would mean scaling UP the predicted blurred region by 10%.

`--input_image_path` use this argument to provide absolute path for the given image on which we want to make detections and perform blurring. You MUST provide either `--input_image_path` or `--input_video_path` or both. If none is provided code will throw a `ValueError`.

`--output_image_path` use this argument to provide absolute path where we want to store the blurred image. You MUST provide `--output_image_path` with `--input_image_path` otherwise code will throw `ValueError`.

`--input_video_path` use this argument to provide absolute path for the given video on which we want to make detections and perform blurring. You MUST provide either `--input_image_path` or `--input_video_path` or both. If none is provided code will throw a `ValueError`.

`--output_video_path` use this argument to provide absolute path where we want to store the blurred video. You MUST provide `--output_video_path` with `--input_video_path` otherwise code will throw `ValueError`.

`--output_video_fps` use this argument to provide FPS for the output video. The values must be positive integers, if not provided this defaults to 30.



### CLI command example
Download the git repo locally and run following commands.
Please note that these commands assumes that you have a created a folder `/home/${USER}/ego_blur_assets/` where you have extracted the zipped models and have test image in the form of `test_image.jpg` and a test video in the form of `test_video.mp4`.

```
conda activate ego_blur
```

#### demo command for face blurring on the demo_assets image

```
python script/demo_ego_blur.py --face_model_path /home/${USER}/ego_blur_assets/ego_blur_face.jit --input_image_path demo_assets/test_image.jpg --output_image_path /home/${USER}/ego_blur_assets/test_image_output.jpg
```

Windows example:

```
python .\script\demo_ego_blur.py --face_model_path C:\\ego_blur_assets\\ego_blur_face.jit --input_image_path .\demo_assets\test_image.jpg --output_image_path C:\\ego_blur_assets\\test_image_output.jpg
```


#### demo command for face blurring on an image using default arguments

```
python script/demo_ego_blur.py --face_model_path /home/${USER}/ego_blur_assets/ego_blur_face.jit --input_image_path /home/${USER}/ego_blur_assets/test_image.jpg --output_image_path /home/${USER}/ego_blur_assets/test_image_output.jpg
```


#### demo command for face blurring on an image
```
python script/demo_ego_blur.py --face_model_path /home/${USER}/ego_blur_assets/ego_blur_face.jit --input_image_path /home/${USER}/ego_blur_assets/test_image.jpg --output_image_path /home/${USER}/ego_blur_assets/test_image_output.jpg --face_model_score_threshold 0.9 --nms_iou_threshold 0.3 --scale_factor_detections 1.15
```

#### demo command for license plate blurring on an image
```
python script/demo_ego_blur.py --lp_model_path /home/${USER}/ego_blur_assets/ego_blur_lp.jit --input_image_path /home/${USER}/ego_blur_assets/test_image.jpg --output_image_path /home/${USER}/ego_blur_assets/test_image_output.jpg --lp_model_score_threshold 0.9 --nms_iou_threshold 0.3 --scale_factor_detections 1
```

#### demo command for face blurring and license plate blurring on an input image and video
```
python script/demo_ego_blur.py --face_model_path /home/${USER}/ego_blur_assets/ego_blur_face.jit --lp_model_path /home/${USER}/ego_blur_assets/ego_blur_lp.jit --input_image_path /home/${USER}/ego_blur_assets/test_image.jpg --output_image_path /home/${USER}/ego_blur_assets/test_image_output.jpg  --input_video_path /home/${USER}/ego_blur_assets/test_video.mp4 --output_video_path /home/${USER}/ego_blur_assets/test_video_output.mp4 --face_model_score_threshold 0.9 --lp_model_score_threshold 0.9 --nms_iou_threshold 0.3 --scale_factor_detections 1 --output_video_fps 20
```

Windows example:

```
python .\script\demo_ego_blur.py --face_model_path C:\\ego_blur_assets\\ego_blur_face.jit --lp_model_path C:\\ego_blur_assets\\ego_blur_lp.jit --input_image_path C:\\ego_blur_assets\\test_image.jpg --output_image_path C:\\ego_blur_assets\\test_image_output.jpg --input_video_path C:\\ego_blur_assets\\test_video.mp4 --output_video_path C:\\ego_blur_assets\\test_video_output.mp4 --face_model_score_threshold 0.9 --lp_model_score_threshold 0.9 --nms_iou_threshold 0.3 --scale_factor_detections 1 --output_video_fps 20
```

## License

The model is licensed under the [Apache 2.0 license](LICENSE).

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Windows notes and troubleshooting

If you encounter an error like the following when importing PyTorch or running the demo on Windows:

```
OSError: [WinError 182] The operating system cannot run %1. Error loading "...\torch\lib\shm.dll" or one of its dependencies.
```

It usually indicates a mismatch in binary dependencies. A reliable workaround is to use CPU-only builds of PyTorch and TorchVision inside the conda environment. You can keep using the provided `environment.yaml` to create the environment, then run these commands in PowerShell:

```powershell
# Optional: if conda activation is blocked by execution policy, prefer conda-run
# conda activate ego_blur

conda run -n ego_blur python -m pip uninstall -y torch torchvision
conda run -n ego_blur python -m pip install torch==1.12.1+cpu torchvision==0.13.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Verify imports
conda run -n ego_blur python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
```

Tip: If your shell blocks `conda activate` due to execution policy, you can run Python inside the environment without activation via `conda run -n ego_blur ...` as shown above.

### Windows GPU setup

If you want to run the demo on your NVIDIA GPU, follow these steps on Windows PowerShell after creating the `ego_blur` environment:

1) Check your driver’s supported CUDA version

```powershell
nvidia-smi
```

In the top-right of the output, note the "CUDA Version: X.Y" shown. Choose a cudatoolkit version that is less than or equal to that value.

2) Install a CUDA-enabled PyTorch build (choose ONE that matches your driver)

```powershell
# CUDA 11.3 build
conda install -n ego_blur -c pytorch -c nvidia pytorch=1.12.1 torchvision=0.13.1 cudatoolkit=11.3 -y

# OR CUDA 11.6 build
conda install -n ego_blur -c pytorch -c nvidia pytorch=1.12.1 torchvision=0.13.1 cudatoolkit=11.6 -y
```

Alternative using pip wheels (reliable on Windows):

```powershell
conda run -n ego_blur python -m pip uninstall -y torch torchvision
conda run -n ego_blur python -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

3) Verify CUDA is detected

```powershell
conda run -n ego_blur python -c "import torch; print('torch', torch.__version__); print('cuda runtime', torch.version.cuda); print('cuda available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

4) Run the demo on GPU

```powershell
$env:CUDA_VISIBLE_DEVICES = "0"
conda run -n ego_blur python .\script\demo_ego_blur.py \
      --face_model_path "C:\path\to\ego_blur_face.jit" \
      --input_image_path ".\demo_assets\test_image.jpg" \
      --output_image_path "C:\path\to\output\test_image_output.jpg"
```

Notes:
- Some Windows installs resolve a CPU-only torchvision. The demo handles this by running NMS on CPU internally, while the model forward pass still uses the GPU when available.
- If large package downloads are flaky, try mamba and strict channel priority:

```powershell
conda install -n base -c conda-forge mamba -y
conda config --set channel_priority strict
mamba install -n ego_blur -c pytorch -c nvidia -c conda-forge pytorch=1.12.1 torchvision=0.13.1 cudatoolkit=11.3 -y
```

If `nvidia-smi` isn’t found, update/install your NVIDIA GPU driver.

## Citing EgoBlur

If you use EgoBlur in your research, please use the following BibTeX entry.

```
@misc{raina2023egoblur,
      title={EgoBlur: Responsible Innovation in Aria},
      author={Nikhil Raina and Guruprasad Somasundaram and Kang Zheng and Sagar Miglani and Steve Saarinen and Jeff Meissner and Mark Schwesinger and Luis Pesqueira and Ishita Prasad and Edward Miller and Prince Gupta and Mingfei Yan and Richard Newcombe and Carl Ren and Omkar M Parkhi},
      year={2023},
      eprint={2308.13093},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
