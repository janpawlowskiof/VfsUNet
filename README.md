# VfsUNet (Variable Filter Size U-Net)
Repository used for bachelor thesis by Jan Pawłowski.

VfsUNet (inspired by VRCNN and UNet) is an architecture for artifact removal on intra predicted frames
in HEVC. It is meant as a replacement for DBF and SAO filters.
Unlike VRCNN or VRCNN-BN this architecture is able to perform 
inference on 1280x720 input 30 times per second on a GTX 1060.

This version is tested in domain specific scenarios, meaning that
model is trained and tested on clips from one videogame
in order to simplify model's task and allow it to be more lightweight.

### Setup

In order to install required packages run
```shell
pip -r install requirements.txt
```
In order to compress clips `ffmpeg` needs to be installed.

### Testing inference speed
To test performance on local machine run
```shell
python -m run.performance_benchmark
```

where example output looks like
```shell
cuda version: 10.2
cudnn version: 7605
VFS-UNet: 97.8225175685277 fps
VRCNN: 36.11236779184184 fps
VRCNN-BN: 10.096310399883293 fps
```

### Reconstructing clip with a trained model
```shell
python -m run.reconstruct --model=<path to model> --qp=<qp value like 39> --clip=<path to clip.mp4> --output=<output folder>
```
for example:
```shell
python -m run.reconstruct --model=trained_models/tf2-qp39 --qp=39 --clip=example_data/raw/example_clip.mp4 --output=reconstuction_output
```

### Compressing clips in directory
You must provide path to folder, which contains folder named `raw` containing high quality `.mp4` clips
```shell
python -m run.compress_clips --paths <path to folder nr 1> <path to folder nr 2> ...
```
for example:
```shell
python -m run.compress_clips --paths example_data
```


### Splitting clips into images
You must provide path to root directory and folders containing clips that you want to split
```shell
python -m run.split_clips --root <path to root folder> --folder <folder nr 1> <foldre nr 2> ...
```
for example:
```shell
python -m run.split_clips --root example_data --folder raw
```


### Performing training
Training settings are configured in file `configs/train_config.yaml`
It is required to be logged in to `wandb`.
```shell
python -m run.train
```

### Calculating metrics and exporting plots
Metrics are calculated and plotted in folder `notebooks`
in form of jupyter notebooks
