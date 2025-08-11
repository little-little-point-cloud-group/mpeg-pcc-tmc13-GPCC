# JEE 6.2 software for G-PCC-based Anchor Generation 

## Building

### OSX
- mkdir build
- cd build
- cmake .. -G Xcode 
- open the generated xcode project and build it

### Linux
- mkdir build
- cd build
- cmake .. 
- make

### Windows
- md build
- cd build
- cmake .. -G "Visual Studio 15 2017 Win64"
- open the generated visual studio solution and build it

## Pre-processing
Pre-processing for ("INRIA" format) GS data should be applied by using `sample_files/JEE6.2/example_pre_process.py`.
Only geometry precision is changed from 22-bit to 18-bit compared with the pre-processing software uploaded to [NextCloud](https://mpeg.expert/live/nextcloud/index.php/s/B8xEyKcj945PsTo) to align with the setting defined in WG4/N0585 (JEE 6.2 description).


## Running

### Configuration files

Configuration files to generate anchor results are stored in `cfg_3dgs_sample`. Parameters including QP values defined in WG4/N0585 (JEE 6.2 description) are set in the config files.

### Example

The following example encodes and decodes a pre-processed GS file named `point_cloud.ply` by setting `r01`.

- `-C`: folder path for output result files. The folder should be created before the encoding.
- `VPATH`: folder path for config files.
- `ENCODER` and `DECODER`: path for the executable file of the software.
- `SRCSEQ`: path for pre-processed input GS file.

```console
mpeg-pcc-tmc13$  make -f $PWD/scripts/Makefile.tmc13-step \
    -C results/r01/ \
    VPATH=$PWD/cfg_3dgs_sample/octree-raht/lossless-geom-lossy-attrs/r01/ \
    ENCODER=$PWD/build/tmc3/tmc3 \
    DECODER=$PWD/build/tmc3/tmc3 \
    SRCSEQ=$PWD/../pre_post_processing/point_cloud.ply
```

When encoding/decoding is done by the above command, the decoded GS file is output as `results/r01/point_cloud.ply.bin.decoded.ply`.

## Post-processing
Post-processing be applied by using `sample_files/JEE6.2/example_post_process.py` with json file generated at pre-processing.

## Evaluation

The specific procedure to evaluate results will be provided by JEE 6.1.

## Examples of test results
Examples of test results are stored in `results` folder. The test condition is the same as described in [m70712](https://dms.mpeg.expert/doc_end_user/current_document.php?id=96781&id_meeting=201).

- Original GS files: [INRIA's pre-trained 3DGS data](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip).
- COLMAP data for evaluation: [Ground truth data of Mip-NeRF 360](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip)

### Evaluations with rendered images by INRIA software
Post-processed GS files are rendered and PSNRs are calculated by [INRIA's software](https://github.com/graphdeco-inria/gaussian-splatting). The results are in `results/INRIA` folder. This is completely the same results as [m70712](https://dms.mpeg.expert/doc_end_user/current_document.php?id=96781&id_meeting=201).

### Evaluations with rendered images by gsplat
Post-processed GS files are rendered and PSNRs are calculated by [gsplat v1.4.0](https://github.com/nerfstudio-project/gsplat) and the script `sample_files/JEE6.1/eval_JEE6_1.py` with `normalize_world_space: False` as follows. 
```console
python3 eval_JEE6_1.py --colmap_data_dir /path/to/COLMAP/folder/ --result_dir ./results --input_ply ./point_cloud.ply --normalize_world_space ""
```
The results are in `results/gsplat` folder. Bitrates are the same as the results in `results/INRIA` folder because the differences are methods for rendering and calculating PSNRs. 
