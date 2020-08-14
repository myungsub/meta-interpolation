# SAVFI - Meta-Learning for Video Frame Interpolation

#### Myungsub Choi, Janghoon Choi, Sungyong Baik, Tae Hyun Kim, Kyoung Mu Lee

Source code for CVPR 2020 paper "Scene-Adaptive Video Frame Interpolation via Meta-Learning"

[Project](https://myungsub.github.io/meta-interpolation) | [Paper-CVF](http://openaccess.thecvf.com/content_CVPR_2020/papers/Choi_Scene-Adaptive_Video_Frame_Interpolation_via_Meta-Learning_CVPR_2020_paper.pdf) | [Paper-ArXiv](https://arxiv.org/abs/2004.00779) | [Supp](http://openaccess.thecvf.com/content_CVPR_2020/supplemental/Choi_Scene-Adaptive_Video_Frame_CVPR_2020_supplemental.zip)

<a href="https://arxiv.org/abs/2004.00779" rel="Video"><img src="./figures/SAVFI_paper_thumb.jpg" alt="Paper" width="100%"></a>


## Requirements

- Ubuntu 18.04
- Python==3.7
- numpy==1.18.1
- PyTorch==1.4.0, cudatoolkit==10.1
- opencv==3.4.2
- cupy==7.3 (recommended: `conda install cupy -c conda-forge`)
- tqdm==4.44.1

For [[DAIN](https://github.com/baowenbo/DAIN)], the environment is different; please check `dain/dain_env.yml` for the requirements.


## Usage

***Disclaimer :*** This code is re-organized to run multiple different models in this single codebase. Due to a lot of version and env changes, the numbers obtained from this code may be different (usually better) from those reported in the paper. The original code modifies the main training scripts for each frame interpolation github repo ([[DVF (voxelflow)](https://github.com/lxx1991/pytorch-voxel-flow)], [[SuperSloMo](https://github.com/avinashpaliwal/Super-SloMo)], [[SepConv](https://github.com/sniklaus/sepconv-slomo)], [[DAIN](https://github.com/baowenbo/DAIN)]), and are put in `./legacy/*.py`. If you want to *exactly* reproduce the numbers reported in our paper, please contact [@myungsub](https://github.com/myungsub) for legacy experimental settings.

### Dataset Preparation

- We use [ [Vimeo90K Septuplet dataset](http://toflow.csail.mit.edu/) ] for training + testing
  - After downloading the full dataset, make symbolic links in `data/` folder:
    - `ln -s /path/to/vimeo_septuplet_data/ ./data/vimeo_septuplet`
- For further evaluation, use:
  - [ [Middlebury-OTHERS dataset](http://vision.middlebury.edu/flow/data/) ] - download `other-color-allframes.zip` and `other-gt-interp.zip`
  - [ [HD dataset](https://github.com/baowenbo/MEMC-Net#hd-dataset-results) ] - download the original ground truth videos [[here](https://merced-my.sharepoint.com/:u:/g/personal/wbao2_ucmerced_edu/EU-1cwJsIGJLmGsIz6a30sEBo-Jv2DWcw65qElR5xwh6VA?e=spBaNI)]

### Frame Interpolation Model Preparation

- Download pretrained models from [[Here](https://www.dropbox.com/sh/4pphxdw8k3j34dq/AABRr61SSw09zVgfjYXlaHe3a?dl=0)], and save them to `./pretrained_models/*.pth`

### Training / Testing with Vimeo90K-Septuplet dataset

- For training, simply run: `./scripts/run_{VFI_MODEL_NAME}.sh`
  - Currently supports: `sepconv`, `voxelflow`, `superslomo`, `cain`, and `rrin`
  - Other models are coming soon!
- For testing, just uncomment two lines containing: `--mode val` and `--pretrained_model {MODEL_NAME}`

### Testing with custom data

- See `scripts/run_test.sh` for details:
- Things to change:
  - Modify the folder directory containing the video frames by changing `--data_root` to your desired dir/
  - Make sure to match the image format `--img_fmt` (defaults to `png`)
  - Change `--model`, `--loss`, and `--pretrained_models` to what you want 
    - For [SepConv](https://github.com/sniklaus/sepconv-slomo), `--model` should be `sepconv`, and `--loss` should be `1*L1`
    - For [VoxelFlow](https://github.com/lxx1991), `--model` should be `voxelflow`, and `--loss` should be `1*MSE`
    - For [SuperSloMo](https://github.com/avinashpaliwal/Super-SloMo),  `--model` should be `superslomo`, `--loss` should be `1*Super`
    - For [DAIN](https://github.com/baowenbo/DAIN), `--model` should be `dain`, and `--loss` should be `1*L1`
    - For [CAIN](https://github.com/myungsub/CAIN), `--model` should be `cain`, and `--loss` should be `1*L1`
    - For [RRIN](https://github.com/HopLee6/RRIN), '`--model` should be `rrin`, and `--loss` should be `1*L1`


### Using Other Meta-Learning Algorithms

- Current code supports using more advanced meta-learning algorithms compared to vanilla MAML, *e.g.* [MAML++](https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch), [L2F](https://github.com/baiksung/L2F), or [Meta-SGD](https://arxiv.org/abs/1707.09835).
  - For MAML++ you can explore many different hyperparameters by adding additional options (see `config.py`)
  - For L2F, just uncomment `--attenuate` in `scripts/run_{VFI_MODEL_NAME}.sh`
  - For Meta-SGD, just uncomment `--metasgd` (This usually results in the best performance!)

### Framework Overview

<center><img src="./figures/fig_1.png" width="80%"></center>



## Results

- Qualitative results for VimeoSeptuplet dataset

<center><img src="./figures/fig_qual.png" width="100%"></center>

- Qualitative results for Middlebury-OTHERS dataset

<center><img src="./figures/fig_qual_supp_middlebury.png" width="100%"></center>

- Qualitative results for HD dataset

<center><img src="./figures/fig_qual_supp_hd.png" width="100%"></center>


### Additional Results Video

<center><a href="https://www.dropbox.com/s/6h4hyeiuoulzyk7/07235-supp-video.mp4" rel="Video"><img src="./figures/thumb.png" alt="Video" width="70%"></a></center>


## Citation

If you find this code useful for your research, please consider citing the following paper:

``` text
@inproceedings{choi2020meta,
    author = {Choi, Myungsub and Choi, Janghoon and Baik, Sungyong and Kim, Tae Hyun and Lee, Kyoung Mu},
    title = {Scene-Adaptive Video Frame Interpolation via Meta-Learning},
    booktitle = {CVPR},
    year = {2020}
}
```

## Acknowledgement

The main structure of this code is based on [MAML++](https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch).
Training scripts for each of the frame interpolation method is adopted from: [[DVF](https://github.com/lxx1991/pytorch-voxel-flow)], [[SuperSloMo](https://github.com/avinashpaliwal/Super-SloMo)], [[SepConv](https://github.com/sniklaus/sepconv-slomo)], [[DAIN](https://github.com/baowenbo/DAIN)], [[CAIN](https://github.com/myungsub/CAIN)], [[RRIN](https://github.com/HopLee6/RRIN)]. 
We thank the authors for sharing the codes for their great works.
