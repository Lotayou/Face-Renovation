[![python](https://img.shields.io/badge/python-3.6+-blue.svg)](https://github.com/Lotayou/Face-Renovation)
[![report](https://img.shields.io/badge/arxiv-report-red)](https://arxiv.org/abs/2005.05005) 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hifacegan-face-renovation-via-collaborative/image-super-resolution-on-ffhq-1024-x-1024-4x)](https://paperswithcode.com/sota/image-super-resolution-on-ffhq-1024-x-1024-4x?p=hifacegan-face-renovation-via-collaborative)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hifacegan-face-renovation-via-collaborative/image-super-resolution-on-ffhq-256-x-256-4x)](https://paperswithcode.com/sota/image-super-resolution-on-ffhq-256-x-256-4x?p=hifacegan-face-renovation-via-collaborative)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hifacegan-face-renovation-via-collaborative/face-hallucination-on-ffhq-512-x-512-16x)](https://paperswithcode.com/sota/face-hallucination-on-ffhq-512-x-512-16x?p=hifacegan-face-renovation-via-collaborative)

<img src='https://user-images.githubusercontent.com/33449901/92845327-6c816300-f419-11ea-9b03-67d29a179edf.gif' align="right" width=512>

<br><br><br><br>

# Face-Renovation

**HiFaceGAN: Face Renovation via Collaborative Suppression and Replenishment**<br>

Lingbo Yang, Chang Liu, Pan Wang, Shanshe Wang, Peiran Ren, Siwei Ma, Wen Gao<br>

### [Project](https://lotayou.github.io/projects/face_renov.html) | [arXiv](https://arxiv.org/abs/2005.05005) | [ACM link](https://dl.acm.org/doi/abs/10.1145/3394171.3413965)| [Supplementary Material](https://github.com/Lotayou/lotayou.github.io/raw/master/396_Face_Renovation_supplementary.pdf)

### Update 20201026: Pretrained checkpoints released to facilitate reproduction.
### Update 20200911: Please find video restoration results at [this repo](https://github.com/Lotayou/Face-Renovation-teaser-gifs)!
### Update: This paper is accepted at ACM Multimedia 2020.

![Stunner](https://user-images.githubusercontent.com/33449901/82039922-47cde680-96d8-11ea-8d16-8158abb3eccf.jpg)

# Contents
0. [Usage](#usage)
1. [Benchmark](#benchmark)
2. [Remarks](#remarks)
3. [License](#license)
4. [Citation](#citation)
5. [Acknowledgements](#acknowledgements)

# Usage
### Environment
- Ubuntu/CentOS
- PyTorch 1.0+
- CUDA 10.1
- python packages: opencv-python, tqdm, 
- Data augmentation tool: [imgaug](https://imgaug.readthedocs.io/en/latest/source/installation.html#installation-in-pip)
- [Face Recognition Toolkit](https://github.com/ageitgey/face_recognition) for evaluation
- [tqdm](https://github.com/tqdm/tqdm) to make you less anxious when testing:)
### Dataset Preparation
Download [FFHQ](https://github.com/NVlabs/ffhq-dataset), resize to 512x512 and split id `[65000, 70000)` for testing. We only use first 10000 images for training, which takes 2~3 days on a P100 GPU, training with full FFHQ is possible, but could take weeks.

After that, run `degrade.py` to acquire paired images for training. You need to specify the degradation type and input root in the script first. 

### Configurations
The configurations is stored in `options/config_hifacegan.py`, the options should be self-explanatory, but feel free to leave an issue anytime.

### Training and Testing
```
python train.py            # A fool-proof training script
python test.py             # Test on synthetic dataset
python test_nogt.py        # Test on real-world images
python two_source_test.py  # Visualization of Fig 5
```

### Pretrained Models

Download, unzip and put under `./checkpoints`. Then change names in configuration file accordingly.

[BaiduNetDisk](https://pan.baidu.com/s/15_vhGQdkHIfLCRgo7xanpg): Extraction code：cxp0

[YandexDisk](https://yadi.sk/d/Pl_hxVZPa_PHew)

#### Note: 
- These checkpoints works best on synthetic degradation prescribed in `degrade.py`, don't expect them to handle real-world LQ face images. You can try to fine-tune them with additional collected samples though. 
- There are two `face_renov` checkpoints trained under different degradation mixtures. Unfortunately I've forgot which one I used for our paper, so just try both and select the better one. Also, this could give you a hint about how our model behaves under a different degradation setting:)  
- You may need to set `netG=lipspade` and `ngf=48` inside the configuration file. In case of loading failure, don't hesitate to submit a issue or email me.



### Evaluation
Please find in `metrics_package` folder:
- `main.py`: GPU-based PSNR, SSIM, MS-SSIM, FID
- `face_dist.py`: CPU-based face embedding distance(FED) and landmark localization error (LLE). 
- `PerceptualSimilarity\main.py`: GPU-based LPIPS
- `niqe\niqe.py`: NIQE, CPU-based, no reference

__Note:__
- Read the scripts and modify result folder path(s) before testing (do not add `/` in the end), the results will be displayed on screen and saved in txt.
- At least 10GB is required for `main.py`. If this is too heavy for you, reduce`bs=250` at [line 79](https://github.com/Lotayou/Face-Renovation/blob/5193b083f598a1291514ea3a4c2d77e1637ac2f6/metrics_package/main.py#L79)
- Initializing Inception V3 Model for FID could take several minutes, just be patient. If you find a solution, please submit a PR.
- By default `face_dist.py` script runs with 8 parallel subprocesses, which could cause error on certain environments. In that case, just disable the multiprocessing and replace with a for loop (This would take 2~3 hours for 5k images, you may want to wrap the loop in [tqdm](https://github.com/tqdm/tqdm) to reduce your anxiety).

# Benchmark
Please refer to [benchmark.md](benchmark.md) for benchmark experimental settings and performance comparison.

__Memory Cost__ The default model is designed to fit in a P100 card with 16 GB memory. For Titan-X or 1080Ti card with 12 GB memory, you can reduce `ngf=48`, or further turn `batchSize=1` without significant performance drop.

__Inference Speed__ Currently the inference script is single-threaded which runs at 5fps. To further increase the inference speed, possible options are using multi-thread dataloader, batch inference, and combine normalization and convolution operations.

# Remarks
#### Face Renovation is not designed to create a perfect specimen OUT OF you, but to bring out the best WITHIN you.
### [The Philosophy of Face Renovation](goal.md) | [Understanding of HiFaceGAN](understanding.md)

# License
Copyright &copy; 2020, Alibaba Group. All rights reserved. 
This code is intended for academic and educational use only, any commercial usage without authorization is strictly prohibited.

# Citation
Please kindly cite our paper when using this project for your research.
```
@article{Yang2020HiFaceGANFR,
  title={HiFaceGAN: Face Renovation via Collaborative Suppression and Replenishment},
  author={Lingbo Yang and C. Liu and P. Wang and Shanshe Wang and P. Ren and Siwei Ma and W. Gao},
  journal={Proceedings of the 28th ACM International Conference on Multimedia},
  year={2020}
}
```

# Acknowledgements
The replenishment module borrows the implementation of [SPADE](https://github.com/NVlabs/SPADE).
