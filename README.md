[![python](https://img.shields.io/badge/python-3.6+-blue.svg)](https://github.com/Lotayou/Face-Renovation)
[![report](https://img.shields.io/badge/arxiv-report-red)](https://arxiv.org/abs/2005.05005) 
[![PWC](https://img.shields.io/badge/SOTA-Blind%20Face%20Restoration-blue)](https://github.com/Lotayou/Face-Renovation/blob/master/benchmark.md)
[![PWC](https://img.shields.io/badge/SOTA-Image%20Super%20Resolution-blue)](https://github.com/Lotayou/Face-Renovation/blob/master/benchmark.md)
[![PWC](https://img.shields.io/badge/SOTA-Face%20Hallucination-blue)](https://paperswithcode.com/paper/hifacegan-face-renovation-via-collaborative)

<img src='https://user-images.githubusercontent.com/33449901/86509021-67030d80-be17-11ea-801d-5bb8b315ef56.png' align="right" width=360>

<br><br><br><br>

# Face-Renovation

**HiFaceGAN: Face Renovation via Collaborative Suppression and Replenishment**<br>

Lingbo Yang, Chang Liu, Pan Wang, Shanshe Wang, Peiran Ren, Siwei Ma, Wen Gao<br>

### [Project](https://github.com/Lotayou/Face-Renovation) | [arXiv](https://arxiv.org/abs/2005.05005) | [Supplementary Materials(TODO)](https://arxiv.org/abs/2005.05005)

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
- Data augmentation tool: [image_augmentor](https://pypi.org/project/image-augmentor/)
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

# Remarks
#### Face Renovation is not designed to create a perfect specimen OUT OF you, but to bring out the best WITHIN you.
### [The Philosophy of Face Renovation](goal.md) | [Understanding of HiFaceGAN](understanding.md)

# License
Copyright &copy; 2020, Alibaba Group. All rights reserved. This code is intended for academic and educational use only, any commercial usage without authorization is strictly prohibited.

# Citation
Please kindly cite our paper when using this project for your research.
```
@journal{Yang2020HiFaceGANFR,
  title={HiFaceGAN: Face Renovation via Collaborative Suppression and Replenishment},
  author={Lingbo Yang and Chang Liu and Pan Wang and Shanshe Wang and Peiran Ren and Siwei Ma and Wen Gao},
  journal={Arxiv},
  url={https://arxiv.org/abs/2005.05005},
  year={2020}
}
```

# Acknowledgements
The replenishment module borrows the implementation of [SPADE](https://github.com/NVlabs/SPADE).
