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

#### Face Renovation is not designed to create a perfect specimen OUT OF you, but to bring out the best WITHIN you.

![Stunner](https://user-images.githubusercontent.com/33449901/82039922-47cde680-96d8-11ea-8d16-8158abb3eccf.jpg)


# Contents
0. [Usage](#Usage)
1. [Benchmark Performances](#Benchmark Performances)
2. [Remarks](#Remarks)
3. [License](#license)
4. [Citation](#citation)
5. [Acknowledgements](#Acknowledgements)

# Usage
### Environment
- Ubuntu/CentOS
- PyTorch 1.0+
- CUDA 10.1
- python packages: opencv-python, tqdm, 
- Data augmentation tool: [image_augmentor](https://pypi.org/project/image-augmentor/) or [albumentation](https://albumentations.readthedocs.io/en/latest/)

### Dataset Preparation
Download [FFHQ](https://github.com/NVlabs/ffhq-dataset), resize to 512-by-512 and split id 65000~69999 for testing. We only use first 10000 images for training, which takes 2~3 days on a P100 GPU, training with full FFHQ is possible but not worth it.

After that, run `degrade.py` to acquire paired images for training. You need to specify the degradation type and input root in the script first. 

### Training and Testing
```
python train.py            # A fool-proof training script
python test.py             # Test on synthetic dataset
python test_nogt.py        # Test on real-world images
python two_source_test.py  # Visualization of Fig 5
```

### Configurations
All configurations are stored in `options/config_hifacegan.py`, the options are self-explanatory. 

# Benchmark Performances
Please refer to [benchmark.md](benchmark.md)

# Remarks
### [The Philosophy of Face Renovation](goal.md) | [Understanding of HiFaceGAN](understanding.md)

# License
Copyright &copy; 2020, Alibaba Group. All rights reserved. This code is for academic and educational use only.

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
