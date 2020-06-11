[![python](https://img.shields.io/badge/python-3.6+-blue.svg)
[![report](https://img.shields.io/badge/arxiv-report-red)](https://arxiv.org/abs/2005.05005) 
[![PWC](https://img.shields.io/badge/SOTA-Blind%20Face%20Restoration-blue)](https://github.com/Lotayou/Face-Renovation/blob/master/benchmark.md)
[![PWC](https://paperswithcode.com/paper/hifacegan-face-renovation-via-collaborative)

# Face-Renovation

**HiFaceGAN: Face Renovation via Collaborative Suppression and Replenishment**<br>

Lingbo Yang, Chang Liu, Pan Wang, Shanshe Wang, Peiran Ren, Siwei Ma, Wen Gao<br>
![Stunner](https://user-images.githubusercontent.com/33449901/82039922-47cde680-96d8-11ea-8d16-8158abb3eccf.jpg)

Arxiv: https://arxiv.org/abs/2005.05005
# Environment
- Ubuntu/CentOS
- PyTorch 1.0+
- CUDA 10.1
- python packages: opencv-python, tqdm, 
- Data augmentation tool: [image_augmentor](https://pypi.org/project/image-augmentor/) or [albumentation](https://albumentations.readthedocs.io/en/latest/)

# Dataset Preparation
Download [FFHQ]()

# Testing
`python test.py`

# Benchmark Performances
Please refer to [benchmark.md](benchmark.md)

# Academic Stuff
### Motivation
Image restoration has been well studied for decades, yet most existing research efforts are dedicated to specific degradation categories (as shown in the table above), leading to limited feasibility for real world cases with complex, heterogeneous degradations. 
![teaser](https://user-images.githubusercontent.com/33449901/82056254-43afc200-96f4-11ea-95f3-09ba1f6b2bde.PNG)

### Network Architecture

![framework](https://user-images.githubusercontent.com/33449901/82056692-df413280-96f4-11ea-8cc0-b2f15b456bd0.PNG)

Our network HiFaceGAN is composed of several nested __collaborative suppression and relenishment (CSR)__ units that each specializes in restoring a specific semantic aspects, leading to a systematic hierarchical face renovation, as displayed below. 
In consequence, our dual-blind solution can even outperform SOTA blind face restoration with additional HQ reference [(GFRNet)](https://github.com/csxmli2016/GFRNet):

![dual_vs_single_blind](https://user-images.githubusercontent.com/33449901/82056900-2d563600-96f5-11ea-891f-68b34430298c.PNG)
 
# License
Copyright &copy; 2020, Alibaba Group. All rights reserved.

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
