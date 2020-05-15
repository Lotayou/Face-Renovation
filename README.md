# Face-Renovation
Official repository of the paper ["HiFaceGAN: Face Renovation via Collaborative Suppression and Replenishment"](https://arxiv.org/abs/2005.05005), addressing the problem of "dual-blind" face renovation with a collaborative suppression and replenishment strategy.

[//]: ![Stunner](https://user-images.githubusercontent.com/33449901/82039922-47cde680-96d8-11ea-8d16-8158abb3eccf.jpg)

# Motivation
Image restoration has been well studied for decades, yet most existing research efforts are dedicated to specific degradation categories (as shown in the table below), leading to limited feasibility for real world cases with complex, heterogeneous degradations. (Einstein fig)
|   Degradation Type  | Corresponding Task           |  Example Method |
|:-------------------:|:----------------------------:|:------------:|
| Downsampling        | Super Resolution             |  ESRGAN |
| Mosaic              | Face Hallucination           |  Super-FAN |
| Additive Noise      | Denoising                    |  VDNet |
| Motion/Defocus Blur | Deblurring                   |  DeblurGAN2 |
| JPEG artifacts      | Compression Artifact Removal |  ARCNN |
| __Mixed/Unknown__   | __Face Renovation__          | __HiFaceGAN__|

Furthermore, for category-specific (face) restoration, it is often believed that utilizing external structural information, even additional HQ reference images [1](https://github.com/csxmli2016/GFRNet) is important for 

# Network Architecture
Our network HiFaceGAN is composed of several nested __collaborative suppression and relenishment (CSR)__ units that each specializes in restoring a specific semantic aspects, leading to a systematic hierarchical face renovation, as displayed below
 
# Environment
- Ubuntu/CentOS
- PyTorch 1.0+
- CUDA 10.1
- python packages: opencv-python, tqdm, 
- Data augmentation tool: [image_augmentor](https://pypi.org/project/image-augmentor/) or [albumentation](https://albumentations.readthedocs.io/en/latest/)

# Acknowledgements
This implementation borrows from [SPADE](https://github.com/NVlabs/SPADE).
