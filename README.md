# Face-Renovation
Official repository of the paper
["HiFaceGAN: Face Renovation via Collaborative Suppression and Replenishment"](https://arxiv.org/abs/2005.05005).

![Stunner](https://user-images.githubusercontent.com/33449901/82039922-47cde680-96d8-11ea-8d16-8158abb3eccf.jpg)

# Environment
- Ubuntu/CentOS
- PyTorch 1.0+
- CUDA 10.1
- python packages: opencv-python, tqdm, 
- Data augmentation tool: [image_augmentor](https://pypi.org/project/image-augmentor/) or [albumentation](https://albumentations.readthedocs.io/en/latest/)

# Benchmark Performance
We conduct experiments on five subtasks in face renovation with state-of-the-art baselines, and choose five most competitive baselines to tackle the fully-degraded Face Renovation task alongside HiFaceGAN. 

|   Degradation Type  | Corresponding Task           |  Example Method |
|:-------------------:|:----------------------------:|:------------:|
| Downsampling        | Super Resolution             |  ESRGAN |
| Mosaic              | Face Hallucination           |  Super-FAN |
| Additive Noise      | Denoising                    |  WaveletCNN
| Motion/Defocus Blur | Deblurring                   |  DeblurGANv2 |
| JPEG artifacts      | Compression Artifact Removal |  ARCNN |
| __Mixed/Unknown__   | __Face Renovation__          | __HiFaceGAN__|

Example of degraded images are showcased here:
![example](https://user-images.githubusercontent.com/33449901/82058141-ebc68a80-96f6-11ea-853f-b7a0ce7eba79.png)

And the quantitative performance are reported below:
![SOTA](https://user-images.githubusercontent.com/33449901/82058403-4a8c0400-96f7-11ea-90a8-701b88bedf0e.png)

# Academic Stuff
### Motivation
Image restoration has been well studied for decades, yet most existing research efforts are dedicated to specific degradation categories (as shown in the table above), leading to limited feasibility for real world cases with complex, heterogeneous degradations. 
![teaser](https://user-images.githubusercontent.com/33449901/82056254-43afc200-96f4-11ea-95f3-09ba1f6b2bde.PNG)

### Network Architecture

![framework](https://user-images.githubusercontent.com/33449901/82056692-df413280-96f4-11ea-8cc0-b2f15b456bd0.PNG)

Our network HiFaceGAN is composed of several nested __collaborative suppression and relenishment (CSR)__ units that each specializes in restoring a specific semantic aspects, leading to a systematic hierarchical face renovation, as displayed below. 
In consequence, our dual-blind solution can even outperform STOA blind face restoration with additional HQ reference [(GFRNet)](https://github.com/csxmli2016/GFRNet):

![dual_vs_single_blind](https://user-images.githubusercontent.com/33449901/82056900-2d563600-96f5-11ea-891f-68b34430298c.PNG)
 

# Acknowledgements
This implementation borrows from [SPADE](https://github.com/NVlabs/SPADE).
