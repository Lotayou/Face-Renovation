
# Benchmark Performance
We conduct experiments on five subtasks over the [FFHQ](https://github.com/NVlabs/stylegan) dataset with state-of-the-art baselines, and choose five most competitive baselines to tackle the fully-degraded Face Renovation task alongside HiFaceGAN. 

|   Degradation Type  | Corresponding Task           |  Example Method |
|:-------------------:|:----------------------------:|:------------:|
| Downsampling        | Super Resolution             |  ESRGAN |
| Mosaic              | Face Hallucination           |  Super-FAN |
| Additive Noise      | Denoising                    |  WaveletCNN
| Motion/Gaussian Blur| Deblurring                   |  DeblurGANv2 |
| JPEG artifacts      | Compression Artifact Removal |  ARCNN |
| __Mixed/Unknown__   | __Face Renovation__          | __HiFaceGAN__|

Example of degraded images are showcased here:
![example](https://user-images.githubusercontent.com/33449901/82058141-ebc68a80-96f6-11ea-853f-b7a0ce7eba79.png)

And the quantitative performance are reported below:
![SOTA](https://user-images.githubusercontent.com/33449901/82058403-4a8c0400-96f7-11ea-90a8-701b88bedf0e.png)
