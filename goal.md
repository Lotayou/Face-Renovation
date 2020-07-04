# The Philosophical aspects of Face Renovation

### [Etymology](https://www.merriam-webster.com/dictionary/renovate)
The term "renovation" is originated from the Latin verb _novare_, which means "to make new". Different from "restoration", renovation is mainly used for things of
significant aesthetic and historic value, _e.g._ renovating an art piece, an old church, or an inhabitant of wild lifes. 

### Restoration, Recreation, and Renovation
The major distinction of face renovation against other face-processing tasks is the ability to consistently produce high-quality renovation results regardless of input conditions.
In contrast, the quality of face restoration algorithms generally deteriorate rapidly along the input degradation level, and are not suitable for real-world applications.
Recently a new work [PULSE](https://github.com/adamian98/pulse) propose a new generative-based restoration framework by optimizing the latent vector over a pretrained StyleGAN manifold.
However, the iteratively optimized result is often different from the original input in terms of gender, race and general disposition. In this regard, we think PULSE is 
not qualified as a restoration (or super-resolution) algorithm, what it does is just "face recreation".

Here is an illustration to the distinction between restoration(ESRGAN, 4x), recreation(PULSE) and renovation(ours):
![20200704205403](https://user-images.githubusercontent.com/33449901/86512934-8fe7ca80-be38-11ea-8f5e-a4207b97f80e.jpg)


### The "racial bias" of PULSE
Recently, there has been some [accusations](https://www.theverge.com/21298762/face-depixelizer-ai-machine-learning-tool-pulse-stylegan-obama-bias) towards [PULSE](https://github.com/adamian98/pulse) 
for turning a pixelated Obama figure into a white guy, which quickly escalates from an academic issue to an act of racism.  

We test the fairness of our HiFaceGAN over 13 famous colored people, and observe 3 gender changes and 9 color-to-white cases for PULSE. 
In contrast, HiFaceGAN always preserves the input person's essential characteristics regardless of the gender, race and head pose of the input. 

 **NOTE: Our HiFaceGAN is also trained on the gender-biased FFHQ dataset.**
 
From left to right: Original image, 16x downsampled input, PULSE, our HiFaceGAN
![black](https://user-images.githubusercontent.com/33449901/85966447-68d16900-b9f2-11ea-96c9-98501803da7e.jpg)

