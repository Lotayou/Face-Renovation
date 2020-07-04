# Understanding of HiFaceGAN

### Why dual-blind?
Image restoration has been well studied for decades, yet most existing research efforts are dedicated to specific degradation categories (as shown in the table above), 
leading to limited feasibility for real world cases with complex, heterogeneous degradations. 
![teaser](https://user-images.githubusercontent.com/33449901/82056254-43afc200-96f4-11ea-95f3-09ba1f6b2bde.PNG)

Some existing works investigate the blind face restoration problem. However, these works often rely on other semantic guidances, such as landmark keypoints, parsing maps, or even
other high quality faces [(GFRNet)](https://github.com/csxmli2016/GFRNet), making the network focus on the frontal face and neglect the background:
![dualvssingle](https://user-images.githubusercontent.com/33449901/86511400-58beec80-be2b-11ea-8198-3fc246ae2a60.png)

### Why collaborative suppression and replenishment?

![framework](https://user-images.githubusercontent.com/33449901/82056692-df413280-96f4-11ea-8cc0-b2f15b456bd0.PNG)

Our network HiFaceGAN is composed of several nested __collaborative suppression and relenishment (CSR)__ units 
that each specializes in restoring a specific semantic aspects, leading to a systematic hierarchical face renovation, as displayed here:

![5stage](https://user-images.githubusercontent.com/33449901/86511401-5ceb0a00-be2b-11ea-9fcc-c600c220b19c.png)

Compare to original SPADE, stylegan or VAE, our framework allows the model learn the optimial feature representation and decomposition through
an end-to-end learning process, hence can achieve superior renovation performance that most baselines.

### Will this work be applicable to generalized images?
Of course. Although we name our model "HiFaceGAN", neither the problem formulation nor the implementation relies on facial prior, 
thus it can adapt to other natural images, no modification required, and achieve satisfactory performances. 
Still, the increasing complexity in textures is a huge challenge for our framework(and presumably for any other generative models).
Here we show the result of our HiFaceGAN for tackling joint rain and haze removal on outdoor scene images (the model has been fine-tuned):


#### From left to right: Input,                                  Output,                     Ground Truth
![image](https://user-images.githubusercontent.com/33449901/86511658-c66c1800-be2d-11ea-96d4-8f1183e21829.png)
![image](https://user-images.githubusercontent.com/33449901/86511678-f9161080-be2d-11ea-96fb-ee2f47f2c095.png)
![image](https://user-images.githubusercontent.com/33449901/86511715-43978d00-be2e-11ea-8da8-c81d8f8296c0.png)


