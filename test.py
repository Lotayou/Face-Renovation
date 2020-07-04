import os, torch
from collections import OrderedDict

import data
# change config file for ablation study...
from options.config_hifacegan import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
import numpy as np
import cv2
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.backends.cudnn.benchmark = True

opt = TestOptions()
dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
### 20200218 Critical Bug
# When model is set to eval mode, the generated image
# is not enhanced whatsoever, with almost 0 residual
# when turned to training mode, it behaves as expected.
###
#model.eval()
#model.netG.eval()
model.netG.train()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
'''
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))
'''
save_path = os.path.join(opt.results_dir, opt.name)
#save_path = os.path.join(opt.results_dir, 'debug_mixed_train')
os.makedirs(save_path, exist_ok=True)
# test
for i, data_i in tqdm(enumerate(dataloader)):
    if i * opt.batchSize >= opt.how_many:
        break

    generated = model(data_i, mode='inference2')

    img_path = data_i['path']
    for b in range(generated.shape[0]):
        #print('process image... %s' % img_path[b])
        #print('absolute error:', (data_i['label'][b] - generated[b]).abs().mean())
        
        # 20200218 debug code: residual map
        #res = (data_i['label'][b] - generated[b] + 1.) / 2.
        
        visuals = OrderedDict([('input_label', data_i['label'][b]),
                               ('synthesized_image', generated[b]),
                               #('residual_input_fake', res),
                               ('ground_truth', data_i['image'][b])
                              ])

        visuals_rgb = visualizer.convert_visuals_to_numpy(visuals)

        name = os.path.splitext(os.path.basename(img_path[b]))[0]
        im1=visuals_rgb['input_label']
        im2=visuals_rgb['synthesized_image']
        im3=visuals_rgb['ground_truth']
        #res_rgb = visuals_rgb['residual_input_fake']
        
        h = im1.shape[0]
        im = np.zeros((h, h*3, 3))
        im[:,:h] = im1
        im[:,h:2*h] = im2
        im[:,2*h:] = im3
        '''
        h = im1.shape[0]
        im = np.zeros((h, h*4, 3))
        im[:,:h] = im1
        im[:,h:2*h] = im2
        im[:,2*h:3*h] = res_rgb
        im[:,3*h:] = im3
        '''
        
        cv2.imwrite(os.path.join(save_path, name+'.jpg'), im[:,:,::-1])
        #print('a')
#webpage.save()
