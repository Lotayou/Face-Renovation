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

def tensor2npy(T):
    middle = T.detach().cpu().numpy().transpose(1,2,0) * 255
    print('T max: ', middle.max())
    return np.array(middle.astype(np.uint8)[:,:,::-1])

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

def all_examples():
    save_path = os.path.join(opt.results_dir, opt.name)
    #save_path = os.path.join(opt.results_dir, opt.name+'_one_plug')
    #save_path = os.path.join(opt.results_dir, 'debug_mixed_train')
    os.makedirs(save_path, exist_ok=True)
    # test
    data_cache = None
    for i, data_i in tqdm(enumerate(dataloader)):
        if i * opt.batchSize >= opt.how_many:
            break

        # Use two different input
        # For this purpose, we need to collect two batches
        # so we jump over every one batch
        if i % 2 == 0:
            data_cache = data_i
            continue
        else:
            # use the cached image for semantic/style guidance
            # use the current image for spatial guidance
            with torch.no_grad():
            #if True:
                semantic_input = data_cache['label'].cuda()
                #spatial_input = data_i['label'].cuda()
                #semantic_input = torch.ones_like(spatial_input) * 0.5
                spatial_input = torch.ones_like(semantic_input) * 0.5
            
                # semantic_feature = model.netG.encoder(semantic_input)
                mixed_output = [None] * 9
                for i in range(9):
                    mixed_output[i] = model.netG.mixed_guidance_forward(
                        input=semantic_input,
                        seg=spatial_input,
                        #input=spatial_input,
                        #seg=semantic_input,
                        n=i,
                        #mode='progressive',
                        #mode='one_plug',
                        mode='one_ablate',
                    )[0].detach()
                
            img_path = data_i['path']
            name = os.path.splitext(os.path.basename(img_path[0]))[0]
            pack = torch.cat(mixed_output, dim=2)
            
            '''
            im = tensor2npy(pack)
            cv2.imwrite(os.path.join(save_path, name+'.jpg'), im)
            '''
            visuals = OrderedDict([('in', pack),])
            visuals_rgb = visualizer.convert_visuals_to_numpy(visuals)
            im = visuals_rgb['in']
            cv2.imwrite(os.path.join(save_path, name+'.jpg'), im[:,:,::-1])
        
        
def one_example(a_id, b_id):
    instance = data.create_dataset(opt)
    data_cache = instance[a_id]
    data_i = instance[b_id]
    
    with torch.no_grad():
        #if True:
            semantic_input = data_cache['label'].unsqueeze(0).cuda()
            spatial_input = data_i['label'].unsqueeze(0).cuda()
            #semantic_input = torch.zeros_like(spatial_input)
            #spatial_input = torch.zeros_like(semantic_input)
            #spatial_input = torch.ones_like(semantic_input) * 0.5
        
            # semantic_feature = model.netG.encoder(semantic_input)
            mixed_output = [None] * 9
            for i in range(9):
                mixed_output[i] = model.netG.mixed_guidance_forward(
                    input=semantic_input,
                    seg=spatial_input,
                    #input=spatial_input,
                    #seg=semantic_input,
                    n=i,
                    mode='progressive',
                    #mode='one_plug',
                )[0].detach()
            
            path = 'results/one_example.jpg'
            pack = torch.cat(mixed_output, dim=2)
            visuals = OrderedDict([('in', pack),])
            visuals_rgb = visualizer.convert_visuals_to_numpy(visuals)
            im = visuals_rgb['in']
            cv2.imwrite(path, im[:,:,::-1])
            
if __name__ == '__main__':
    #one_example(14,5)
    all_examples()