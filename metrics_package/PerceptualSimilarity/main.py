import os
import models
import numpy as np
from util import util
from tqdm import tqdm
import cv2
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.backends.cudnn.enabled=True
torch.backends.cudnn.benchmark=True

## Initializing the model
model = models.PerceptualLoss(model='net-lin',net='alex',use_gpu=True)

def parse_ims(dir):
    pack = cv2.imread(dir)
    # [0,255] -> [-1., 1.]
    pack = pack[np.newaxis].transpose(0,3,1,2) / 127.5 - 1
    pack = torch.from_numpy(pack.astype(np.float32))
    
    fake = pack[:,:,:,512:1024].cuda()
    real = pack[:,:,:,1024:1536].cuda()
    #fake = pack[:,:,:512].cuda()
    #real = pack[:,:,512:].cuda()
    return fake, real

# crawl directories
def main_process(folder):
    files = os.listdir(folder)
    dist_sum = 0
    for file in tqdm(files):
        fake, real = parse_ims(os.path.join(folder, file))
        dist01 = model.forward(fake,real).item()
        dist_sum += (dist01)

    dist_mean = dist_sum / len(files)
    print('Mean: %.4f' % dist_mean)
    return dist_mean

if __name__ == '__main__':
    debug = [
        '/home/lingbo.ylb/projects/face_sr_sota/Super-FAN/results_3mix',
        '/home/lingbo.ylb/projects/face_sr_sota/mmsr/results/RRDB_ESRGAN_x4/FFHQ_3mix',
        '/home/lingbo.ylb/projects/face_sr_sota/WaveletSRNet-master/results_3mix',
        '/home/lingbo.ylb/projects/face_sr_sota/DeblurGANv2/results/FFHQ_3mix',
        '/home/lingbo.ylb/projects/face_sr_sota/PyTorch-ARCNN/FFHQ_3mix',
        '/home/lingbo.ylb/projects/pg_spade_face_v4/results/face_enhan_lingbo.ylb_3mix_LIP_encoder',
    ]
    folders = [
        # SR
        '/home/lingbo.ylb/projects/face_sr_sota/mmsr/results/MSRResNetx4/FFHQ_scratch',
        '/home/lingbo.ylb/projects/face_sr_sota/mmsr/results/MSRGANx4/FFHQ_scratch',
        '/home/lingbo.ylb/projects/face_sr_sota/mmsr/results/RRDB_ESRGAN_x4/FFHQ_scratch',
        '/home/lingbo.ylb/projects/face_sr_sota/SRFBN_CVPR19/results/SR/MyImage/SRFBN/x4',
        '/home/lingbo.ylb/projects/face_sr_sota/Super-FAN/results',
        '/home/lingbo.ylb/projects/face_sr_sota/WaveletSRNet-master/results',
        '/home/lingbo.ylb/projects/pg_spade_face_v4/results/face_enhan_dixian.wp_no_pg',
        
        # 16x
        '/home/lingbo.ylb/projects/face_sr_sota/Super-FAN/results_16x',
        '/home/lingbo.ylb/projects/face_sr_sota/ESRGAN/results/ESRGAN-V1-FFHQ-16x',
        '/home/lingbo.ylb/projects/face_sr_sota/WaveletSRNet-master/results_16x_epoch30','/home/lingbo.ylb/projects/pg_spade_face_v4/results/face_enhan_lingbo.ylb_16x_LIP_encoder',
        
        # denoising
        '/home/lingbo.ylb/projects/face_sr_sota/RIDNet-master/TestCode/experiment/RIDNET_RNI15/results',
        '/home/lingbo.ylb/projects/face_sr_sota/WaveletSRNet-master/results_noise',
        '/home/lingbo.ylb/projects/face_sr_sota/VDNet-master/results/noise','/home/lingbo.ylb/projects/pg_spade_face_v4/results/face_enhan_lingbo.ylb_noise_LIP_encoder',
        
        # debluring
        '/home/lingbo.ylb/projects/face_sr_sota/DeblurGAN/results',
        '/home/lingbo.ylb/projects/face_sr_sota/DeblurGANv2/results/FFHQ_motion_blur',
        '/home/lingbo.ylb/projects/pg_spade_face_v4/results/face_enhan_lingbo.ylb_motion_blur_LIP_encoder',
        
        # jpeg
        '/home/lingbo.ylb/projects/face_sr_sota/PyTorch-ARCNN/results',
        '/home/lingbo.ylb/projects/face_sr_sota/EPGAN/results/FFHQ_jpeg',
        '/home/lingbo.ylb/projects/pg_spade_face_v4/results/face_enhan_lingbo.ylb_jpeg_LIP_encoder',
        
        # Face Renovation
        '/home/lingbo.ylb/projects/face_sr_sota/ESRGAN/results/ESRGAN-V1-FFHQ-mixed',
        '/home/lingbo.ylb/projects/face_sr_sota/Super-FAN/results_mixed',
        '/home/lingbo.ylb/projects/face_sr_sota/WaveletSRNet-master/results_mixed',
        '/home/lingbo.ylb/projects/pg_spade_face_v4/results/face_enhan_lingbo.ylb_mixed_LIP_encoder',
    ]
    
    ablation = [    '/home/lingbo.ylb/projects/pg_spade_face_v4/results/face_enhan_lingbo.ylb_16x_spade','/home/lingbo.ylb/projects/pg_spade_face_v4/results/face_enhan_lingbo.ylb_16x_contrasive_encoder','/home/lingbo.ylb/projects/pg_spade_face_v4/results/face_enhan_lingbo.ylb_16x_LIP_encoder',
        '/home/lingbo.ylb/projects/pg_spade_face_v4/results/face_enhan_lingbo.ylb_16x_subadd_encoder',
        '/home/lingbo.ylb/projects/pg_spade_face_v4/results/face_enhan_lingbo.ylb_16x_LIP_encoder_L1_Loss',
    ]
    
    with open('Face_Renovation_debug_LPIPS.txt', 'w') as f:
        #for folder in debug:
        #for folder in ablation:
        #for folder in folders:
        for folder in [
            '/home/lingbo.ylb/datasets/FFHQ_yuv_recon/37',
            '/home/lingbo.ylb/datasets/FFHQ_yuv_recon/42',
            '/home/lingbo.ylb/datasets/FFHQ_yuv_recon/45',
            '/home/lingbo.ylb/datasets/FFHQ_yuv_recon/47',
            '/home/lingbo.ylb/datasets/FFHQ_yuv_recon/50',
        ]:
            print('Processing %s' % folder)
            dist = main_process(folder)
            f.write('%s: %.4f\n' % (folder, dist))
