###################
# face_dist.py:
# - Calculating face embedding distance
# - Calculating landmark distance
###################

import face_recognition as fr
import numpy as np
import os
import cv2
import torch
from tqdm import tqdm
from time import time

from multiprocessing import Pool

def calc_distances(item):
    global emb_sum, lm_sum
    def convert_dict(lm_dict):
        lm_list = []
        for k, v in lm_dict.items():
            lm_list += v
        return np.array(lm_list)
    
    s = 512
    pack = fr.load_image_file(item)
    fake = pack[:,s:2*s]
    real = pack[:,2*s:3*s]
    #fake = pack[:s]
    #real = pack[s:]
    #fake_enc = fr.face_encodings(fake)[0]
    try:
        fake_enc = fr.face_encodings(fake)[0]
        real_enc = fr.face_encodings(real)[0]
        enc_dist = np.linalg.norm(fake_enc - real_enc)
        
        fake_lm = convert_dict(fr.face_landmarks(fake)[0])
        real_lm = convert_dict(fr.face_landmarks(real)[0])
        # result: a dictionary containing landmarks and such
        lm_dist = np.linalg.norm(fake_lm - real_lm, axis=1).mean()
        
    except:
        return None, None
    
    return enc_dist, lm_dist
    
def main_process(FOLDER):
    print('Processing folder:\n', FOLDER)
    file_packs = [os.path.join(FOLDER, l) for l in os.listdir(FOLDER)]

    tic = time()
    
    with Pool(8) as p:
        pack = p.map(calc_distances, file_packs)
        pack = [t for t in pack if t[0] is not None]
        print('valid num: %d' % len(pack))
        pack = np.array(pack).mean(axis=0)
        emb_sum, lm_sum = pack[0], pack[1]
    '''    
    for file in file_packs:
        emb_dist, lm_dist = calc_distances(file)
        emb_sum += emb_dist
        lm_sum += lm_dist
    '''
    
    print('Time: {} seconds'.format(time() - tic))
    print('Mean embedding distance: %.6f' % emb_sum)
    print('Mean landmark distance: %.6f' % lm_sum)
    with open(FOLDER + '_face_metrics.txt', 'w') as f:
        f.write('Mean embedding distance: %.6f\n' % emb_sum)
        f.write('Mean landmark distance: %.6f\n' % lm_sum)
    
    
if __name__ == '__main__':
    #FOLDER = '/home/lingbo.ylb/projects/pg_spade_face_v4/results/face_enhan_dixian.wp_16x'
    #FOLDER = '/home/lingbo.ylb/projects/face_sr_sota/ESRGAN/results/ESRGAN-V1-FFHQ-16x'
    #FOLDER = '/home/lingbo.ylb/projects/face_sr_sota/WaveletSRNet-master/results'
    #FOLDER = '/home/lingbo.ylb/projects/face_sr_sota/mmsr/results/RRDB_ESRGAN_x4/FFHQ_scratch'
    #FOLDER = '/home/lingbo.ylb/projects/face_sr_sota/WaveletSRNet-master/results_16x_epoch30'
    #FOLDER='/home/lingbo.ylb/projects/face_sr_sota/Super-FAN/results_16x'
    
    debug = [
        #'/home/lingbo.ylb/datasets/FFHQ_degrade_512_test_3mix',
        #'/home/lingbo.ylb/projects/face_sr_sota/GFRNet/results/FFHQ_512',
        '/home/lingbo.ylb/projects/pg_spade_face_v4/results/conceptual_compression'
    ]
    
    # 18 folders, 5000 * 1.4s / 3600 -> 2h each ----> 36h? ---> 8 thread ---> 4.5h
    '''
    folders = [
        #'/home/lingbo.ylb/projects/pg_spade_face_v4/results/face_enhan_dixian.wp',
        '/home/lingbo.ylb/projects/pg_spade_face_v4/results/face_enhan_dixian.wp-full','/home/lingbo.ylb/projects/pg_spade_face_v4/results/face_enhan_dixian.wp_16x','/home/lingbo.ylb/projects/pg_spade_face_v4/results/face_enhan_dixian.wp_3_stage(4x)',
        '/home/lingbo.ylb/projects/pg_spade_face_v4/results/face_enhan_dixian.wp_no_pg',
        '/home/lingbo.ylb/projects/pg_spade_face_v4/results/face_enhan_dixian.wp_mixed',
        # wavelet
        '/home/lingbo.ylb/projects/face_sr_sota/WaveletSRNet-master/results',
        '/home/lingbo.ylb/projects/face_sr_sota/WaveletSRNet-master/results_16x_epoch30',
        '/home/lingbo.ylb/projects/face_sr_sota/WaveletSRNet-master/results_mixed',
        # super-fan
        '/home/lingbo.ylb/projects/face_sr_sota/Super-FAN/results',
        '/home/lingbo.ylb/projects/face_sr_sota/Super-FAN/results_16x',
        '/home/lingbo.ylb/projects/face_sr_sota/Super-FAN/results_mixed',
        # ESRGAN
        '/home/lingbo.ylb/projects/face_sr_sota/ESRGAN/results/ESRGAN-V1-FFHQ',
        
    folders = [
        '/home/lingbo.ylb/projects/face_sr_sota/ESRGAN/results/ESRGAN-V1-FFHQ-16x',
        '/home/lingbo.ylb/projects/face_sr_sota/ESRGAN/results/ESRGAN-V1-FFHQ-mixed',
        # SR
        '/home/lingbo.ylb/projects/face_sr_sota/mmsr/results/MSRGANx4/FFHQ_scratch/',
        '/home/lingbo.ylb/projects/face_sr_sota/mmsr/results/MSRResNetx4/FFHQ_scratch/',
        '/home/lingbo.ylb/projects/face_sr_sota/mmsr/results/RRDB_ESRGAN_x4/FFHQ_scratch/',
    ]
    
    folders = [#'/home/lingbo.ylb/projects/pg_spade_face_v4/results/face_enhan_lingbo.ylb_16x_contrasive_encoder',
    #'/home/lingbo.ylb/projects/pg_spade_face_v4/results/face_enhan_lingbo.ylb_noise_LIP_encoder',
    #'/home/lingbo.ylb/projects/face_sr_sota/SRFBN_CVPR19/results/SR/MyImage/SRFBN/x4'
    #'/home/lingbo.ylb/projects/face_sr_sota/SRFBN_CVPR19/results/SR/MyImage/SRFBN/x4'
    #'/home/lingbo.ylb/projects/face_sr_sota/VDNet-master/results/noise', 
    #'/home/lingbo.ylb/projects/face_sr_sota/RIDNet-master/TestCode/experiment/RIDNET_RNI15/results'
    #'/home/lingbo.ylb/projects/face_sr_sota/WaveletSRNet-master/results_noise',
    #'/home/lingbo.ylb/projects/face_sr_sota/EPGAN/results/FFHQ_jpeg',
    #'/home/lingbo.ylb/projects/face_sr_sota/DeblurGANv2/results/FFHQ_motion_blur',
    #'/home/lingbo.ylb/projects/face_sr_sota/DeblurGAN/results',
    #'/home/lingbo.ylb/projects/pg_spade_face_v4/results/face_enhan_lingbo.ylb_3mix_LIP_encoder',
    #'/home/lingbo.ylb/projects/pg_spade_face_v4/results/face_enhan_lingbo.ylb_jpeg_LIP_encoder',
    #'/home/lingbo.ylb/projects/pg_spade_face_v4/results/face_enhan_lingbo.ylb_motion_blur_LIP_encoder',
    #'/home/lingbo.ylb/projects/pg_spade_face_v4/results/face_enhan_lingbo.ylb_16x_LIP_encoder_L1_loss',
    #'/home/lingbo.ylb/projects/pg_spade_face_v4/results/face_enhan_lingbo.ylb_16x_spade',
    #'/home/lingbo.ylb/projects/face_sr_sota/PyTorch-ARCNN/results'
    ]
    '''
    folders = [
        '/home/lingbo.ylb/projects/face_sr_sota/Super-FAN/results_3mix',
        '/home/lingbo.ylb/projects/face_sr_sota/mmsr/results/RRDB_ESRGAN_x4/FFHQ_3mix',
        '/home/lingbo.ylb/projects/face_sr_sota/WaveletSRNet-master/results_3mix',
        '/home/lingbo.ylb/projects/face_sr_sota/DeblurGANv2/results/FFHQ_3mix',
        '/home/lingbo.ylb/projects/face_sr_sota/PyTorch-ARCNN/FFHQ_3mix',
        '/home/lingbo.ylb/projects/pg_spade_face_v4/results/face_enhan_lingbo.ylb_3mix_LIP_encoder',
    ]
    
    HEVC = [
        '/home/lingbo.ylb/datasets/FFHQ_yuv_recon/37',
        '/home/lingbo.ylb/datasets/FFHQ_yuv_recon/42',
        '/home/lingbo.ylb/datasets/FFHQ_yuv_recon/45',
        '/home/lingbo.ylb/datasets/FFHQ_yuv_recon/47',
        '/home/lingbo.ylb/datasets/FFHQ_yuv_recon/50',
    ]
    
    #for folder in folders:
    for folder in HEVC:
        main_process(folder)
    
