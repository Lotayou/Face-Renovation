import os
import numpy as np
from skimage.io import imread
from tf_frechet_inception_distance import get_fid
from tf_inception_score import get_inception_score

def padding(x):
    _h, _w, _c = x.shape
    _im = (np.ones((_h, _h, _c)) * 255).astype(np.uint8)
    _left = (_h - _w) // 2
    _im[:, _left: _left + _w, :] = x
    return _im
    
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    #name = 'TWO_STAGE_FULL_20191018'
    name = '../../results/debug'
    testing_dir = os.path.join('../results/%s' % name)
    h, w = 256, 176
    '''
    gt_images = []
    g1_images = []
    g2_images = []
    
    for item in os.listdir(testing_dir):
        test_img = imread(os.path.join(testing_dir, item))
        gt_images.append(padding(test_img[h:,:w]))
        g1_images.append(padding(test_img[:h,2*w:]))
        g2_images.append(padding(test_img[h:,2*w:]))
        
    print('Image loading complete')
    gt_images = np.stack(gt_images, axis=0).transpose(0,3,1,2)
    g1_images = np.stack(g1_images, axis=0).transpose(0,3,1,2)
    g2_images = np.stack(g2_images, axis=0).transpose(0,3,1,2)
        
    is_mean1, is_std1 = get_inception_score(g1_images)
    is_mean2, is_std2 = get_inception_score(g2_images)
    
    fid1 = get_fid(g1_images, gt_images)
    fid2 = get_fid(g2_images, gt_images)
    
    print('Stage I: IS = %.6f +- %.6f, FID = %.6f' % (is_mean1, is_std1, fid1))
    print('Stage II: IS = %.6f +- %.6f, FID = %.6f' % (is_mean2, is_std2, fid2))
    '''
    
    gt_images = []
    patn_images = []
    
    for item in os.listdir(testing_dir):
        test_img = imread(os.path.join(testing_dir, item))
        gt_images.append(test_img[:,1024:])
        patn_images.append(test_img[:,512:1024])
        
    gt_images = np.stack(gt_images, axis=0).transpose(0,3,1,2)
    patn_images = np.stack(patn_images, axis=0).transpose(0,3,1,2)
    
    is_mean, is_std = get_inception_score(patn_images)
    fid = get_fid(patn_images, gt_images)
    
    print('PATN baseline: IS = %.6f +- %.6f, FID = %.6f' % (is_mean, is_std, fid))
    
