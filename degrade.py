'''
degrade.py: Apply degradations on clean images to acquire paired training samples.
Please modify the degradation type and source image directory before applying it.
'''
import cv2
import os
from tqdm import tqdm
import numpy as np
import imgaug.augmenters as ia

def get_down():
    return ia.Sequential([
        # random downsample between 4x to 8x and get back
        ia.Resize((0.125,0.25)),
        ia.Resize({"height": 512, "width": 512}),
    ])

def get_noise():
    return ia.OneOf([
        ia.AdditiveGaussianNoise(scale=(20,40), per_channel=True),
        ia.AdditiveLaplaceNoise(scale=(20,40), per_channel=True),
        ia.AdditivePoissonNoise(lam=(15,30), per_channel=True),
    ])

def get_blur():
    return ia.OneOf([
        ia.MotionBlur(k=(10,20)),
        ia.GaussianBlur((3.0, 8.0)),
    ])

def get_jpeg():
    return ia.JpegCompression(compression=(50,85))

def get_full():
    return ia.Sequential([
        get_blur(),
        get_noise(),
        get_jpeg(),
        get_down(),
    ], random_order=True)

def get_by_suffix(suffix):
    if suffix == 'down':
        return get_down()
    # elif...
    else:
        raise('%s not supported' % suffix)

def create_mixed_dataset(input_dir, suffix='full'):
    output_dir = input_dir + '_' + suffix
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    trans = get_by_suffix(suffix) # or use other functions
    mix_degrade = lambda x: trans.augment_image(x)

    for item in tqdm(os.listdir(input_dir)):
        # We arrange the data in [LR, HR] format
        # change the following script to fit your needs
        hr = cv2.imread(os.path.join(input_dir, item))
        lr = mix_degrade(hr)
        img = np.concatenate((lr,hr), axis=0)
        cv2.imwrite(os.path.join(output_dir, item), img)

if __name__ == '__main__':
    suffix = 'degradation_type' # [down/16x/noise/blur/jpeg/full]
    source_dir = '/path/to/your/source/directory'
    create_mixed_dataset(source_dir, suffix)
