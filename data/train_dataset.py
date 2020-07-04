"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import cv2, random
from random import randint, random
from data.base_dataset import BaseDataset, get_params, get_transform
import os, torch
from data.image_folder import make_dataset
import glob,os
import numpy as np
import traceback

class MyDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def initialize(self, opt):
        self.opt = opt

        files = [l for l in glob.glob(os.path.join(opt.dataroot, '*')) if l.lower().endswith(('jpg','png','jpeg'))]
        '''
        if opt.dataroot_assist is not None:
            files += [l for l in glob.glob(os.path.join(opt.dataroot_assist, '*')) if
                     l.lower().endswith(('jpg', 'png', 'jpeg'))]
        '''

        label_paths=image_paths=instance_paths = files

        label_paths = label_paths[:opt.max_dataset_size]
        image_paths = image_paths[:opt.max_dataset_size]
        instance_paths = instance_paths[:opt.max_dataset_size]

        self.label_paths = label_paths
        self.image_paths = image_paths
        self.instance_paths = instance_paths

        size = len(self.label_paths)
        self.dataset_size = size

    def read_face_pair(self, file):
        org = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
        if random() > 0.5:
            org = org[:, ::-1]
        h, w = org.shape[:2]
        sz = min(w, h)
        if h > w:
            dh = sz
            datas = [org[dh * i:dh * (i + 1), 0:w] for i in range(h // dh)]
        else:
            dw = sz
            datas = [org[0:h, dw * i:dw * (i + 1)] for i in range(w // dw)]

        label = datas[self.opt.choose_pair[0]]
        image = datas[self.opt.choose_pair[1]]

        label = cv2.resize(label, (self.opt.crop_size, self.opt.crop_size), interpolation=cv2.INTER_AREA)
        image = cv2.resize(image, (self.opt.crop_size, self.opt.crop_size), interpolation=cv2.INTER_CUBIC)

        label = label.transpose(2, 0, 1) / 255.
        label = torch.from_numpy(label).to(torch.float32)

        image = image.transpose(2, 0, 1) / 255.
        image = torch.from_numpy(image).to(torch.float32)

        input_dict = {'label': label, 'instance': 0,
                      'image': image, 'path': file}

        return input_dict

    # def read_film_image(self, file):
    #     org = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
    #     h, w = org.shape[:2]
    #     sz = randint(self.opt.crop_size, min(w, h))
    #     dx, dy = randint(0, w-sz), randint(0, h-sz)
    #
    #     image = org[dy:dy+sz, dx:dx+sz]
    #     if random() > 0.5:
    #         image = image[:, ::-1]
    #
    #     h, w = self.opt.crop_size, self.opt.crop_size
    #
    #     label = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    #     quality = randint(60, 100)
    #     label_en = cv2.imencode('.jpg', label, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
    #     label = cv2.imdecode(label_en, cv2.IMREAD_UNCHANGED)
    #
    #     image = cv2.resize(image, (4 * w, 4 * h), interpolation=cv2.INTER_CUBIC)
    #
    #     label = label.transpose(2, 0, 1) / 255.
    #     label = torch.from_numpy(label).to(torch.float32)
    #
    #     image = image.transpose(2, 0, 1) / 255.
    #     image = torch.from_numpy(image).to(torch.float32)
    #
    #     input_dict = {'label': label, 'instance': 0,
    #                   'image': image, 'path': file}
    #
    #     return input_dict

    def __getitem__(self, index):
        while True:
            try:
                # Label Image
                file = self.image_paths[index]
                return self.read_face_pair(file)
            except:
                index = randint(0, self.dataset_size-1)
                print(traceback.print_exc())

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size


class TrainDataset(MyDataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir, --instance_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = MyDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        load_size = 286 if is_train else 256
        parser.set_defaults(load_size=load_size)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=1)
        parser.set_defaults(contain_dontcare_label=False)

        parser.add_argument('--label_dir', type=str, required=True,
                            help='path to the directory that contains label images')
        parser.add_argument('--image_dir', type=str, required=True,
                            help='path to the directory that contains photo images')
        parser.add_argument('--instance_dir', type=str, default='',
                            help='path to the directory that contains instance maps. Leave black if not exists')
        return parser

    def get_paths(self, opt):
        label_dir = opt.label_dir
        label_paths = make_dataset(label_dir, recursive=False, read_cache=True)

        image_dir = opt.image_dir
        image_paths = make_dataset(image_dir, recursive=False, read_cache=True)

        if len(opt.instance_dir) > 0:
            instance_dir = opt.instance_dir
            instance_paths = make_dataset(instance_dir, recursive=False, read_cache=True)
        else:
            instance_paths = []

        assert len(label_paths) == len(image_paths), "The #images in %s and %s do not match. Is there something wrong?"

        return label_paths, image_paths, instance_paths
