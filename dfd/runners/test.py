#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2021 The TARTRL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""""""


import sys
from PIL import Image
import numpy as np
import torch

from dfd.utils import check_file
from dfd.params import padding_image, img_mean, img_std, resize, DeepFakeModel
from dfd.timm.models import create_deepfake_model_v4

def test_img(model_path, img_files):
    assert all(check_file(img_file) for img_file in img_files), 'file not exist!'

    use_cuda = True
    use_half = True
    print('To load model from {}'.format(model_path))
    model = create_deepfake_model_v4(
        'efficientnet_deepfake_v4',
        num_classes=2,
        in_chans=12,
        checkpoint_path=model_path,
        strict=False)
    print('Model loaded!')
    model = DeepFakeModel(model)
    if use_cuda:
        model.cuda()
        if use_half:
            model.half()
    model.eval()
    for img_file in img_files:
        img = np.transpose(padding_image(resize(np.array(Image.open(img_file).convert('RGB'), np.uint8))), (2, 0, 1))
        img = torch.from_numpy(img).float()
        img = img.sub_(img_mean).div_(img_std)
        if use_cuda:
            img = img.cuda()
            if use_half:
                img = img.half()
        img = [torch.cat([img, img.clone(), img.clone(), img.clone()], dim=0)]
        with torch.no_grad():
            scores = model(torch.stack(img, dim=0))
        scores = scores.cpu().numpy()[:, 0].tolist()
        print('{}\'s fake score:{}'.format(img_file, scores[0]))


if __name__ == '__main__':
    model_path = '../models/model_half.pth.tar'
    if len(sys.argv) <= 1:
        print('Please input your images. e.g. python test_images.py image_path1 image_path2')
        exit()
    img_files = sys.argv[1:]
    test_img(model_path, img_files)