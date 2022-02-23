import torch                                        # root package
from torch.utils.data import Dataset, DataLoader    # dataset representation and loading
from torchvision import datasets, models            # vision datasets,architectures & transforms
import torchvision.transforms as T                  # composable transforms

import imgaug as ia
from imgaug import augmenters as iaa
import os
from pathlib import Path
from glob import glob

from collections import Counter
import random
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageOps
from scipy.ndimage.interpolation import shift
from skimage.filters import threshold_otsu
import numpy as np
import torchvision
import random

# Set seed
ia.seed(42)

# weather
transform_weather = iaa.OneOf([ iaa.imgcorruptlike.Fog(severity=1),
                                iaa.imgcorruptlike.Frost(severity=1),
                                iaa.imgcorruptlike.Snow(severity=2),
                                iaa.imgcorruptlike.Spatter(severity=4),
                                iaa.Clouds(),
                                iaa.Rain(speed=(0.35, 0.45)),
                             ])
# transform
transform_ela = iaa.geometric.ElasticTransformation(alpha=(0, 40.0), sigma=(4.0, 8.0))

# noise
transform_noise = iaa.OneOf([   iaa.ImpulseNoise(0.09),
                                iaa.imgcorruptlike.ShotNoise(severity=1),
                                iaa.imgcorruptlike.SpeckleNoise(severity=2),
                                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                                ])

# cartoon
transform_artistic = iaa.artistic.Cartoon(blur_ksize=3, segmentation_size=(0.9, 1.1), saturation=(1.35, 1.75), edge_prevalence=(0.9, 1.1))

# blur
transform_blur = iaa.OneOf([iaa.imgcorruptlike.GlassBlur(severity=2),
                            iaa.imgcorruptlike.DefocusBlur(severity=1),
                            iaa.imgcorruptlike.MotionBlur(severity=2),
                            iaa.imgcorruptlike.ZoomBlur(severity=2),
                            iaa.GaussianBlur(sigma=(0, 0.5))
                              ])

transform_aff = iaa.Affine(
                        scale={"x": (0.7, 1.3), "y": (0.7, 1.3)},
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        rotate=(-20, 20),
                        shear=(-8, 8),
                        cval=0
                    )

transform_linear = iaa.SomeOf((0, 3), [
            iaa.LinearContrast((0.75, 1.5)),
            iaa.Multiply((0.7, 1.3), per_channel=0.2),
            transform_aff
            ])

transform_top1 = iaa.SomeOf(2, [
                            transform_artistic,
                            transform_noise,
                            transform_blur,
                            transform_ela,
                            transform_weather,
                          ], random_order=True)

transform = iaa.Sequential([
              transform_linear,
              transform_top1
])


images = np.load("./data/soft_test_data.npy")
labels = np.load("./data/soft_test_label.npy")

images_list = []
labels_list = []

for repeat in range(1):
    for idx in range(len(labels)):
        img = images[idx]
        label = labels[idx]
        # label_idx = label.argmax()
        trans_img = transform(image=img)
        trans_img_list = trans_img.tolist()
        images_list.append(trans_img_list)

        # soft_label = np.zeros(10)
        # soft_label[label_idx] += random.uniform(0, 10) # an unnormalized soft label vector
        labels_list.append(label)
        
    print(f"{len(labels_list)}/50000")


images_aug_ori_3w = np.array(images_list, dtype=np.uint8)
labels_aug_ori_3w = np.array(labels_list)

images_save = images_aug_ori_3w
labels_save = labels_aug_ori_3w

print(images_save.shape, images_save.dtype, labels_save.shape, labels_save.dtype)
np.save('st3/images_top1aug_1w.npy', images_save)
np.save('st3/labels_top1aug_1w.npy', labels_save)