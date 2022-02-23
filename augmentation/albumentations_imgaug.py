import numpy as np
from PIL import Image
import cv2
from albumentations import (
    IAAPerspective, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, PiecewiseAffine,
    Sharpen, Emboss, RandomBrightnessContrast, OneOf, Compose, Cutout, CoarseDropout, ShiftScaleRotate,
)
import torchvision
from tqdm.std import trange
from tqdm import tqdm
def get_train_transforms():
    return Compose(
        [
            Transpose(p=0.25),
            GaussNoise(p=0.75),
            OneOf([
                    # 模糊相关操作
                    MotionBlur(p=.75),
                    MedianBlur(blur_limit=3, p=0.5),
                    Blur(blur_limit=3, p=0.75),
                ], p=0.25),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.25),
            OneOf([
                # 畸变相关操作
                OpticalDistortion(p=0.75),
                GridDistortion(p=0.25),
                PiecewiseAffine(p=0.75),
            ], p=0.25),
            OneOf([
                    # 锐化、浮雕等操作
                    CLAHE(clip_limit=2),
                    Sharpen(),
                    Emboss(),
                    RandomBrightnessContrast(),            
                ], p=0.25),
            #
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            OneOf(
                [
                CoarseDropout(max_holes=4,
                            max_height=4,
                            max_width=4,
                            p=0.5),
                Cutout(
                    num_holes=4,
                    max_h_size=4,
                    max_w_size=4,
                    p=0.5,)],
                p=0.5)
        ]
        )
'''
image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))  
img = cv2.cvtColor(numpy.asarray(image),cv2.COLOR_RGB2BGR)  
'''

import imgaug as ia
from imgaug import augmenters as iaa

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


images = np.load("./best_data/best_data.npy")
labels = np.load("./best_data/best_soft_label.npy")

images_list = []
labels_list = []
albu_transform = get_train_transforms()

for idx in range(len(labels)):
    img = images[idx]
    label = labels[idx]
    r = np.random.randint(2)
    print(r)
    if r:
        trans_img = transform(image=img)
    else:
        trans_img = albu_transform(image=img)['image']
    images_list.append(trans_img)
    labels_list.append(label)
    
print(f"{len(labels_list)}/50000")


images_aug_ori_3w = np.array(images_list, dtype=np.uint8)
labels_aug_ori_3w = np.array(labels_list)

images_save = images_aug_ori_3w
labels_save = labels_aug_ori_3w

print(images_save.shape, images_save.dtype, labels_save.shape, labels_save.dtype)
np.save('se5/data_aug_1w66.npy', images_save)
np.save('se5/label_aug_1w66.npy', labels_save)

