
# coding: utf-8

# In[2]:


import random
import math
import numpy as np

from PIL import Image
import torchvision
from torchvision.transforms import (
    ToTensor, Normalize,)
    #Compose,
   # Resize, CenterCrop, RandomCrop,
   # RandomHorizontalFlip,RandomRotation)

#from albumentations import RandomBrightnessContrast#,Compose

from albumentations import (Rotate,
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, 
    Compose,Resize,RandomScale,RandomGamma,RandomCrop,CenterCrop,RandomSizedCrop,Cutout,RandomSunFlare
)

# ==========================================================================
# パイパーパラメータの設定
IMG_SIZE = 288
ROTATE = 180
# ==========================================================================

class _RandomSizedCrop:
    """Random crop the given PIL.Image to a random size
    of the original size and and a random aspect ratio
    of the original aspect ratio.
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR,
                 min_aspect=4/5, max_aspect=5/4,
                 min_area=0.25, max_area=1):
        self.size = size
        self.interpolation = interpolation
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.min_area = min_area
        self.max_area = max_area

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(self.min_area, self.max_area) * area
            aspect_ratio = random.uniform(self.min_aspect, self.max_aspect)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))

                return img.resize((self.size, self.size), self.interpolation)

        # Fallback
        scale = Resize(self.size, interpolation=self.interpolation)
        crop = CenterCrop(self.size)
        return crop(scale(img))


# Resize(IMG_SIZE)にすると、正方形ではない画像が出力されるバグ(バグなのか知らんが)を確認した。
# Cropを追加
# 1. RandomSizedCrop
# 2. CenterCrop
# 3. RandomCrop
# One_ofを使ってもよい気がしてきた。

#MIN_MAX_HEIGHT = [IMG_SIZE/2,IMG_SIZE]

# 下記、Kernelを参考にData Augmentationを作成
'''Use case from https://www.kaggle.com/alexanderliao/image-augmentation-demo-with-albumentation/'''

'''1. Rotate or Flip'''
aug1 = OneOf([
    Rotate(p=0.99, limit=160, border_mode=0,value=0), # value=black
    Flip(p=0.5)
    ],p=1)

'''2. Adjust Brightness or Contrast'''
aug2 = RandomBrightnessContrast(brightness_limit=0.45, contrast_limit=0.45,p=1)
h_min=np.round(IMG_SIZE*0.72).astype(int)
h_max= np.round(IMG_SIZE*0.9).astype(int)
#print(h_min,h_max)

'''3. Random Crop and then Resize'''
#w2h_ratio = aspect ratio of cropping
aug3 = RandomSizedCrop((h_min, h_max),IMG_SIZE,IMG_SIZE, w2h_ratio=IMG_SIZE/IMG_SIZE,p=1)

'''4. CutOut Augmentation'''
max_hole_size = int(IMG_SIZE/5)
aug4 = Cutout(p=1,max_h_size=max_hole_size,max_w_size=max_hole_size,num_holes=8 )#default num_holes=8

'''5. SunFlare Augmentation'''
aug5 = RandomSunFlare(src_radius=max_hole_size,
                      num_flare_circles_lower=10,
                      num_flare_circles_upper=20,
                      p=1)#default flare_roi=(0,0,1,0.5),

# 学習時のData Augmentationを作成
train_transform = Compose([
    aug1,
    aug2,
  #  aug3,
    aug4,
    aug5,
    Resize(IMG_SIZE,IMG_SIZE),
],p=1)

"""Compose([
  #  RandomCrop(288),
    HorizontalFlip(),
    Rotate((-ROTATE, ROTATE)),
    RandomBrightnessContrast(),
#    HueSaturationValue(),
    RandomScale(),
    RandomGamma(),
  #  Resize(width=IMG_SIZE,height=IMG_SIZE),
  #  RandomSizedCrop(min_max_height = MIN_MAX_HEIGHT,
   #            width = IMG_SIZE,
   #            height = IMG_SIZE)
   # Resize(INPUT_IMG_SIZE,INPUT_IMG_SIZE),
    OneOf([CenterCrop(IMG_SIZE,IMG_SIZE)],p=0.5),
    OneOf([RandomCrop(IMG_SIZE,IMG_SIZE)],p=1),
])"""

# Validation, Test時のData Augmentationを定義
## TTAの時は、rotation, random vertical flipとか入れてもよいかもしれない。
test_transform = Compose([
 #   Rotate((-ROTATE, ROTATE)),
 #   Flip(p=0.5),
 #   Resize(IMG_SIZE,IMG_SIZE)
    aug1,
    aug2,
    Resize(IMG_SIZE,IMG_SIZE)
    #aug3,
    #aug4,
    #aug5
],p=1)


"""Compose([
    HorizontalFlip(),
 #   Resize(width=IMG_SIZE,height=IMG_SIZE),
  #  RandomSizedCrop(min_max_height = MIN_MAX_HEIGHT,
  #         width = IMG_SIZE,
  #         height = IMG_SIZE)
 #   Resize(600,600),
    Rotate((-ROTATE, ROTATE)),
    RandomCrop(IMG_SIZE,IMG_SIZE),
])"""

tensor_transform = torchvision.transforms.Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

