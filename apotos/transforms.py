
# coding: utf-8

# In[2]:


import random
import math

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
    Compose,Resize,RandomScale,RandomGamma,RandomCrop,CenterCrop,RandomSizedCrop
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

train_transform = Compose([
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
])


## TTAの時は、rotation, random vertical flipとか入れてもよいかもしれない。
test_transform = Compose([
    HorizontalFlip(),
 #   Resize(width=IMG_SIZE,height=IMG_SIZE),
  #  RandomSizedCrop(min_max_height = MIN_MAX_HEIGHT,
  #         width = IMG_SIZE,
  #         height = IMG_SIZE)
 #   Resize(600,600),
    Rotate((-ROTATE, ROTATE)),
    RandomCrop(IMG_SIZE,IMG_SIZE),
])

tensor_transform = torchvision.transforms.Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

