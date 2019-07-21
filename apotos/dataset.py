
# coding: utf-8

# In[3]:


from pathlib import Path
from typing import Callable, List

import cv2
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from transforms import tensor_transform
from utils import ON_KAGGLE


# In[4]:


N_CLASSES = 5
DATA_ROOT = Path('../input/aptos2019-blindness-detection' if ON_KAGGLE else './data')

class TrainDataset(Dataset):
    def __init__(self, root: Path, df: pd.DataFrame,
                 image_transform: Callable, debug: bool = True):
        super().__init__()
        self._root = root
        self._df = df
        self._image_transform = image_transform
        self._debug = debug

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx: int):
        item = self._df.iloc[idx]
        
        #print(item)
        #print(self._root)
        image = load_transform_image(
            item, self._root, self._image_transform, debug=self._debug)
        target = torch.zeros(N_CLASSES)
    #    for cls in item.attribute_ids.split():
    #        target[int(cls)] = 1
        cls = item.diagnosis
        target[int(cls)] = 1
        return image, target


class TTADataset:
    def __init__(self, root: Path, df: pd.DataFrame,
                 image_transform: Callable, tta: int):
        self._root = root
        self._df = df
        self._image_transform = image_transform
        self._tta = tta

    def __len__(self):
        return len(self._df) * self._tta

    def __getitem__(self, idx):
        item = self._df.iloc[idx % len(self._df)]
        image = load_transform_image(item, self._root, self._image_transform)
        return image, item.id_code


def load_transform_image(
        item, root: Path, image_transform: Callable, debug: bool = False):
    image = load_image(item, root)
    image = image_transform(image)
    if debug:
        image.save('_debug.png')
    return tensor_transform(image)

def get_ids(root: Path) -> List[str]:
    return sorted({p.name.split('_')[0] for p in root.glob('*.png')})


# In[5]:


def load_image(item, root: Path) -> Image.Image:
    image = cv2.imread(str(root / f'{item.id_code}.png'))
    
  #  from IPython.core.debugger import Pdb; Pdb().set_trace()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)


# In[9]:


if __name__ == "__main__":
    # load_imageの実行テスト
    import pandas as pd
    
    df = pd.read_csv(DATA_ROOT / 'train.csv')
    item = df.iloc[1]
    image = load_image(item,DATA_ROOT)
   


# In[1]:


if __name__ == "__main__":
    pass
 #   image_transformed = load_transform_image(item,DATA_ROOT,image_transform)

