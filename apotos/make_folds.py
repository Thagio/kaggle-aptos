
# coding: utf-8

# In[1]:


# FIXME  : 以下の関数は定義されたファイルの形式に依存するので、utilsに記載できない。
def is_env_notebook():
    """Determine wheather is the environment Jupyter Notebook"""
    if 'get_ipython' not in globals():
        # Python shell
        return False
    env_name = get_ipython().__class__.__name__
    if env_name == 'TerminalInteractiveShell':
        # IPython shell
        return False
    # Jupyter Notebook
    return True


# In[2]:


#import sys
#sys.path.append('.')

import argparse
from collections import defaultdict, Counter
import random
import os

import pandas as pd
import tqdm

ON_KAGGLE: bool = 'KAGGLE_WORKING_DIR' in os.environ

if ON_KAGGLE:
    from .dataset import DATA_ROOT,EXTERNAL_ROOT
else:
    from dataset import DATA_ROOT,EXTERNAL_ROOT


# In[3]:


# make_foldsはマルチラベル用になってる。
def make_folds_for_multilabel(n_folds: int) -> pd.DataFrame:
    df = pd.read_csv(DATA_ROOT / 'train.csv')
    cls_counts = Counter(cls for classes in df['attribute_ids'].str.split()
                         for cls in classes)
    fold_cls_counts = defaultdict(int)
    folds = [-1] * len(df)
    for item in tqdm.tqdm(df.sample(frac=1, random_state=42).itertuples(),
                          total=len(df)):
        cls = min(item.attribute_ids.split(), key=lambda cls: cls_counts[cls])
        fold_counts = [(f, fold_cls_counts[f, cls]) for f in range(n_folds)]
        min_count = min([count for _, count in fold_counts])
        random.seed(item.Index)
        fold = random.choice([f for f, count in fold_counts
                              if count == min_count])
        folds[item.Index] = fold
        for cls in item.attribute_ids.split():
            fold_cls_counts[fold, cls] += 1
    df['fold'] = folds
    return df

def make_folds(n_folds:int,seed:int=42) -> pd.DataFrame:
    df = pd.read_csv(DATA_ROOT / 'train.csv')
    cls_counts = Counter(cls for cls in df["diagnosis"])
    
    fold_cls_counts = defaultdict(int)
    folds = [-1] * len(df)
    for item in tqdm.tqdm(df.sample(frac=1, random_state=seed).itertuples(),
                      total=len(df)):
    
        cls = item.diagnosis
        fold_counts = [(f, fold_cls_counts[f, cls]) for f in range(n_folds)]
        min_count = min([count for _, count in fold_counts])
        random.seed(item.Index)
        fold = random.choice([f for f, count in fold_counts
                              if count == min_count])
        folds[item.Index] = fold
        #for cls in item.diagnosis:
        fold_cls_counts[fold, cls] += 1
        
#   from IPython.core.debugger import Pdb; Pdb().set_trace()
    df['fold'] = folds
    
    return df


# In[12]:


def external_data() -> pd.DataFrame:
    df = pd.read_csv(EXTERNAL_ROOT / "trainLabels.csv")            .rename(columns = {"id_code":"image","diagnosis":"level"})
    
    df["fold"] = 99
    
    return df


# In[13]:


if __name__ == "__main__":
    pass
    # df = external_data()
   # print(df.head())


# In[4]:


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-folds', type=int, default=4)
    
    ## jupyter-notebookかどうか判定
    if is_env_notebook():
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()
        
    df = make_folds(n_folds=args.n_folds)
    df.to_csv('folds.csv', index=None)
   # from IPython.core.debugger import Pdb; Pdb().set_trace()

if __name__ == '__main__':
    main()

