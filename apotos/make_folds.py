
# coding: utf-8

# In[1]:


#import sys
#sys.path.append('.')

import argparse
from collections import defaultdict, Counter
import random

import pandas as pd
import tqdm

from dataset import DATA_ROOT
#from utils import is_env_notebook


# In[2]:


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


# In[28]:


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

def make_folds(n_folds:int) -> pd.DataFrame:
    df = pd.read_csv(DATA_ROOT / 'train.csv')
    cls_counts = Counter(cls for cls in df["diagnosis"])
    
    fold_cls_counts = defaultdict(int)
    folds = [-1] * len(df)
    for item in tqdm.tqdm(df.sample(frac=1, random_state=42).itertuples(),
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


# In[32]:


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

if __name__ == '__main__':
    main()

