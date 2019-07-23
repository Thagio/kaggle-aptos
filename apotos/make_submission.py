
# coding: utf-8

# In[9]:


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


# In[10]:


import argparse
import os
import pandas as pd
from IPython.core.debugger import Pdb

ON_KAGGLE: bool = 'KAGGLE_WORKING_DIR' in os.environ

if ON_KAGGLE:
    from .utils import mean_df
    from .dataset import DATA_ROOT
    from .main import binarize_prediction
else:
    from utils import mean_df
    from dataset import DATA_ROOT
    from main import binarize_prediction


# In[28]:


def main(*args):
    
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
 #   Pdb().set_trace()
    arg('--predictions', nargs='+')
    arg('--output')
    arg('--threshold', type=float, default=0.2)
    
    if is_env_notebook():
        args = parser.parse_args(args=args[0])
    else:
        args = parser.parse_args()
    
    sample_submission = pd.read_csv(
        DATA_ROOT / 'sample_submission.csv', index_col='id_code')
    dfs = []
    
    for prediction in args.predictions:
        #Pdb().set_trace()
        df = pd.read_hdf(prediction, index_col='id_code')
        df = df.reindex(sample_submission.index)
        dfs.append(df)
        
    df = pd.concat(dfs)
    df = mean_df(df)
   # Pdb().set_trace()
    
  #  df[:] = binarize_prediction(df.values, threshold=args.threshold)
    df["diagnosis"] = df.values.argmax(axis=1)
   # df = df.apply(get_classes, axis=1)
    #df.name = 'diagnosis'
    df.loc[:,["diagnosis"]].to_csv(args.output, header=True)

def get_classes(item):
    return ' '.join(cls for cls, is_present in item.items() if is_present)


# In[29]:


if __name__ == '__main__':
    
    args = ["--predictions","model_1/test.h5",
            "--output","submission.csv"]
    main(args)

