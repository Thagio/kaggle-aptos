
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


import argparse
from itertools import islice
import json
from pathlib import Path
import shutil
import warnings
from typing import Dict
import os

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
from sklearn.exceptions import UndefinedMetricWarning
import torch
from torch import nn, cuda
from torch.optim import Adam
import tqdm

import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score

from IPython.core.debugger import Pdb


# In[3]:


ON_KAGGLE: bool = 'KAGGLE_WORKING_DIR' in os.environ

if ON_KAGGLE:
    from . import models
    from .dataset import TrainDataset, TTADataset, get_ids, N_CLASSES, DATA_ROOT
    from .transforms import train_transform, test_transform
    from .utils import (
        write_event, load_model, mean_df, ThreadingDataLoader as DataLoader,
        ON_KAGGLE)
else:
    import models
    from dataset import TrainDataset, TTADataset, get_ids, N_CLASSES, DATA_ROOT
    from transforms import train_transform, test_transform
    from utils import (
        write_event, load_model, mean_df, ThreadingDataLoader as DataLoader,
        ON_KAGGLE)


# In[8]:


def main(*args):
#def main():   
   # print("do main")
    
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    
    # TODO : "--modeにGradCAMを追加"
    
    arg('--mode', choices=['train', 'validate', 'predict_valid', 'predict_test'])
    arg('--run_root')
    arg('--model', default='resnet50')
    arg('--loss',default="focalloss")
    arg('--pretrained', type=int, default=1)
    arg('--batch-size', type=int, default=64)
    arg('--step', type=int, default=1)
    arg('--workers', type=int, default=4 if ON_KAGGLE else 4)
    arg('--lr', type=float, default=1e-4)
    arg('--patience', type=int, default=4)
    arg('--clean', action='store_true')
    arg('--n-epochs', type=int, default=100)
    arg('--epoch-size', type=int)
    arg('--tta', type=int, default=4)
    arg('--use-sample', action='store_true', help='use a sample of the dataset')
    arg('--debug', action='store_true')
    arg('--limit', type=int)
    arg('--fold', type=int, default=0)
    arg('--regression',type=int,default=0)
    arg('--finetuning',type=int,default=1)
    # TODO : classificationかregressionかをオプションで追加できるようにする。
  
   # from IPython.core.debugger import Pdb; Pdb().set_trace()
    if is_env_notebook():       
        args = parser.parse_args(args=args[0])
    else:
        args = parser.parse_args()

 #   from IPython.core.debugger import Pdb; Pdb().set_trace()
    run_root = Path(args.run_root)
    folds = pd.read_csv('folds.csv')
    
    train_root = DATA_ROOT / ('train_sample' if args.use_sample else 'train_images')
    
    if args.use_sample:
        folds = folds[folds['Id'].isin(set(get_ids(train_root)))]
    
  #  Pdb().set_trace()
    # -1 はleakデータ
    train_fold = folds[folds['fold'] != args.fold]
    leak_fold = folds[folds['fold'] == -1]
    train_fold = pd.concat([train_fold,leak_fold])
    
    valid_fold = folds[folds['fold'] == args.fold]
    
    
    if args.limit:
        train_fold = train_fold[:args.limit]
        valid_fold = valid_fold[:args.limit]
        
    
    def make_loader(df: pd.DataFrame, image_transform,regression=args.regression,shuffle=False,balanced=True) -> DataLoader:
        return DataLoader(
            TrainDataset(train_root, df, image_transform, debug=args.debug,regression=regression,balanced=balanced),
            shuffle=shuffle,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )
    
    
    ## TODO : regressionようにモデルを書き換え
    
    if args.regression:
        criterion = nn.MSELoss()
        # TODO : 回帰モデルへ変更
        model = getattr(models, args.model)(
            num_classes=1, pretrained=args.pretrained)

    else:
        # 分類モデル
        criterion = FocalLoss()#nn.BCEWithLogitsLoss(reduction='none') 
        model = getattr(models, args.model)(
            num_classes=N_CLASSES, pretrained=args.pretrained)

 #   Pdb().set_trace()
    
    use_cuda = cuda.is_available()
    fresh_params = list(model.fresh_params())
    all_params = list(model.parameters())
    if use_cuda:
        model = model.cuda()

    if args.mode == 'train':
        if run_root.exists() and args.clean:
            shutil.rmtree(run_root)
        run_root.mkdir(exist_ok=True, parents=True)
        (run_root / 'params.json').write_text(
            json.dumps(vars(args), indent=4, sort_keys=True))

        train_loader = make_loader(train_fold, train_transform,regression=args.regression,balanced=True)
        valid_loader = make_loader(valid_fold, test_transform,regression=args.regression,balanced=False)
        print(f'{len(train_loader.dataset):,} items in train, '
              f'{len(valid_loader.dataset):,} in valid')

        train_kwargs = dict(
            args=args,
            model=model,
            criterion=criterion,
            train_loader=train_loader,
            valid_loader=valid_loader,
            patience=args.patience,
            init_optimizer=lambda params, lr: Adam(params, lr),
            use_cuda=use_cuda,
        )
     #   from IPython.core.debugger import Pdb; Pdb().set_trace()
        if args.pretrained:
            if train(params=fresh_params, n_epochs=1, **train_kwargs):
                train(params=all_params, **train_kwargs)
        else:
            train(params=all_params, **train_kwargs)
            
        # fine-tunig after balanced learning 
        if args.finetuning:
            print("Start Fine-tuning")
            TUNING_EPOCH = 5
            train_loader = make_loader(train_fold, train_transform,regression=args.regression,balanced=False)
            # 学習率を小さくする
            args.lr = args.lr / 5
            train_kwargs["train_loader"] = train_loader
            train(params=all_params,n_epochs=args.n_epochs+TUNING_EPOCH,**train_kwargs,finetuning=args.finetuning)

    elif args.mode == 'validate':
        valid_loader = make_loader(valid_fold, test_transform)
        load_model(model, run_root / 'model.pt')
        validation(model, criterion, tqdm.tqdm(valid_loader, desc='Validation',valid_fold=valid_fold),
                   use_cuda=use_cuda)

    elif args.mode.startswith('predict'):
        print("load model predict")
        load_model(model, run_root / 'best-model.pt')
        predict_kwargs = dict(
            batch_size=args.batch_size,
            tta=args.tta,
            use_cuda=use_cuda,
            workers=args.workers,
        )
        if args.mode == 'predict_valid':
            #predict(model, df=valid_fold, root=train_root,
            #        out_path=run_root / 'val.h5',
            #        **predict_kwargs)
            
            valid_loader = make_loader(valid_fold, test_transform,shuffle=False,balanced=False)
            #model: nn.Module, criterion, valid_loader, use_cuda,valid_predict:bool=False
            
            # TODO : valid foldに予測結果をくっ付ける操作を追加
            validation(model,criterion,valid_loader,use_cuda,valid_fold=valid_fold,valid_predict=True,save_path=run_root)
                        
        elif args.mode == 'predict_test':
            test_root = DATA_ROOT / (
                'test_sample' if args.use_sample else 'test_images')
            ss = pd.read_csv(DATA_ROOT / 'sample_submission.csv')
            if args.use_sample:
                ss = ss[ss['id'].isin(set(get_ids(test_root)))]
            if args.limit:
                ss = ss[:args.limit]
            predict(model, df=ss, root=test_root,
                    out_path=run_root / 'test.h5',
                    **predict_kwargs)


def predict(model, root: Path, df: pd.DataFrame, out_path: Path,
            batch_size: int, tta: int, workers: int, use_cuda: bool):
    loader = DataLoader(
        dataset=TTADataset(root, df, test_transform, tta=tta),
        shuffle=False,
        batch_size=batch_size,
        num_workers=workers,
    )
    model.eval()
    all_outputs, all_ids = [], []
    with torch.no_grad():
        for inputs, ids in tqdm.tqdm(loader, desc='Predict'):
            if use_cuda:
                inputs = inputs.cuda()
            outputs = torch.sigmoid(model(inputs))
            all_outputs.append(outputs.data.cpu().numpy())
            all_ids.extend(ids)
            
    df = pd.DataFrame(
        data=np.concatenate(all_outputs),
        index=all_ids,
        columns=map(str, range(N_CLASSES)))
    df = mean_df(df)
    df.to_hdf(out_path, 'prob', index_label='id')
    print(f'Saved predictions to {out_path}')


def train(args, model: nn.Module, criterion, *, params,
          train_loader, valid_loader, init_optimizer, use_cuda,
          n_epochs=None, patience=2, max_lr_changes=2,finetuning=False) -> bool:
    
    lr = args.lr
    n_epochs = n_epochs or args.n_epochs
    params = list(params)
    optimizer = init_optimizer(params, lr)

    run_root = Path(args.run_root)
    model_path = run_root / 'model.pt'
    best_model_path = run_root / 'best-model.pt'
    
    if model_path.exists():
        state = load_model(model, model_path)
        epoch = state['epoch']
        step = state['step']
        best_valid_loss = state['best_valid_loss']
        
    if best_model_path.exists() and finetuning:
        state = load_model(model,best_model_path)
        #epoch = 1
        #step = 0
        epoch = state['epoch']
        step = state['step']
        best_valid_loss = state['best_valid_loss']
    
    else:
        epoch = 1
        step = 0
        best_valid_loss = float('inf')
        
    lr_changes = 0

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
        'best_valid_loss': best_valid_loss
    }, str(model_path))

    report_each = 10
    log = run_root.joinpath('train.log').open('at', encoding='utf8')
    valid_losses = []
    lr_reset_epoch = epoch
    for epoch in range(epoch, n_epochs + 1):
        model.train()
        tq = tqdm.tqdm(total=(args.epoch_size or
                              len(train_loader) * args.batch_size))
        tq.set_description(f'Epoch {epoch}, lr {lr}')
        losses = []
        tl = train_loader
     #   from IPython.core.debugger import Pdb; Pdb().set_trace()
        if args.epoch_size:
            tl = islice(tl, args.epoch_size // args.batch_size)
            
        try:
            mean_loss = 0
          #  Pdb().set_trace()
            for i, (inputs, targets) in enumerate(tl):        
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, targets)#_reduce_loss(criterion(outputs, targets))
                batch_size = inputs.size(0)
                (batch_size * loss).backward()
                if (i + 1) % args.step == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    step += 1
                tq.update(batch_size)
                losses.append(loss.item())
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss=f'{mean_loss:.3f}')
                if i and i % report_each == 0:
                    write_event(log, step, loss=mean_loss)
          #  Pdb().set_trace()
            write_event(log, step, loss=mean_loss)
            tq.close()
            save(epoch + 1)
            valid_metrics = validation(model, criterion, valid_loader, use_cuda)
            
         #   Pdb().set_trace()
            write_event(log, step, **valid_metrics)
            
            valid_loss = valid_metrics['valid_loss']
            valid_losses.append(valid_loss)
            
        #    from IPython.core.debugger import Pdb; Pdb().set_trace()
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                shutil.copy(str(model_path), str(best_model_path))
            elif (patience and epoch - lr_reset_epoch > patience and
                  min(valid_losses[-patience:]) > best_valid_loss):
                # "patience" epochs without improvement
                lr_changes +=1
                if lr_changes > max_lr_changes:
                    break
                lr /= 5
                print(f'lr updated to {lr}')
                lr_reset_epoch = epoch
                optimizer = init_optimizer(params, lr)
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return False
    return True


def validation(
        model: nn.Module, criterion, valid_loader, use_cuda,valid_fold=None,valid_predict:bool=False,save_path:Path=""
        ) -> Dict[str, float]:
    model.eval()
    all_losses, all_predictions, all_targets = [], [], []
    with torch.no_grad():
        for inputs, targets in valid_loader:
            all_targets.append(targets.numpy().copy())
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
          #  all_losses.append(_reduce_loss(loss).item())
            all_losses.append(loss.item())
            predictions = torch.sigmoid(outputs)
            all_predictions.append(predictions.cpu().numpy())
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
#    Pdb().set_trace()

    def get_score(y_pred):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UndefinedMetricWarning)
          #  return fbeta_score(
          #      all_targets, y_pred, beta=2, average='samples')
            return qk(y_pred,all_targets)

    metrics = {}
    #argsorted = all_predictions.argsort(axis=1)
    
   # Pdb().set_trace()
    #for threshold in [0.05, 0.10, 0.15, 0.20]:
    #    metrics[f'valid_f2_th_{threshold:.2f}'] = get_score(
    #        binarize_prediction(all_predictions, threshold, argsorted))
    #metrics = get_score(all_predictions) 
    
    if valid_predict:
     #   Pdb().set_trace()
        # TOOD : 予測結果をpd データフレーム形式で保存
        valid_fold["prediction"] = all_predictions.argmax(axis=1)
        valid_fold.to_csv(save_path / "valid_prediction.csv",index=False)
        
       # run_root = Path(args.run_root)
        with open(save_path / "best_score.txt",mode="w") as f:
            f.write("best valid kapa : {score}".format(score=get_score(all_predictions)))
            f.write("best valid loss : {loss}".format(loss=np.mean(all_losses)))
    
 #   from IPython.core.debugger import Pdb; Pdb().set_trace()
    metrics['valid_kapa'] = get_score(all_predictions)
    metrics['valid_loss'] = np.mean(all_losses)
    #print(' | '.join(f'{k} {v:.3f}' for k, v in sorted(
    #    metrics.items(), key=lambda kv: -kv[1])))
    print(metrics)

    return metrics


def visualization():
    """
    GRAD-CAMによるNNが判断の根拠としている領域の可視化
    
    """
    
    pass


def binarize_prediction(probabilities, threshold: float, argsorted=None,
                        min_labels=1, max_labels=10):
    """ Return matrix of 0/1 predictions, same shape as probabilities.
    """
    assert probabilities.shape[1] == N_CLASSES
    if argsorted is None:
        argsorted = probabilities.argsort(axis=1)
    max_mask = _make_mask(argsorted, max_labels)
    min_mask = _make_mask(argsorted, min_labels)
    prob_mask = probabilities > threshold
    return (max_mask & prob_mask) | min_mask


def _make_mask(argsorted, top_n: int):
    mask = np.zeros_like(argsorted, dtype=np.uint8)
    col_indices = argsorted[:, -top_n:].reshape(-1)
    row_indices = [i // top_n for i in range(len(col_indices))]
    mask[row_indices, col_indices] = 1
    return mask


def _reduce_loss(loss):
    return loss.sum() / loss.shape[0]

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logit, target):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val +                ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size())==2:
            loss = loss.sum(dim=1)
        return loss.mean()
    
def qk(y_pred, y):
   ## Pdb().set_trace()
    #y_pred = torch.from_numpy(y_pred)
    y_pred = np.argmax(y_pred,axis=1)
    y = np.argmax(y,axis=1)
    #y = torch.argmax(y,dim=1)
  #  Pdb().set_trace()
    return cohen_kappa_score(y_pred, y, weights='quadratic')
    #return torch.tensor(cohen_kappa_score(torch.round(y_pred), y, weights='quadratic'), device='cuda:0')


# In[4]:


if __name__ == '__main__' and ON_KAGGLE:
    main()


# In[ ]:


if __name__ == '__main__' and  not(ON_KAGGLE):
    import gc
    
    ###########################################################
    # FOLDを修正
    
    folds = [0,1,2,3]
             #1,2,3]#[0,1,
             
  #  N_EPOCH = 25
  #  model_name = "02_brightness_cotrast"
    #model_name = "HueSaturationValue"
    #model_name = "RandomGamma"
  #  model_name = "regression"
  #  model_name = "RandomSizeCrop_validation"
  #  model_name = "CentorCrop_Oneof"
    #model_name = "CentorCrop_Rotation_Val"
  #  model_name = "Rockman_aug_nonCircleCrop"
   # model_name = "Rockman_aug_CircleCrop"
  #  model_name = "10_test"
#    model_name = "11_No_Crop_balanced_finetuning"
  #  model_name = "12_add_11_scale_and_gamma"
   # model_name = "13_nodup"
    model_name = "14_nodup_refine"
    
    N_EPOCH = 10
    # limit変更
    
    for fold in folds:
        # 学習
        # jupyter-notebookの場合、ここで引数を選択しないといけない。
        train_args = ["--mode","train",
                   "--run_root","{model_name}_{fold}".format(model_name=model_name,fold=fold),
               #    "--limit","100", # TODO : 適宜変更
                    "--fold","{fold}".format(fold=fold),
                   "--n-epochs","{epoch}".format(epoch=N_EPOCH),
                   '--workers',"16",
                      '--patience',"2",
                      "--finetuning","1"
                    # "--regression","1"
                     ]
        
        main(train_args)
        
        # validation
        val_args = ["--mode","predict_valid",
               "--run_root","{model_name}_{fold}".format(model_name=model_name,fold=fold),
             #  "--limit","100"
                   ]
        main(val_args)
        
        gc.collect()
    #    break
        
        #print(N_CLASSES)


# In[12]:


if __name__ == '__main__' and not(ON_KAGGLE):
    
    model_name = "CentorCrop_Rotation_Val/CentorCrop_Rotation_Val_0"
    # jupyter-notebookの場合、ここで引数を選択しないといけない。
    arg_list = ["--mode","predict_test",
               "--run_root",model_name,
           #    "--limit","100",
                "--tta","4"
               ]
    main(arg_list)
    #print(N_CLASSES)


# In[7]:


if __name__ == '__main__' and not(ON_KAGGLE):
    # jupyter-notebookの場合、ここで引数を選択しないといけない。
    model_name = model_name + "_0"
    
    arg_list = ["--mode","predict_valid",
               "--run_root",model_name,
               "--limit","100"]
    
    main(arg_list)

