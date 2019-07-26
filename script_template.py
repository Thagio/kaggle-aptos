import gzip
import base64
import os
from pathlib import Path
from typing import Dict


# this is base64 encoded source code
file_data: Dict = {file_data}


for path, encoded in file_data.items():
    print(path)
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(gzip.decompress(base64.b64decode(encoded)))


def run(command):
    os.system('export PYTHONPATH=${PYTHONPATH}:/kaggle/working && ' + command)

FOLD = 0
N_EPOCH = 25
LIMIT = True

run('python setup.py develop --install-dir /kaggle/working')
run('python -m apotos.make_folds')

if LIMIT:
    run('python -m apotos.main --mode train --run_root model_1 --n-epochs {epoch} --fold {fold} --limit 100'.format(epoch=N_EPOCH,fold=FOLD))
    run('python -m apotos.main --mode predict_valid --run_root model_1 --fold {fold} --limit 100'.format(fold=FOLD))
    run('python -m apotos.main --mode predict_test --run_root model_1 --limit 100')
else:
    run('python -m apotos.main --mode train --run_root model_1 --n-epochs {epoch} --fold {fold}'.format(epoch=N_EPOCH,fold=FOLD))
    run('python -m apotos.main --mode predict_valid --run_root model_1 --fold {fold}'.format(fold=FOLD))
    run('python -m apotos.main --mode predict_test --run_root model_1')

run('python -m apotos.make_submission --predictions model_1/test.h5 --output submission.csv')
