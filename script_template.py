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


run('python setup.py develop --install-dir /kaggle/working')
run('python -m apotos.make_folds')
run('python -m apotos.main --mode train --run_root model_1 --n-epochs 25')
run('python -m apotos.main --mode predict_test --run_root model_1')
run('python -m apotos.make_submission --predictions model_1/test.h5 --output submission.csv')
