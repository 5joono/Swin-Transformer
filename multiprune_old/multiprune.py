import os
import numpy as np
import pandas as pd
os.environ['MKL_THREADING_LAYER'] = 'GNU'
# arr1 = [2,2,6,2]
# arr2 = [3,6,12,24]
df = pd.DataFrame(columns=['multiprune', 'acc1'])
df.to_csv("multiprune.csv",index=False)
for multiprune in range(1,13):
    os.system(f'python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval --cfg configs/swin_tiny_patch4_window7_224.yaml --resume swin_tiny_patch4_window7_224.pth --data-path data/imagenet/ --prune {multiprune}')