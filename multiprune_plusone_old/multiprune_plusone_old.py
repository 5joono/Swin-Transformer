import os
import numpy as np
import pandas as pd
os.environ['MKL_THREADING_LAYER'] = 'GNU'
# arr1 = [2,2,6,2]
# arr2 = [3,6,12,24]
df = pd.DataFrame(columns=['multiprune', 'pluslayer', 'plushead', 'acc1'])
df.to_csv("multiprune_plusone.csv",index=False)
headlist = [[7,10,5,8,6,9,3,4,1,11,2,0],
            [11,4,3,6,0,1,2,7,10,9,8,5],
            [0,5,9,1,7,10,3,4,8,6,2,11],
            [7,8,4,2,1,11,5,3,6,0,10,9],
            [9,6,2,3,7,4,10,0,8,1,11,5],
            [9,11,1,3,6,7,5,8,10,2,0,4]]
for multiprune in range(1,13):
    for pluslayer in range(6):
        for plushead in headlist[pluslayer][multiprune:]:
            os.system(f'python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval --cfg configs/swin_tiny_patch4_window7_224.yaml --resume swin_tiny_patch4_window7_224.pth --data-path data/imagenet/ --prune {multiprune}_{pluslayer}_{plushead}')