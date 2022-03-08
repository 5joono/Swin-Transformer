import os
import numpy as np
import pandas as pd
os.environ['MKL_THREADING_LAYER'] = 'GNU'

# df = pd.DataFrame(columns=['multiprune', 'headstr', 'pluslayer', 'plushead', 'acc1'])
# df.to_csv("multiprune_plusone.csv",index=False)

prevheadlist = [set([7]),set([11]),set([0]),set([7]),set([9]),set([9])]
plusheadlist = [set(range(12))-{7},set(range(12))-{11},set(range(12))-{0},set(range(12))-{7},set(range(12))-{9},set(range(12))-{9}]

for multiprune in range(1,12):
    
    headstr = []
    for oneset in prevheadlist:
        setstr = [str(int(s)) for s in oneset]
        setstr = '+'.join(setstr)
        headstr.append(setstr)
    headstr = '.'.join(headstr)
        
    for pluslayer in range(6):
        for plushead in plusheadlist[pluslayer]:
            os.system(f'python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval --cfg configs/swin_tiny_patch4_window7_224.yaml --resume swin_tiny_patch4_window7_224.pth --data-path data/imagenet/ --prune {multiprune}_{headstr}_{pluslayer}_{plushead}')
        
        df = pd.read_csv("multiprune_plusone.csv")
        df = df[(df.multiprune == multiprune) & (df.pluslayer == pluslayer)]
        df = df.apply(pd.to_numeric, errors = 'coerce')
        max_acc1_idx = df.idxmax().acc1
        plusheadlist[pluslayer].remove(df.loc[max_acc1_idx].plushead)
        prevheadlist[pluslayer].add(df.loc[max_acc1_idx].plushead)
