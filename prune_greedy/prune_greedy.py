import os
import numpy as np
import pandas as pd
os.environ['MKL_THREADING_LAYER'] = 'GNU'
# arr1 = [2,2,6,2]
# arr2 = [3,6,12,24]
df = pd.DataFrame(columns=['stage', 'layer', 'onehead', 'otherheads', 'headnum', 'acc1'])
df.to_csv("prune_greedy.csv",index=False)
for layer in range(6):
    headlist = set(range(12))
    for iter in range(12):
        for onehead in headlist:
            otherheads=set(range(12))-headlist
            list_of_strings = [str(s) for s in otherheads]
            otherheads="+".join(list_of_strings)
            os.system(f'python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval --cfg configs/swin_tiny_patch4_window7_224.yaml --resume swin_tiny_patch4_window7_224.pth --data-path data/imagenet/ --prune 2_{layer}_{onehead}_{otherheads}')
        df = pd.read_csv("prune_greedy.csv")
        df_iter = df[(df.headnum == iter+1) & (df.layer == layer)]
        df_iter = df_iter.apply(pd.to_numeric, errors = 'coerce')
        max_acc1_idx = df_iter.idxmax().acc1
        headlist.remove(df_iter.loc[max_acc1_idx].onehead)