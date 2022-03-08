import os
import numpy as np
import pandas as pd
os.environ['MKL_THREADING_LAYER'] = 'GNU'

df = pd.DataFrame(columns=['multiprune', 'headstr', 'pluslayer', 'plushead', 'acc1'])
df.to_csv("multiprune.csv",index=False)

os.system(f'python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval --cfg configs/swin_tiny_patch4_window7_224.yaml --resume swin_tiny_patch4_window7_224.pth --data-path data/imagenet/ --prune 1_7.11.0.7.9.9_999_999')
os.system(f'python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval --cfg configs/swin_tiny_patch4_window7_224.yaml --resume swin_tiny_patch4_window7_224.pth --data-path data/imagenet/ --prune 2_1+7.11+4.0+3.8+7.9+6.9+11_999_999')
os.system(f'python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval --cfg configs/swin_tiny_patch4_window7_224.yaml --resume swin_tiny_patch4_window7_224.pth --data-path data/imagenet/ --prune 3_8+1+7.1+11+4.0+3+5.8+2+7.9+2+6.9+11+6_999_999')
os.system(f'python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval --cfg configs/swin_tiny_patch4_window7_224.yaml --resume swin_tiny_patch4_window7_224.pth --data-path data/imagenet/ --prune 4_8+1+4+7.3+1+11+4.0+10+3+5.8+2+5+7.9+2+11+6.3+9+11+6_999_999')
os.system(f'python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval --cfg configs/swin_tiny_patch4_window7_224.yaml --resume swin_tiny_patch4_window7_224.pth --data-path data/imagenet/ --prune 5_1+4+6+7+8.1+3+4+6+11.0+3+5+8+10.1+2+5+7+8.2+6+7+9+11.3+6+7+9+11_999_999')
os.system(f'python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval --cfg configs/swin_tiny_patch4_window7_224.yaml --resume swin_tiny_patch4_window7_224.pth --data-path data/imagenet/ --prune 6_1+4+6+7+8+10.0+1+3+4+6+11.0+3+5+7+8+10.1+2+4+5+7+8.2+3+6+7+9+11.3+6+7+9+10+11_999_999')
os.system(f'python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval --cfg configs/swin_tiny_patch4_window7_224.yaml --resume swin_tiny_patch4_window7_224.pth --data-path data/imagenet/ --prune 7_1+4+5+6+7+8+10.0+1+2+3+4+6+11.0+3+4+5+7+8+10.1+2+3+4+5+7+8.2+3+4+6+7+9+11.1+3+6+7+9+10+11_999_999')
os.system(f'python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval --cfg configs/swin_tiny_patch4_window7_224.yaml --resume swin_tiny_patch4_window7_224.pth --data-path data/imagenet/ --prune 8_1+3+4+5+6+7+8+10.0+1+2+3+4+6+8+11.0+1+3+4+5+7+8+10.1+2+3+4+5+7+8+11.2+3+4+6+7+8+9+11.1+3+5+6+7+9+10+11_999_999')
os.system(f'python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval --cfg configs/swin_tiny_patch4_window7_224.yaml --resume swin_tiny_patch4_window7_224.pth --data-path data/imagenet/ --prune 9_1+3+4+5+6+7+8+10+11.0+1+2+3+4+6+8+9+11.0+1+3+4+5+7+8+9+10.1+2+3+4+5+6+7+8+11.0+2+3+4+6+7+8+9+11.1+3+5+6+7+8+9+10+11_999_999')
os.system(f'python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval --cfg configs/swin_tiny_patch4_window7_224.yaml --resume swin_tiny_patch4_window7_224.pth --data-path data/imagenet/ --prune 10_1+3+4+5+6+7+8+9+10+11.0+1+2+3+4+6+8+9+10+11.0+1+3+4+5+6+7+8+9+10.0+1+2+3+4+5+6+7+8+11.0+1+2+3+4+6+7+8+9+11.0+1+3+5+6+7+8+9+10+11_999_999')
os.system(f'python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval --cfg configs/swin_tiny_patch4_window7_224.yaml --resume swin_tiny_patch4_window7_224.pth --data-path data/imagenet/ --prune 11_1+2+3+4+5+6+7+8+9+10+11.0+1+2+3+4+6+7+8+9+10+11.0+1+2+3+4+5+6+7+8+9+10.0+1+2+3+4+5+6+7+8+9+11.0+1+2+3+4+6+7+8+9+10+11.0+1+2+3+5+6+7+8+9+10+11_999_999')

