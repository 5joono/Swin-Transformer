import torch
for i in range(12):
    for j in range(i+1,12):
        count = 0
        for idx in range(361):
            attn_map = torch.load(f'attn_map/{idx}.pth', map_location = torch.device('cuda'))
            for batch in range(attn_map[0].shape[0]):
                for token in range(49):
                    if(torch.dot(attn_map[0][batch,i,:,token], attn_map[0][batch,j,:,token])/torch.norm(attn_map[0][batch,i,:,token])/torch.norm(attn_map[0][batch,j,:,token]) > 0.7):
                        count = count + 1
        similarity = count / 50000 / 49 / 4
        print(i, j, count, similarity)