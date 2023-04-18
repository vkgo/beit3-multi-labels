print(a)
import torch
path=r'./saved_model/checkpoint-f10.721_pre80.902_rec65.088_acc96.939_threshold0.900/mp_rank_00_model_states.pt'
print('loading')
model=torch.load(path)
print('loaded')
model.eval()

a=torch.rand((1,3,224,224))
b=model(a)
print(b)
print(b.size())