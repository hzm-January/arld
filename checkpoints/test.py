import torch
import torch.nn as nn
import torch.nn.functional as F

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)

empty_token_embedding = nn.Embedding(1, 10)
print(empty_token_embedding)
obj_feats = [empty_token_embedding(torch.LongTensor(0))]
print(obj_feats)
obj_feats2 = [empty_token_embedding(torch.LongTensor([0]))]
print(obj_feats2)
print(obj_feats2[0].shape)

n_modes = 100
embed_dim = 512
init_main_modes = torch.randn(n_modes, embed_dim)
# project main modes to the surface of a hyper-sphere.
init_main_modes = F.normalize(init_main_modes, p=2, dim=-1)
main_modes = nn.Parameter(init_main_modes, requires_grad=False)
mode_weights = torch.randn(1, n_modes)
mode_weights = mode_weights.softmax(dim=-1)
latent_z = torch.mm(mode_weights, main_modes)
latent_z = F.normalize(latent_z, p=2, dim=-1)
print(latent_z.shape)

a = [torch.randn((1, 5)) for _ in range(8)]
print(len(a))
print(a[0].shape)
b = torch.cat(a, dim=0)
c = torch.cat(a, dim=1)
print(b.shape)
print(c.shape)

input_emb = torch.randn(1, 1, 1024)

aaa = nn.Linear(in_features=1024, out_features=512)
ab = aaa(input_emb)
print(ab)