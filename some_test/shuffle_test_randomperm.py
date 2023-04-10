import torch

a = torch.rand(3, 5)
print(a)

a = a[torch.randperm(a.size()[0])]
print(a)

a = a[torch.randperm(a.size()[0])]
print(a)

a = a[torch.randperm(a.size()[0])]
print(a)

a = a[torch.randperm(a.size()[0])]
print(a)
