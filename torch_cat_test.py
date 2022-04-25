import torch

x = torch.randn(2, 3)

y = torch.cat((x, x, x), 0)  # 在 0 维(纵向)进行拼接
z = torch.cat((x, x))
print(y.size())
print(z.size())

dict = {'a': [15, ]}
for idx, (k, v) in enumerate(dict.items()):
    print(idx)
    print(k)
    print(v)


def f():
    x = 3
    print(x)
f()
y = x.detach()
print(y)