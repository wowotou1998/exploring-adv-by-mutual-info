import torch

from ModelSet import *

model = WideResNet(depth=1 * 6 + 4, num_classes=10, widen_factor=10, dropRate=0.0)
Model_Name = 'resnet18'
checkpoint = torch.load('./Checkpoint/%s.pth' % Model_Name)
model.load_state_dict(checkpoint['model'])
# data moved to GPU
model = model.to(device)
# 必须先将模型进行迁移,才能再装载optimizer,不然会出现数据在不同设备上的错误
# optimizer.load_state_dict(checkpoint['optimizer'])
best_test_acc = checkpoint['epoch_test_acc']
print('--> Load checkpoint successfully!')
