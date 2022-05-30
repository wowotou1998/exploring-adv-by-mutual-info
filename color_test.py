import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

# cmap1 = plt.get_cmap('tab20b')
# cmap2 = plt.get_cmap('tab20c')
cmap3 = plt.get_cmap('Set3')
# new_cmap = ListedColormap(cmap1.colors + cmap2.colors)
# sm = plt.cm.ScalarMappable(cmap='Set3', norm=plt.Normalize(vmin=0, vmax=11))
sm = plt.cm.ScalarMappable(cmap='Greens', norm=plt.Normalize(vmin=0, vmax=11))
# sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=0, vmax=11))
x = np.random.random([40, 2])
cluster_labels = np.arange(12)
fig0, ax0 = plt.subplots(figsize=[12, 8])
plt.figure(figsize=[12, 8])
plt.scatter(x[:12, 0], x[:12, 1], s=500, c=cluster_labels, cmap=cmap3, marker='o', linewidths=2)
# for i in range(12):
#     c = sm.to_rgba(i)
#     ax0.scatter(x[i, 0], x[i, 1], s=1000, color=c)
# fig0.colorbar(sm, ax=ax0)
# plt.show()
# cmap_names = sorted(m for m in plt.colormaps if not m.endswith("_r"))
# print(cmap_names)
