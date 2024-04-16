import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader
from torchvision.transforms.functional import center_crop

from byol.datasets import MiraBest_F, RGZ108k
from byol.paths import Path_Handler, create_path

paths = Path_Handler()._dict()

# rgz = RGZ108k(
#     root=paths["rgz"],
#     transform=T.ToTensor(),
#     remove_duplicates=False,
#     crop_resize=True,
#     cut_threshold=20,
# )

# X, _ = next(iter(DataLoader(rgz, 25, shuffle=False)))

# # Plot the first 25 images in the dataset as a 5x5 grid
# # Hide axes in each subplot
# fig, ax = plt.subplots(5, 5, figsize=(10, 10))
# for i in range(5):
#     for j in range(5):
#         ax[i, j].imshow(X[i * 5 + j].squeeze(), cmap="hot")
#         ax[i, j].axis("off")
# plt.tight_layout()
# plt.savefig("test.png")

mb = MiraBest_F(root=paths["mb"], transform=T.ToTensor())

imgs = []
for img, _ in DataLoader(mb, 1):
    imgs.append(center_crop(img, 70).squeeze())
    img = img.numpy()
    if np.max(img) > 0.98:
        img[img > 0.98] = 1.2


fig, ax = plt.subplots(5, 5, figsize=(10, 10))
for i in range(5):
    for j in range(5):
        ax[i, j].imshow(imgs[i * 5 + j], cmap="hot")
        ax[i, j].axis("off")
plt.tight_layout()
plt.savefig("test.png")
