import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm

from byol.finetuning_regression import FineTuneRegression
from byol.datamodules import RGZ_DataModule_Finetune_Regression
from byol.paths import Path_Handler
from byol.resnet import _get_resnet

# Make some plots
paths = Path_Handler()._dict()
ckpt = torch.load(paths["main"] / "regression.ckpt")
# weights = ckpt["state_dict"]

model = FineTuneRegression.load_from_checkpoint(paths["main"] / "regression.ckpt")
model.eval()
config = model.config

# model.load_state_dict(weights)
# model.eval()

finetune_datamodule = RGZ_DataModule_Finetune_Regression(
    paths["rgz"],
    batch_size=config["finetune"]["batch_size"],
    center_crop=config["augmentations"]["center_crop"],
    num_workers=config["dataloading"]["num_workers"],
    prefetch_factor=config["dataloading"]["prefetch_factor"],
    pin_memory=config["dataloading"]["pin_memory"],
)
finetune_datamodule.setup()


preds = []
targets = []
s_int = []

csv_path = paths["rgz"] / "rgz_data.csv"
df = pd.read_csv(csv_path)

for x, y in tqdm(finetune_datamodule.test_dataloader()[0]):
    x = x.to(model.device)
    targets.append(y["size"])
    preds.append(model(x).squeeze().detach().cpu().numpy())

    for id in y["id"]:
        s_int.append(df.loc[df["rgz_name"] == id, "radio.total_flux"].values[0])

preds = np.concatenate(preds)
targets = np.concatenate(targets)
s_int = np.array(s_int)


print(np.mean(np.abs(preds - targets)))

# Pyplot hparams
plt.rc("font", family="Liberation Mono")
plt.rcParams["font.family"] = "Liberation Mono"

alpha = 0.6
marker_size = 0.1
fig_size = (10 / 3, 3)
seed = 69
fontsize = 9
marker = "."

# Plot a scatter plot with preds on y axis and targets on x axis and s_int colorbar
fig, ax = plt.subplots()
fig.set_size_inches(fig_size)
scatter = ax.scatter(
    targets,
    preds,
    c=s_int,
    cmap="Spectral",
    s=marker_size,
    marker=marker,
    vmin=25,
    vmax=100,
    alpha=alpha,
)
plt.gca().set_aspect("equal", "datalim")
cbar = fig.colorbar(scatter)
cbar.ax.tick_params(labelsize=fontsize)

ax.set_xlabel("Targets", fontsize=fontsize)
ax.set_ylabel("Predictions", fontsize=fontsize)
ax.set_ylim(bottom=0)
fig.savefig("scatter.png", bbox_inches="tight", pad_inches=0.05, dpi=600)

##############################
# Plot residual scatter plot #
##############################
fig, ax = plt.subplots()
fig.set_size_inches(fig_size)
scatter = ax.scatter(
    targets,
    preds - targets,
    c=s_int,
    cmap="Spectral",
    s=marker_size,
    marker=marker,
    vmin=25,
    vmax=100,
    alpha=alpha,
)
plt.gca().set_aspect("equal", "datalim")
cbar = fig.colorbar(scatter)
cbar.ax.tick_params(labelsize=fontsize)

ax.set_xlabel("Targets", fontsize=fontsize)
ax.set_ylabel("Predictions", fontsize=fontsize)
fig.savefig("residual_scatter.png", bbox_inches="tight", pad_inches=0.05, dpi=600)


##############################
# Plot histograms of small/medium/large source predictions #
##############################
# Get ranges for quartiles of targets
q0 = np.min(targets, 0)
q1 = np.quantile(targets, 0.25)
q2 = np.quantile(targets, 0.5)
q3 = np.quantile(targets, 0.75)
q4 = np.max(targets, 0)

quartiles = [(q0, q1), (q1, q2), (q2, q3), (q3, q4)]

# Plot histogram of errors for each quartile
fig, ax = plt.subplots(2, 2, sharey=True)
fig.set_size_inches(fig_size)
for i, n, q in zip([(0, 0), (1, 0), (0, 1), (1, 1)], range(4), quartiles):
    idx = np.where((targets >= q[0]) & (targets < q[1]))[0]
    ax[i].hist(np.abs(preds[idx] - targets[idx]), bins=20)
    # ax[i].set_title(f"Sizes in range {q[0]:.2f} to {q[1]:.2f}")
    ax[i].set_title(f"Q{n}")
    # ax[i].set_xlabel("Error", fontsize=fontsize)
    # ax[i].set_ylabel("Frequency", fontsize=fontsize)

# add a big axis, hide frame
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor="none", which="both", top=False, bottom=False, left=False, right=False)
plt.xlabel("error", fontsize=fontsize)
plt.ylabel("frequency", fontsize=fontsize)

fig.savefig("residual_histograms.png", bbox_inches="tight", pad_inches=0.05, dpi=600)
