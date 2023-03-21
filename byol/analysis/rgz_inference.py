import pandas as pd
import torch
import torchvision.transforms as T
import logging

from torch.utils.data import DataLoader
from einops import rearrange
from finetune.main import FineTune
from tqdm import tqdm
from collections import Counter

from byol.paths import Path_Handler
from byol.dataloading.datamodules.rgz import RGZ108k

# Get paths
print("Getting paths...")
path_handler = Path_Handler()
paths = path_handler._dict()

# Load in RGZ csv
print("Loading in RGZ data...")
csv_path = paths["rgz"] / "rgz_data.csv"
df_rgz = pd.read_csv(csv_path)

# Add fr_prediction/confidence columns
df_rgz.insert(len(df_rgz.columns), "fr_prediction", -1)
df_rgz.insert(len(df_rgz.columns), "fr_vote_fraction", -1)

# Load transform
transform = T.Compose(
    [
        T.CenterCrop(70),
        T.ToTensor(),
        T.Normalize((0.008008896,), (0.05303395,)),
    ]
)
# Load in RGZ data
d_rgz = RGZ108k(paths["rgz"], train=True, transform=transform)

print("Making predictions...")
model_dir = paths["main"] / "analysis" / "weights"
# Make predictions on RGZ data
y = {id: {"pred": [], "softmax": []} for id in d_rgz.rgzid}
for i, model_path in enumerate(model_dir.iterdir()):
    print(f"Loading model {i}...")
    model = FineTune.load_from_checkpoint(model_path)
    model.eval()

    print("Performing inference on RGZ data...")
    for (x, _), id in tqdm(
        zip(DataLoader(d_rgz, batch_size=1, shuffle=False), d_rgz.rgzid), total=len(d_rgz)
    ):
        logits = model(x)
        preds = logits.softmax(dim=-1)
        y_softmax, y_pred = torch.max(preds, dim=1)
        y_pred += 1

        y[id]["pred"].append(y_pred.item())
        y[id]["softmax"].append(y_softmax.item())


print("Aggregating predictions...")


def aggregate(preds: list) -> tuple:
    counter = Counter(preds)
    print(counter)
    print(counter.most_common(1))
    agg_pred, count = counter.most_common(1)[0]
    vote_frac = count / len(preds)

    return agg_pred, vote_frac


for id, value in tqdm(y.items()):
    preds, softmax = value["pred"], value["softmax"]
    print(preds)
    agg_pred, vote_frac = aggregate(preds)

    df_rgz.loc[df_rgz["rgz_name"] == id, "fr_prediction"] = agg_pred
    df_rgz.loc[df_rgz["rgz_name"] == id, "fr_vote_fraction"] = vote_frac


df_rgz.to_csv(paths["rgz"] / "rgz_data_preds.csv", index=False)
print(df_rgz[["fr_prediction", "fr_vote_fraction"]].head())
print("Done!")
