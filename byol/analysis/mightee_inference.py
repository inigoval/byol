from typing import Optional
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch

from astropy.io import fits
from astropy.wcs import WCS
from typing import Any, Callable, Optional
from astropy.nddata import Cutout2D
from pathlib import Path
from cata2data import CataData
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import Counter
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image

from byol.paths import Path_Handler
from byol.finetuning import FineTune
from byol.utilities import compute_mu_sig_images


def array_to_png(img):
    im = Image.fromarray(img)
    im = im.convert("L")
    return im


def rescale_image(img, low):
    img_max = np.max(img)
    img_min = low
    # img -= img_min
    img /= max(1e-7, img_max - img_min)  # clamp divisor so it can't be zero
    # img /= img_max - img_min
    img *= 255.0
    return img


def image_preprocessing(image: np.ndarray, field: str) -> np.ndarray:
    """Example preprocessing function for basic images.
    Args:
        image (np.ndarray): image
        field (str): Not Implemented here, but will be passed.
    Returns:
        np.ndarray: Squeezed image. I.e. removed empty axis.
    """
    return np.squeeze(image)


def wcs_preprocessing(wcs, field: str):
    """Example preprocessing function for wcs (world coordinate system).
    Args:
        wcs: Input wcs.
        field (str): field name matching the respective wcs.
    Returns:
        Altered wcs.
    """
    if field in ["COSMOS"]:
        return (wcs.dropaxis(3).dropaxis(2),)
    elif field in ["XMMLSS"]:
        raise UserWarning(
            f"This may cause issues in the future. It is unclear where header would have been defined."
        )
        wcs = WCS(header, naxis=2)  # This surely causes a bug right?
    else:
        return wcs


def catalogue_preprocessing(df: pd.DataFrame, random_state: Optional[int] = None) -> pd.DataFrame:
    """Example Function to make preselections on the catalog to specific
    sources meeting given criteria.
    Args:
        df (pd.DataFrame): Data frame containing catalogue information.
        random_state (Optional[int], optional): Random state seed. Defaults to None.
    Returns:
        pd.DataFrame: Subset catalogue.
    """
    # Only consider resolved sources
    df = df.loc[df["RESOLVED"] == 1]

    # Sort by S_INT (integrated flux)
    df = df.sort_values("S_INT", ascending=False)

    # Only consider unique islands of sources
    # df = df.groupby("ISL_ID").first()
    df = df.drop_duplicates(subset=["ISL_ID"], keep="first")

    # Drop values below integrated flux
    # df.drop(df[df["S_INT"] < 0.0003].index, inplace=True)
    # df.drop(df[df["S_INT"] < 0.0002].index, inplace=True)

    # Drop values below SNR
    df["SNR"] = df.apply(lambda row: row.S_PEAK / row.ISL_RMS, axis=1)
    df.drop(df[df["SNR"] < 20].index, inplace=True)

    # Sort by field
    # df = df.sort_values("field")

    return df.reset_index(drop=True)


class MighteeCataData:
    def __init__(self, catadata, transform):
        self.catadata = catadata
        self.transform = transform

    def __getitem__(self, index: int) -> tuple:
        # rms = self.catadata.df.loc[index, "ISL_RMS"]

        img = self.catadata[index]

        _, _, rms = sigma_clipped_stats(img)

        # Remove NaNs
        img = np.nan_to_num(img, nan=0.0)

        # Clip values below 3 sigma
        img[np.where(img <= 3 * rms)] = 0.0

        img = rescale_image(img, 3 * rms)

        img = array_to_png(np.squeeze(img))

        img = self.transform(np.squeeze(img))

        name = self.catadata.df.loc[index, "NAME"]

        return img, name

    def __len__(self):
        return self.catadata.__len__()


def _mightee_dataset(data_dir: Path, transform):
    catalogue_paths = [
        data_dir / "COSMOS_source_catalogue.fits",
        data_dir / "XMMLSS_source_catalogue.fits",
    ]
    image_paths = [
        data_dir / "COSMOS_image.fits",
        data_dir / "XMMLSS_image.fits",
    ]

    field_names = ["COSMOS", "XMMLSS"]

    # Create Data Set #
    mightee_data = CataData(
        catalogue_paths=catalogue_paths,
        image_paths=image_paths,
        field_names=field_names,
        cutout_width=114,
        # cutout_width=120,
        catalogue_preprocessing=catalogue_preprocessing,
        image_preprocessing=image_preprocessing,
    )

    return MighteeCataData(mightee_data, transform)


def aggregate(preds: list) -> tuple:
    counter = Counter(preds)
    agg_pred, count = counter.most_common(1)[0]
    vote_frac = count / len(preds)

    return agg_pred, vote_frac


if __name__ == "__main__":
    path_handler = Path_Handler()
    paths = path_handler._dict()

    # # Load transform
    # transform = T.Compose(
    #     [
    #         T.ToTensor(),
    #         T.CenterCrop(70),
    #     ]
    # )
    # mightee_data = _mightee_dataset(paths["data"] / "MIGHTEE_continuum_early_release", transform)

    # mu, sig = compute_mu_sig_images(mightee_data, 256)

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize(70),  # Rescale to adjust for resolution difference between MIGHTEE & RGZ
            # T.CenterCrop(70),
            # T.Normalize(1.59965605788234e-05, 0.0038063037602458706),
            T.Normalize(0.008008896, 0.05303395),
        ]
    )

    mightee_data = _mightee_dataset(paths["data"] / "MIGHTEE", transform)

    # print("Making predictions...")
    model_dir = paths["main"] / "analysis" / "weights"
    # Make predictions on RGZ data
    y = {name: {"pred": [], "softmax": []} for name in mightee_data.catadata.df["NAME"]}
    for i, model_path in enumerate(model_dir.iterdir()):
        print(f"Loading model {i}...")
        model = FineTune.load_from_checkpoint(model_path)
        model.eval()

        # d_rgz = rgz_cut(d_rgz, 20, mb_cut=True)

        print("Performing inference on mightee data...")
        for x, name in tqdm(DataLoader(mightee_data, batch_size=1, shuffle=False)):
            # x = transform(x)
            x = x.float()
            logits = model(x)
            # print(logits)
            preds = logits.softmax(dim=-1)
            y_softmax, y_pred = torch.max(preds, dim=1)
            y_pred += 1

            y[name[0]]["pred"].append(y_pred.item())
            y[name[0]]["softmax"].append(y_softmax.item())

    print("Aggregating predictions...")

    df_vals = []
    for name, value in tqdm(y.items()):
        preds, softmax = value["pred"], value["softmax"]
        # print(preds, softmax)
        agg_pred, vote_frac = aggregate(preds)

        df_vals.append([name, agg_pred, vote_frac])

    df = pd.DataFrame(df_vals, columns=["name", "pred", "vote_frac"])

    for i, (X, names) in tqdm(
        enumerate(DataLoader(mightee_data, batch_size=16, shuffle=False)), total=len(mightee_data)
    ):
        fig = plt.figure(figsize=(13.0, 13.0))
        grid = ImageGrid(fig, 111, nrows_ncols=(4, 4), axes_pad=0.1)

        for name, ax, im in zip(names, grid, list(X)):
            im = torch.squeeze(im)
            ax.axis("off")
            ax.imshow(im, cmap="hot")

            # Add text
            pred = df.loc[df["name"] == name, "pred"].values.item()
            vote_frac = df.loc[df["name"] == name, "vote_frac"].values.item()

            text = f"FR{pred}, C={vote_frac}"
            ax.text(1, 66, text, fontsize=23, color="yellow")

        plt.axis("off")
        plt.savefig(
            paths["main"] / "analysis" / "imgs" / "mightee" / f"grid_{i:03d}.png", bbox_inches="tight"
        )
        plt.close(fig)

    # Drop unecessary columns
    df.to_csv(paths["data"] / "MIGHTEE" / "mightee_data_preds.csv", index=False)
    print(df.head())
    print("Done!")
