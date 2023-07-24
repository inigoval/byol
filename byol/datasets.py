import torch.utils.data as D
import torch.nn as nn
import torch
import numpy as np
import os
import sys
import torchvision.transforms as T
import pytorch_lightning as pl
import torch.utils.data as data

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity
from torch.utils.data import DataLoader
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from typing import Any, Dict, List, Tuple, Type, Optional
from torch.utils.data import Subset
from einops import rearrange
from torchvision.transforms.functional import center_crop, resize

from byol.utilities import rgz_cut
from byol.paths import Path_Handler


class MiraBest_F(data.Dataset):
    """
    Inspired by `HTRU1 <https://as595.github.io/HTRU1/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``MiraBest-F.py` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        test_size (float, optional): Fraction of data to be stratified into a test set. i.e. 0.2
            stratifies 20% of the MiraBest into a test set. Default (None) returns the
            standard MiraBest data set.
    """

    base_folder = "F_batches"
    url = "http://www.jb.man.ac.uk/research/MiraBest/MiraBest_F/MiraBest_F_batches.tar.gz"
    filename = "MiraBest_F_batches.tar.gz"
    tgz_md5 = "7d4e3a623d29db7204bce81676ee8ce2"
    train_list = [
        ["data_batch_1", "f7a470b7367e8e0d0c5093d2cf266d54"],
        ["data_batch_2", "bb65ecd7e748e9fb789419b1efbf1bab"],
        ["data_batch_3", "32de1078e7cd47f5338c666a1b563ede"],
        ["data_batch_4", "a1209aceedd8806c88eab27ce45ee2c4"],
        ["data_batch_5", "1619cd7c54f5d71fcf4cfefea829728e"],
        ["data_batch_6", "636c2b84649286e19bcb0684fc9fbb01"],
        ["data_batch_7", "bc67bc37080dc4df880ffe9720d680a8"],
    ]

    test_list = [
        ["test_batch", "ac7ea0d5ee8c7ab49f257c9964796953"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "e1b5450577209e583bc43fbf8e851965",
    }

    def __init__(
        self,
        root,
        train: Optional[bool] = True,
        transform=None,
        target_transform=None,
        download=False,
        test_size=None,
        aug_type="torchvision",
        data_type="double",
    ):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.aug_type = aug_type

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted." + " You can use download=True to download it"
            )

        if self.train and test_size is None:
            downloaded_list = self.train_list
        elif not self.train and test_size is None:
            downloaded_list = self.test_list
        else:
            downloaded_list = self.train_list + self.test_list

        self.data = []
        self.targets = []
        self.filenames = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)

            with open(file_path, "rb") as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding="latin1")

                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                    self.filenames.extend(entry["filenames"])
                else:
                    self.targets.extend(entry["fine_labels"])
                    self.filenames.extend(entry["filenames"])

        # Extract metadata
        self.las = [float(filename[-11:-4]) for filename in self.filenames]
        self.ra = [float(filename[-26:-19]) for filename in self.filenames]
        self.dec = [float(filename[-34:-27]) for filename in self.filenames]

        self.data = np.vstack(self.data).reshape(-1, 1, 150, 150)
        self.data = self.data.transpose((0, 2, 3, 1))
        self.full_targets = self.targets

        # Stratify entire data set according to input ratio (seeded)
        if test_size is not None:
            data_train, data_test, targets_train, targets_test = train_test_split(
                self.data,
                self.targets,
                test_size=test_size,
                stratify=self.targets,  # Targets to stratify according to
                random_state=42,
            )
            if self.train:
                self.data = data_train
                self.targets = targets_train
                self.full_targets = targets_train
            else:
                self.data = data_test
                self.targets = targets_test
                self.full_targets = targets_test

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError(
                "Dataset metadata file not found or corrupted."
                + " You can use download=True to download it"
            )
        with open(path, "rb") as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = np.reshape(img, (150, 150))

        if self.aug_type == "albumentations":
            if self.transform is not None:
                img = self.transform(image=img)["image"]

            if self.target_transform is not None:
                target = self.target_transform(image=target)["image"]

        elif self.aug_type == "torchvision":
            img = Image.fromarray(img, mode="L")
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            raise NotImplementedError(
                f"{self.aug_type} not implemented. Currently 'aug_type' must be either 'albumentations' which defaults to Albumentations or 'torchvision' to be functional."
            )

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            # print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        tmp = "train" if self.train is True else "test"
        fmt_str += "    Split: {}\n".format(tmp)
        fmt_str += "    Root Location: {}\n".format(self.root)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp)))
        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(
            tmp, self.target_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str


class MBFRFull(MiraBest_F):

    """
    Child class to load all FRI (0) & FRII (1)
    [100, 102, 104, 110, 112] and [200, 201, 210]
    """

    def __init__(self, *args, **kwargs):
        super(MBFRFull, self).__init__(*args, **kwargs)

        fr1_list = [0, 1, 2, 3, 4]
        fr2_list = [5, 6, 7]
        exclude_list = [8, 9]

        if exclude_list == []:
            return
        if self.train:
            targets = np.array(self.targets)

            exclude = np.array(exclude_list).reshape(1, -1)

            # Create a mask, with False where we have excluded labels
            exclude_mask = ~(targets.reshape(-1, 1) == exclude).any(axis=1)

            # Create a mask with True where we want to change the label to fri/frii
            fr1 = np.array(fr1_list).reshape(1, -1)
            fr2 = np.array(fr2_list).reshape(1, -1)
            fr1_mask = (targets.reshape(-1, 1) == fr1).any(axis=1)
            fr2_mask = (targets.reshape(-1, 1) == fr2).any(axis=1)

            # Set labels to fri/frii
            targets[fr1_mask] = 0  # set all FRI to Class~0
            targets[fr2_mask] = 1  # set all FRII to Class~1

            # Remove excluded labels
            self.data = self.data[exclude_mask]
            self.targets = targets[exclude_mask].tolist()
            self.full_targets = np.array(self.full_targets)[exclude_mask].tolist()
        else:
            targets = np.array(self.targets)
            exclude = np.array(exclude_list).reshape(1, -1)
            exclude_mask = ~(targets.reshape(-1, 1) == exclude).any(axis=1)
            fr1 = np.array(fr1_list).reshape(1, -1)
            fr2 = np.array(fr2_list).reshape(1, -1)
            fr1_mask = (targets.reshape(-1, 1) == fr1).any(axis=1)
            fr2_mask = (targets.reshape(-1, 1) == fr2).any(axis=1)

            targets[fr1_mask] = 0  # set all FRI to Class~0
            targets[fr2_mask] = 1  # set all FRII to Class~1
            self.data = self.data[exclude_mask]
            self.targets = targets[exclude_mask].tolist()
            self.full_targets = np.array(self.full_targets)[exclude_mask].tolist()


class MBFRConfident(MiraBest_F):

    """
    Child class to load only confident FRI (0) & FRII (1)
    [100, 102, 104] and [200, 201]
    """

    def __init__(self, *args, **kwargs):
        super(MBFRConfident, self).__init__(*args, **kwargs)

        fr1_list = [0, 1, 2]
        fr2_list = [5, 6]
        exclude_list = [3, 4, 7, 8, 9]

        if exclude_list == []:
            return
        if self.train:
            targets = np.array(self.targets)
            exclude = np.array(exclude_list).reshape(1, -1)
            exclude_mask = ~(targets.reshape(-1, 1) == exclude).any(axis=1)
            fr1 = np.array(fr1_list).reshape(1, -1)
            fr2 = np.array(fr2_list).reshape(1, -1)
            fr1_mask = (targets.reshape(-1, 1) == fr1).any(axis=1)
            fr2_mask = (targets.reshape(-1, 1) == fr2).any(axis=1)
            targets[fr1_mask] = 0  # set all FRI to Class~0
            targets[fr2_mask] = 1  # set all FRII to Class~1
            self.data = self.data[exclude_mask]
            self.targets = targets[exclude_mask].tolist()
            self.full_targets = np.array(self.full_targets)[exclude_mask].tolist()
        else:
            targets = np.array(self.targets)
            exclude = np.array(exclude_list).reshape(1, -1)
            exclude_mask = ~(targets.reshape(-1, 1) == exclude).any(axis=1)
            fr1 = np.array(fr1_list).reshape(1, -1)
            fr2 = np.array(fr2_list).reshape(1, -1)
            fr1_mask = (targets.reshape(-1, 1) == fr1).any(axis=1)
            fr2_mask = (targets.reshape(-1, 1) == fr2).any(axis=1)
            targets[fr1_mask] = 0  # set all FRI to Class~0
            targets[fr2_mask] = 1  # set all FRII to Class~1
            self.data = self.data[exclude_mask]
            self.targets = targets[exclude_mask].tolist()
            self.full_targets = np.array(self.full_targets)[exclude_mask].tolist()


class MBFRUncertain(MiraBest_F):

    """
    Child class to load only uncertain FRI (0) & FRII (1)
    [110, 112] and [210]
    """

    def __init__(self, *args, **kwargs):
        super(MBFRUncertain, self).__init__(*args, **kwargs)

        fr1_list = [3, 4]
        fr2_list = [7]
        exclude_list = [0, 1, 2, 5, 6, 8, 9]

        if exclude_list == []:
            return
        if self.train:
            targets = np.array(self.targets)
            exclude = np.array(exclude_list).reshape(1, -1)
            exclude_mask = ~(targets.reshape(-1, 1) == exclude).any(axis=1)
            fr1 = np.array(fr1_list).reshape(1, -1)
            fr2 = np.array(fr2_list).reshape(1, -1)
            fr1_mask = (targets.reshape(-1, 1) == fr1).any(axis=1)
            fr2_mask = (targets.reshape(-1, 1) == fr2).any(axis=1)
            targets[fr1_mask] = 0  # set all FRI to Class~0
            targets[fr2_mask] = 1  # set all FRII to Class~1
            self.data = self.data[exclude_mask]
            self.targets = targets[exclude_mask].tolist()
            self.full_targets = np.array(self.full_targets)[exclude_mask].tolist()
        else:
            targets = np.array(self.targets)
            exclude = np.array(exclude_list).reshape(1, -1)
            exclude_mask = ~(targets.reshape(-1, 1) == exclude).any(axis=1)
            fr1 = np.array(fr1_list).reshape(1, -1)
            fr2 = np.array(fr2_list).reshape(1, -1)
            fr1_mask = (targets.reshape(-1, 1) == fr1).any(axis=1)
            fr2_mask = (targets.reshape(-1, 1) == fr2).any(axis=1)
            targets[fr1_mask] = 0  # set all FRI to Class~0
            targets[fr2_mask] = 1  # set all FRII to Class~1
            self.data = self.data[exclude_mask]
            self.targets = targets[exclude_mask].tolist()
            self.full_targets = np.array(self.full_targets)[exclude_mask].tolist()


class MBHybrid(MiraBest_F):

    """
    Child class to load confident(0) and uncertain (1) hybrid sources
    [110, 112] and [210]
    """

    def __init__(self, *args, **kwargs):
        super(MBHybrid, self).__init__(*args, **kwargs)

        h1_list = [8]
        h2_list = [9]
        exclude_list = [0, 1, 2, 3, 4, 5, 6, 7]

        if exclude_list == []:
            return
        if self.train:
            targets = np.array(self.targets)
            exclude = np.array(exclude_list).reshape(1, -1)
            exclude_mask = ~(targets.reshape(-1, 1) == exclude).any(axis=1)
            fr1 = np.array(h1_list).reshape(1, -1)
            fr2 = np.array(h1_list).reshape(1, -1)
            fr1_mask = (targets.reshape(-1, 1) == fr1).any(axis=1)
            fr2_mask = (targets.reshape(-1, 1) == fr2).any(axis=1)
            targets[fr1_mask] = 0  # set all FRI to Class~0
            targets[fr2_mask] = 1  # set all FRII to Class~1
            self.data = self.data[exclude_mask]
            self.targets = targets[exclude_mask].tolist()
            self.full_targets = np.array(self.full_targets)[exclude_mask].tolist()
        else:
            targets = np.array(self.targets)
            exclude = np.array(exclude_list).reshape(1, -1)
            exclude_mask = ~(targets.reshape(-1, 1) == exclude).any(axis=1)
            fr1 = np.array(h1_list).reshape(1, -1)
            fr2 = np.array(h2_list).reshape(1, -1)
            fr1_mask = (targets.reshape(-1, 1) == fr1).any(axis=1)
            fr2_mask = (targets.reshape(-1, 1) == fr2).any(axis=1)
            targets[fr1_mask] = 0  # set all FRI to Class~0
            targets[fr2_mask] = 1  # set all FRII to Class~1
            self.data = self.data[exclude_mask]
            self.targets = targets[exclude_mask].tolist()
            self.full_targets = np.array(self.full_targets)[exclude_mask].tolist()


class MBRandom(MiraBest_F):

    """
    Child class to load 50 random FRI and 50 random FRII sources
    """

    def __init__(self, certainty="all", morphologies="all", *args, **kwargs):
        super(MBRandom, self).__init__(*args, **kwargs)

        # Checking flags
        # ------------------

        if certainty == "certain":
            certainty_list1 = np.array([0, 1, 2])
            certainty_list2 = np.array([5, 6])
        elif certainty == "uncertain":
            certainty_list1 = np.array([3, 4])
            certainty_list2 = np.array([7])
        else:
            certainty_list1 = np.array([0, 1, 2, 3, 4])
            certainty_list2 = np.array([5, 6, 7])

        if morphologies == "standard":
            morphology_list1 = np.array([0, 3])
            morphology_list2 = np.array([5, 7])
        else:
            morphology_list1 = np.array([0, 1, 2, 3, 4])
            morphology_list2 = np.array([5, 6, 7])

        list_matches1 = np.in1d(certainty_list1, morphology_list1)
        list_matches2 = np.in1d(certainty_list2, morphology_list2)

        h1_list = certainty_list1[np.where(list_matches1)[0]]
        h2_list = certainty_list2[np.where(list_matches2)[0]]

        # ------------------

        if self.train:
            targets = np.array(self.targets)
            h1 = np.array(h1_list).reshape(1, -1)
            h2 = np.array(h2_list).reshape(1, -1)
            h1_mask = (targets.reshape(-1, 1) == h1).any(axis=1)
            h2_mask = (targets.reshape(-1, 1) == h2).any(axis=1)
            h1_indices = np.where(h1_mask)
            h2_indices = np.where(h2_mask)
            h1_random = np.random.choice(h1_indices[0], 50, replace=False)
            h2_random = np.random.choice(h2_indices[0], 50, replace=False)
            targets[h1_random] = 0  # set all FRI to Class~0
            targets[h2_random] = 1  # set all FRII to Class~1
            target_list = np.concatenate((h1_random, h2_random))
            exclude_mask = (targets.reshape(-1, 1) == target_list).any(axis=1)
            self.data = self.data[exclude_mask]
            self.targets = targets[exclude_mask].tolist()
        else:
            targets = np.array(self.targets)
            h1 = np.array(h1_list).reshape(1, -1)
            h2 = np.array(h2_list).reshape(1, -1)
            h1_mask = (targets.reshape(-1, 1) == h1).any(axis=1)
            h2_mask = (targets.reshape(-1, 1) == h2).any(axis=1)
            h1_indices = np.where(h1_mask)
            h2_indices = np.where(h2_mask)
            h1_random = np.random.choice(h1_indices[0], 50, replace=False)
            h2_random = np.random.choice(h2_indices[0], 50, replace=False)
            targets[h1_random] = 0  # set all FRI to Class~0
            targets[h2_random] = 1  # set all FRII to Class~1
            target_list = np.concatenate((h1_random, h2_random))
            exclude_mask = (targets.reshape(-1, 1) == target_list).any(axis=1)
            self.data = self.data[exclude_mask]
            self.targets = targets[exclude_mask].tolist()


class RGZ108k(D.Dataset):
    """`RGZ 108k <>`_Dataset

    Args:
        root (string): Root directory of dataset where directory
            ``htru1-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = "rgz108k-batches-py"

    # Need to upload this, for now download is commented out
    # url = "http://www.jb.man.ac.uk/research/ascaife/rgz20k-batches-python.tar.gz"
    filename = "rgz108k-batches-python.tar.gz"
    tgz_md5 = "3fef587aa2aa3ece3b01b125977ae19d"
    train_list = [
        ["data_batch_1", "3f0c0eefdfafc0c5b373ad82c6cf9e38"],
        ["data_batch_2", "c39657f335fa8957e9e7cfe35b3503fe"],
        ["data_batch_3", "711cb401d9a039ad90ee61f652361a7e"],
        ["data_batch_4", "2d46b5031e9b8220886124876cb8b426"],
        ["data_batch_5", "8cb6644a59368abddc91af006fd67160"],
        ["data_batch_6", "b523102c655c44e575eb1ccae8af3a56"],
        ["data_batch_7", "58da5e781a566c331e105799d35b801c"],
        ["data_batch_8", "cdab6dd1245f9e91c2e5efb00212cd04"],
        ["data_batch_9", "20ef83c0c07c033c2a1b6b0e028a342d"],
        ["data_batch_10", "dd4e59f515b1309bbf80488f7352c2a6"],
        ["data_batch_11", "23d4e845d685c8183b4283277cc5ed72"],
        ["data_batch_12", "7905d89ba2ef1bc722e4d45357cc5562"],
        ["data_batch_13", "753ce85f565a72fa0c2aaa458a6ea5e0"],
        ["data_batch_14", "4145e21c48163d593eac403fdc259c5d"],
        ["data_batch_15", "713b1f15328e58c210a815affc6d4104"],
        ["data_batch_16", "bd45f4895bed648f20b2b2fa5d483281"],
        ["data_batch_17", "e8fe6c5f408280bd122b64eb1bbc9ad0"],
        ["data_batch_18", "1b35a3c4da301c7899356f890f8c08af"],
        ["data_batch_19", "357af43d0c18b448d38f37d1390d194e"],
        ["data_batch_20", "c908a88e9f62975fabf9e2241fe0a02b"],
        ["data_batch_21", "231b1413a2f0c8fda02c496c0b0d9ffb"],
        ["data_batch_22", "8f1b27f220f5253d18da1a4d7c46cc91"],
        ["data_batch_23", "6008ce450b4a4de0f81407da811e6fbf"],
        ["data_batch_24", "180c351fd32c3b204cac17e2fac7b98d"],
        ["data_batch_25", "51be04715b303da51cbe3640a164662b"],
        ["data_batch_26", "9cb972ae3069541dc4fa096ea95149eb"],
        ["data_batch_27", "065d888e4b131485f0a54089245849df"],
        ["data_batch_28", "d0430812428aefaabcec8c4cd8f0a838"],
        ["data_batch_29", "221bdd97fa36d0697deb13e4f708b74f"],
        ["data_batch_30", "81eaec70f17f7ff5f0c7f3fbc9d4060c"],
        ["data_batch_31", "f6ccddbf6122c0bac8befb7e7d5d386e"],
        ["data_batch_32", "e7cdf96948440478929bc0565d572610"],
        ["data_batch_33", "940d07f47d5d98f4a034d2dbc7937f59"],
        ["data_batch_34", "a5c97a274671c0536751e1041a05c0a9"],
        ["data_batch_35", "d4dbb71e9e92b61bfde9a2d31dfb6ec8"],
        ["data_batch_36", "208ef8426ce9079d65a215a9b89941bc"],
        ["data_batch_37", "60d0ca138812e1a8e2d439f5621fa7f6"],
        ["data_batch_38", "b17ff76a0457dc47e331668c34c0e7e6"],
        ["data_batch_39", "28712e629d7a7ceba527ba77184ee9c5"],
        ["data_batch_40", "a9b575bb7f108e63e4392f5dd1672d31"],
        ["data_batch_41", "3390460da44022c13d24f883556a18eb"],
        ["data_batch_42", "7297ca4b77c6059150f471969ca3827a"],
        ["data_batch_43", "0d0e610231994ff3663c662f4b960340"],
        ["data_batch_44", "386a2d3472fbd97330bb7b8bb7e0ff2f"],
        ["data_batch_45", "1124b3bbbe0c7f9c14f964c4533bd565"],
        ["data_batch_46", "18a53af11a51c44632f4ce3c0b012e5c"],
        ["data_batch_47", "05e6a4d27381dcd505e9bea7286929a6"],
        ["data_batch_48", "2c666e471cbd0b547d72bfe0aba04988"],
        ["data_batch_49", "1fde041df048985818326d4f587126c9"],
        ["data_batch_50", "8f2f127fab28d83b8b9182119db16732"],
        ["data_batch_51", "30b39c698faca92bc1a7c2a68efad3e8"],
        ["data_batch_52", "e9866820972ed2b23a46bea4cea1afd8"],
        ["data_batch_53", "379b92e4ad1c6128ec09703120a5e77f"],
    ]

    test_list = [["test_batch", "6c42ba92dc3239fd6ab5597b120741a0"]]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "d5d3d04e1d462b02b69285af3391ba25",
    }

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        remove_duplicates: bool = True,
        cut_threshold: float = 0.0,
        mb_cut=False,
    ):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.remove_duplicates = remove_duplicates
        self.cut_threshold = cut_threshold
        self.mb_cut = mb_cut

        # if download:
        #     self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted." + " You can use download=True to download it"
            )

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []  # object image data
        self.names = []  # object file names
        self.rgzid = []  # object RGZ ID
        self.mbflg = []  # object MiraBest flag
        self.sizes = []  # object largest angular sizes

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)

            with open(file_path, "rb") as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding="latin1")

                # print(entry.keys())

                self.data.append(entry["data"])
                self.names.append(entry["filenames"])
                self.rgzid.append(entry["src_ids"])
                self.mbflg.append(entry["mb_flag"])
                self.sizes.append(entry["LAS"])

        self.rgzid = np.vstack(self.rgzid).reshape(-1)
        self.sizes = np.vstack(self.sizes).reshape(-1)
        self.mbflg = np.vstack(self.mbflg).reshape(-1)
        self.names = np.vstack(self.names).reshape(-1)

        self.data = np.vstack(self.data).reshape(-1, 1, 150, 150)
        self.data = self.data.transpose((0, 2, 3, 1))

        self._load_meta()

        # Make cuts on the data
        n = self.__len__()
        idx_bool = np.ones(n, dtype=bool)

        if self.remove_duplicates:
            print(f"Removing duplicates from RGZ dataset...")
            idx_bool = np.zeros(n, dtype=bool)
            _, idx_unique = np.unique(self.data, axis=0, return_index=True)
            idx_bool[idx_unique] = True

            print(f"Removed {n - np.count_nonzero(idx_bool)} duplicate samples")
            n = np.count_nonzero(idx_bool)

        idx_bool *= self.sizes > self.cut_threshold
        print(f"Removing {n - np.count_nonzero(idx_bool)} samples below angular size threshold.")
        n = np.count_nonzero(idx_bool)

        if mb_cut:
            idx_bool *= self.mbflg == 0

            # Print number of MB samples removed
            print(f"Removed {n - np.count_nonzero(idx_bool)} MiraBest samples from RGZ")

        idx = np.argwhere(idx_bool).squeeze()

        self.data = self.data[idx]
        self.names = self.names[idx]
        self.rgzid = self.rgzid[idx]
        self.mbflg = self.mbflg[idx]
        self.sizes = self.sizes[idx]

        print(self.data.shape)

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError(
                "Dataset metadata file not found or corrupted."
                + " You can use download=True to download it"
            )
        with open(path, "rb") as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding="latin1")

            self.classes = data[self.meta["key"]]

        # self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            image (array): Image
        """

        img = self.data[index]
        las = self.sizes[index].squeeze()
        mbf = self.mbflg[index].squeeze()
        rgz = self.rgzid[index].squeeze()

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = np.reshape(img, (150, 150))
        img = img.squeeze()
        img = Image.fromarray(img, mode="L")

        if self.transform is not None:
            img = self.transform(img)

        # return img, self.sizes[index].squeeze()

        return img, {"size": las, "mb": mbf, "id": rgz, "index": index}

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        tmp = "train" if self.train is True else "test"
        fmt_str += "    Split: {}\n".format(tmp)
        fmt_str += "    Root Location: {}\n".format(self.root)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp)))
        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(
            tmp, self.target_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str

    def get_from_id(self, id):
        index = np.argwhere(self.rgzid.squeeze() == id).squeeze()

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = self.data[index]
        img = np.reshape(img, (150, 150))
        img = Image.fromarray(img, mode="L")

        if self.transform is not None:
            img = self.transform(img)

        return img
