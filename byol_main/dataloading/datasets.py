import os
import sys
import numpy as np
import torch.utils.data as data

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity


class MiraBest_full(data.Dataset):
    """​
    Inspired by `HTRU1 <https://as595.github.io/HTRU1/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``MiraBest-full.py` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again."""

    base_folder = "batches"
    url = "http://www.jb.man.ac.uk/research/MiraBest/full_dataset/MiraBest_full_batches.tar.gz"
    filename = "MiraBest_full_batches.tar.gz"
    tgz_md5 = "965b5daa83b9d8622bb407d718eecb51"
    train_list = [
        ["data_batch_1", "b15ae155301f316fc0b51af16b3c540d"],
        ["data_batch_2", "0bf52cc1b47da591ed64127bab6df49e"],
        ["data_batch_3", "98908045de6695c7b586d0bd90d78893"],
        ["data_batch_4", "ec9b9b77dc019710faf1ad23f1a58a60"],
        ["data_batch_5", "5190632a50830e5ec30de2973cc6b2e1"],
        ["data_batch_6", "b7113d89ddd33dd179bf64cb578be78e"],
        ["data_batch_7", "626c866b7610bfd08ac94ca3a17d02a1"],
    ]

    test_list = [
        ["test_batch", "5e443302dbdf3c2003d68ff9ac95f08c"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "e1b5450577209e583bc43fbf8e851965",
    }

    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

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
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 1, 150, 150)
        self.data = self.data.transpose((0, 2, 3, 1))

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
        img = Image.fromarray(img, mode="labelled")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

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
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(
            tmp, self.target_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str


class MB_nohybrids(MiraBest_full):
    """
    Child class to load only frii/frii samples and collapses classes to 0 for FRI and 1 for FRII
    """

    def __init__(self, *args, **kwargs):
        super(MB_nohybrids, self).__init__(*args, **kwargs)

        fr1_list = [0, 1, 2, 3, 4]
        fr2_list = [5, 6, 7]
        exclude_list = [8, 9]

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
        else:
            targets = np.array(self.targets)
            exclude = np.array(exclude_list).reshape(1, -1)
            exclude_mask = ~(targets.reshape(-1, 1) == exclude).any(axis=1)
            fr1 = np.array(fr1_list).reshape(1, -1)
            fr2 = np.array(fr2_list).reshape(1, -1)
            fr1_mask = (targets.reshape(-1, 1) == fr1).any(axis=1)
            fr2_mask = (targets.reshape(-1, 1) == fr2).any(axis=1)
            targets[fr1_mask] = 0  # set all FRI to Class 0
            targets[fr2_mask] = 1  # set all FRII to Class 1
            self.data = self.data[exclude_mask]
            self.targets = targets[exclude_mask].tolist()


class MBFRConfident(MiraBest_full):
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
        else:
            targets = np.array(self.targets)
            exclude = np.array(exclude_list).reshape(1, -1)
            exclude_mask = ~(targets.reshape(-1, 1) == exclude).any(axis=1)
            fr1 = np.array(fr1_list).reshape(1, -1)
            fr2 = np.array(fr2_list).reshape(1, -1)
            fr1_mask = (targets.reshape(-1, 1) == fr1).any(axis=1)
            fr2_mask = (targets.reshape(-1, 1) == fr2).any(axis=1)
            targets[fr1_mask] = 0  # set all FRI to Class 0
            targets[fr2_mask] = 1  # set all FRII to Class 1
            self.data = self.data[exclude_mask]
            self.targets = targets[exclude_mask].tolist()


class MBFRUncertain(MiraBest_full):
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


class RGZ20k(data.Dataset):
    """`RGZ 20k <>`_Dataset

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

    base_folder = "rgz20k-batches-py"
    url = "http://www.jb.man.ac.uk/research/ascaife/rgz20k-batches-python.tar.gz"
    filename = "rgz20k-batches-python.tar.gz"
    tgz_md5 = "3a85fb4167fb08619e36e77bbba40896"
    train_list = [
        ["data_batch_1", "9ffdcb485fc0c96e1afa1cdc342f00e7"],
        ["data_batch_2", "8961fa4a2fb5a8482ec5606e3d501fb4"],
        ["data_batch_3", "8cbf4fa7b34282b8f1522a350df4a882"],
        ["data_batch_4", "a58c94b5905d0ad2d97ba3a8895538c9"],
        ["data_batch_5", "13c9132ee374b7b63dac22c17a412d86"],
        ["data_batch_6", "232ff5854df09d5a68471861b1ee5576"],
        ["data_batch_7", "bea739fe0f7bd6ffb77b8e7def7f2edf"],
        ["data_batch_8", "48b23b3f0f37478b61ccca462fe53917"],
        ["data_batch_9", "2ac1208dec744cc136d0cd7842c180a2"],
        ["data_batch_10", "4ad68d1d8179da93ca1f8dfa7fe8e11d"],
    ]

    test_list = [
        ["test_batch", "35a588227816ad08d37112f23b6e2ea4"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "8f3138fbc912134239a779a1f3f6eaf8",
    }

    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
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
        # las = self.sizes[index]
        # mbf = self.mbflg[index]
        # rgz = self.rgzid[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = np.reshape(img, (150, 150))
        img = Image.fromarray(img, mode="labelled")

        if self.transform is not None:
            img = self.transform(img)

        return img, -1

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
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(
            tmp, self.target_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str


class RGZ20k_test(data.Dataset):
    """`RGZ 20k <>`_Dataset

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

    base_folder = "rgz20k-batches-py"
    url = "http://www.jb.man.ac.uk/research/ascaife/rgz20k-batches-python.tar.gz"
    filename = "rgz20k-batches-python.tar.gz"
    tgz_md5 = "3a85fb4167fb08619e36e77bbba40896"
    train_list = [
        ["data_batch_1", "9ffdcb485fc0c96e1afa1cdc342f00e7"],
        ["data_batch_2", "8961fa4a2fb5a8482ec5606e3d501fb4"],
        ["data_batch_3", "8cbf4fa7b34282b8f1522a350df4a882"],
        ["data_batch_4", "a58c94b5905d0ad2d97ba3a8895538c9"],
        ["data_batch_5", "13c9132ee374b7b63dac22c17a412d86"],
        ["data_batch_6", "232ff5854df09d5a68471861b1ee5576"],
        ["data_batch_7", "bea739fe0f7bd6ffb77b8e7def7f2edf"],
        ["data_batch_8", "48b23b3f0f37478b61ccca462fe53917"],
        ["data_batch_9", "2ac1208dec744cc136d0cd7842c180a2"],
        ["data_batch_10", "4ad68d1d8179da93ca1f8dfa7fe8e11d"],
    ]

    test_list = [
        ["test_batch", "35a588227816ad08d37112f23b6e2ea4"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "8f3138fbc912134239a779a1f3f6eaf8",
    }

    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
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
        las = self.sizes[index]
        mbf = self.mbflg[index]
        rgz = self.rgzid[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = np.reshape(img, (150, 150))
        img = Image.fromarray(img, mode="labelled")

        if self.transform is not None:
            img = self.transform(img)

        return img, rgz, las, mbf

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
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(
            tmp, self.target_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str
