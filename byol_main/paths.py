from pathlib import Path


class Path_Handler:
    """Handle and generate paths in project directory"""

    def __init__(self, **kwargs):

        # use defaults except where specified in kwargs e.g. Path_Handler(data=some_alternative_dir)
        path_dict = {}
        path_dict["root"] = kwargs.get("root", Path(__file__).resolve().parent.parent.parent)
        path_dict["project"] = kwargs.get(
            "project", Path(__file__).resolve().parent.parent
        )  # i.e. this repo

        path_dict["data"] = kwargs.get("data", path_dict["root"] / "_data")

        path_dict["files"] = kwargs.get("files", path_dict["project"] / "files")
        path_dict["main"] = kwargs.get("main", path_dict["project"] / "byol_main")
        path_dict["config"] = kwargs.get("config", path_dict["project"] / "config")

        for key, path_str in path_dict.copy().items():
            path_dict[key] = Path(path_str)

        self.path_dict = path_dict

    def fill_dict(self):
        """Create dictionary of required paths"""

        self.path_dict["rgz"] = self.path_dict["data"] / "rgz"
        self.path_dict["mb"] = self.path_dict["data"] / "mb"
        self.path_dict["imagenette"] = self.path_dict["data"] / "imagenette-160"
        self.path_dict["stl10"] = self.path_dict["data"] / "stl10"
        self.path_dict["cifar10"] = self.path_dict["data"] / "cifar10"
        self.path_dict["gzmnist"] = self.path_dict["data"] / "gzmnist"
        self.path_dict["gz2"] = self.path_dict["data"] / "gz2"
        self.path_dict["legs"] = (
            self.path_dict["data"] / "does_nothing"
        )  # will ignore this and use hardcoded paths
        # TODO this is going to get really ugly, needs a rethink?
        self.path_dict["legs_and_rings"] = self.path_dict[
            "data"
        ]  # sets its own subdirectories in foundation.datasets.legs.legs_and_rings
        self.path_dict["mixed"] = self.path_dict[
            "data"
        ]  # sets its own subdirectories in foundation.datasets.legs.legs_and_rings
        self.path_dict["decals_dr5"] = self.path_dict["data"] / "decals_dr5"
        self.path_dict["rings"] = self.path_dict["data"] / "rings"
        self.path_dict["tidal"] = self.path_dict["data"] / "tidal"
        self.path_dict["candels"] = self.path_dict["data"] / "candels"
        self.path_dict["candels"] = self.path_dict["data"] / "hubble"

    def create_paths(self):
        """Create missing directories"""
        for path in self.path_dict.values():
            create_path(path)

    def _dict(self):
        """Generate path dictionary, create any missing directories and return dictionary"""
        self.fill_dict()
        self.create_paths()
        return self.path_dict


def create_path(path):
    if not Path.exists(path):
        Path.mkdir(path)
