from pathlib import Path


class Path_Handler:
    """Handle and generate paths in project directory"""

    def __init__(self):
        self.dict = {}

    def fill_dict(self):
        """Create dictionary of required paths"""
        path_dict = {}
        path_dict["root"] = Path(__file__).resolve().parent.parent.parent
        path_dict["project"] = Path(__file__).resolve().parent.parent

        path_dict["data"] = path_dict["root"] / "_data"

        path_dict["files"] = path_dict["project"] / "files"
        path_dict["main"] = path_dict["project"] / "byol_main"
        path_dict["config"] = path_dict["project"] / "config"

        path_dict["rgz"] = path_dict["data"] / "rgz"
        path_dict["mb"] = path_dict["data"] / "mb"
        path_dict["imagenette"] = path_dict["data"] / "imagenette-160"
        path_dict["stl10"] = path_dict["data"] / "stl10"
        path_dict["cifar10"] = path_dict["data"] / "cifar10"
        path_dict["gzmnist"] = path_dict["data"] / "gzmnist"
        path_dict["gz2"] = path_dict["data"] / "gz2"
        path_dict["legs"] = path_dict["data"] / "legs"
        path_dict["decals_dr5"] = path_dict["data"] / "decals_dr5"
        path_dict["rings"] = path_dict["data"] / "rings"
        path_dict["tidal"] = path_dict["data"] / "tidal"
        path_dict["candels"] = path_dict["data"] / "candels"

        self.dict = path_dict

    def create_paths(self):
        """Create missing directories"""
        for path in self.dict.values():
            if not Path.exists(path):
                Path.mkdir(path)

    def _dict(self):
        """Generate path dictionary, create any missing directories and return dictionary"""
        self.fill_dict()
        self.create_paths()
        return self.dict
