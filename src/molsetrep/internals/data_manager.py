import requests
import shutil
import tarfile
from typing import Union, List
from os.path import basename
from pathlib import Path
from itertools import zip_longest
import platformdirs
from tqdm.auto import tqdm


class DataManager:
    def __init__(self):
        self.data_path = platformdirs.user_data_path("molsetrep", "daenuprobst")
        if not self.data_path.exists():
            try:
                self.data_path.mkdir(parents=True, exist_ok=True)
            except IOError:
                print(
                    f"Could not create data directory '{str(self.data_path)}', please check that you have the required permissions."
                )

    def download_file(
        self,
        url: Union[List[str], str],
        local_name: str = None,
        overwrite: bool = False,
        tar: bool = False,
    ):
        if isinstance(url, str):
            url = [url]

        if isinstance(local_name, str) or local_name is None:
            local_name = [local_name]

        i = 1
        for u, ln in zip_longest(url, local_name):
            if ln is None:
                ln = basename(u).split("?")[0]

            target_path = Path(self.data_path, ln)

            if target_path.exists() and not overwrite:
                continue

            print(f"{i}/{len(url)}: {u} -> {str(target_path)}")
            with requests.get(u, stream=True) as r:
                total_length = int(r.headers.get("Content-Length"))
                with tqdm.wrapattr(r.raw, "read", total=total_length, desc="") as raw:
                    with open(target_path, "wb") as output:
                        shutil.copyfileobj(raw, output)
            i += 1

            if tar:
                tar_path = Path(self.data_path, ln)
                with tarfile.open(tar_path) as tf:
                    tf.extractall(self.data_path)
                tar_path.unlink()

    def get_path(self, name: str):
        path = Path(self.data_path, name)

        if not path.exists():
            raise Exception(
                f"The file '{str(path)}' does not exist. If this error persists please delete the folder '{str(self.data_path)}' and reinstall the package."
            )

        return path

    def exists(self, name: str):
        return Path(self.data_path, name).exists()

    def new_path(self, name: str):
        return Path(self.data_path, name)
