"""uetai utilities"""
import os
import torch
import gdown
# import zipfile
from pathlib import Path
from typing import Union

URL_PREFIX = "https://drive.google.com/uc?id="
CHUNK_SIZE = 32768


def download_from_url(url: str, save_dir: Union[str, Path] = None) -> str:
    save_dir.mkdir(parents=True, exist_ok=True)  # create save_dir
    save_dir = save_dir / Path(url).name
    print(f"Downloading {url} to {save_dir}")
    torch.hub.download_url_to_file(url, save_dir)
    # if save_dir.endswith(".zip"):  # unzip
    # unzip_file()
    return save_dir


def download_from_google_drive(id_or_url: str, save_path: Union[str, Path] = None,) -> str:
    # https://github.com/wkentaro/gdown
    if URL_PREFIX not in id_or_url:
        id_or_url = str(URL_PREFIX + id_or_url)
    if save_path is not None:
        if not Path(save_path).exists():
            Path(save_path).mkdir(parents=True, exist_ok=True)  # create save dir
    save_path = str(save_path) + str(os.path.sep)
    filename = gdown.download(url=id_or_url, output=save_path, quiet=False)
    # if 'zip' in filename:
    #     _unzip_file()
    #     filename = filename[:-4]  # remove '.zip'
    return os.path.join(save_path + filename)


# def _unzip_file(path: Union[str, Path], destination: Union[str, Path]) -> None:
#     path = path / Path(filename.name[: -len(".zip")])
#     print(f"Unzipping {filename} to {path}")
#     with zipfile.ZipFile(filename, "r") as zip_ref:
#         zip_ref.extractall(path)
#     url = str(path)
