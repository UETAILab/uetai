"""uetai utilities"""
import os
import torch
import gdown
import zipfile
from pathlib import Path
from typing import Union

URL_PREFIX = "https://drive.google.com/uc?id="
CHUNK_SIZE = 32768


def download_from_url(url: str, save_dir: Union[str, Path] = './') -> str:
    Path(save_dir).mkdir(parents=True, exist_ok=True)  # create save_dir
    save_dir = Path(save_dir) / Path(url).name
    print(f"Downloading {url} to {save_dir}")
    torch.hub.download_url_to_file(url, save_dir)
    if str(save_dir).endswith(".zip"):  # unzip
        save_dir = _unzip_file(path=save_dir)
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
    assert filename is not None, 'Unable to download content, please check your `id` or `url`'
    if filename.endswith('.zip'):
        save_path = _unzip_file(filename)
    return save_path


def _unzip_file(path: Union[str, Path]) -> str:
    filename = Path(path).name
    save_path = str(path)[:-len(".zip")]
    Path(save_path).mkdir(parents=True, exist_ok=True)
    print(f"Unzipping {filename} to {save_path}")
    with zipfile.ZipFile(path, "r") as zip_ref:
        zip_ref.extractall(save_path)
    return str(save_path)
