import argparse
import os
import platform
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from subprocess import check_output

import pkg_resources as pkg
import torch
import yaml


class Args:
    def __init__(self):
        self.parser, self.opt = self.__parse_opt()

    def get_opt(self):
        return self.opt

    def get_parser(self):
        return self.parser

    def __parse_opt(self):
        """
        Setup arguments for this run, including:
            - weights (str): initial weights local path or W&B path
            - data (str): path to .yaml data file
            - epochs (int): total epochs
            - batch_size (int): total batch size for all GPUs
            - project (str): W&B project name, save to project/name
            - entity (str): W&B entity
            - upload_dataset (boolean): upload dataset as W&B  artifact
            - artifact_alias (str): version of dataset artifact to be used
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', type=str, default='', help='initial weights path')
        parser.add_argument('--data', type=str, default='data/data.yaml', help='dataset.yaml path')
        parser.add_argument('--epochs', type=int, default=5)
        parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
        parser.add_argument('--project', default='', help='save to project/name')
        parser.add_argument('--entity', default=None, help='W&B entity')
        parser.add_argument('--name', default='', help='save to project/name')
        parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
        parser.add_argument('--artifact_alias', type=str, default="latest",
                            help='version of dataset artifact to be used')

        opt = parser.parse_args()
        return parser, opt


def try_except(func):
    # try-except function. Usage: @try_except decorator
    def handler(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(e)

    return handler


def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False):
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)
    assert result, f'{name}{minimum} required, but {name}{current} is currently installed'


def check_python(minimum='3.6.2'):
    # Check current python version vs. required python version
    check_version(platform.python_version(), minimum, name='Python ')


def check_online():
    # Check internet connectivity
    import socket
    try:
        socket.create_connection(("1.1.1.1", 443), 5)  # check host accessibility
        return True
    except OSError:
        return False


def check_dataset(data, autodownload=True):
    # Download and/or unzip dataset if not found locally
    # Download (optional)
    extract_dir = ''
    if isinstance(data, (str, Path)) and str(data).endswith('.zip'):  # i.e. gs://bucket/dir/coco128.zip
        download(data, dir='../datasets', unzip=True, delete=False, curl=False, threads=1)
        data = next((Path('../datasets') / Path(data).stem).rglob('*.yaml'))
        extract_dir, autodownload = data.parent, False

    # Read yaml (optional)
    if isinstance(data, (str, Path)):
        with open(data, encoding='ascii', errors='ignore') as f:
            data = yaml.safe_load(f)  # dictionary

    # Parse yaml
    path = extract_dir or Path(data.get('path') or '')  # optional 'path' default to '.'
    for k in 'train', 'val', 'test':
        if data.get(k):  # prepend path
            data[k] = str(path / data[k]) if isinstance(data[k], str) else [str(path / x) for x in data[k]]

    train, val, test, s = [data.get(x) for x in ('train', 'val', 'test', 'download')]

    # Check local dataset
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        if not all(x.exist() for x in val):
            print('\nWARNING: Dataset not found, nonexistent paths: %s' % [str(x) for x in val if not x.exists()])
            if s and autodownload:  # download script
                if s.startswith('http') and s.endswith('.zip'):  # URL
                    f = Path(s).name  # filename
                    print(f'Downloading {s} ...')
                    torch.hub.download_url_to_file(s, f)
                    root = path.parent if 'path' in data else '..'  # unzip directory i.e. '../'
                    Path(root).mkdir(parents=True, exist_ok=True)  # create root
                    r = os.system(f'unzip -q {f} -d {root} && rm {f}')  # unzip
                elif s.startswith('bash '):  # bash script
                    print(f'Running {s} ...')
                    r = os.system(s)
                else:  # python script
                    r = exec(s, {'yaml': data})  # return None
                print('Dataset autodownload %s\n' % ('success' if r in (0, None) else 'failure'))  # print result
            else:
                raise Exception('Dataset not found.')

    return data  # dictionary


def download(url, dir='.', unzip=True, delete=True, curl=False, threads=1):
    # Multi-threaded file download and unzip function, used in data.yaml for autodownload
    def download_one(url, dir):
        # Download 1 file
        f = dir / Path(url).name  # filename
        if Path(url).is_file():  # exists in current path
            Path(url).rename(f)  # move to dir
        elif not f.exists():
            print(f'Downloading {url} to {f}...')
            if curl:
                os.system(f"curl -L '{url}' -o '{f}' --retry 9 -C -")  # curl download, retry and resume on fail
            else:
                torch.hub.download_url_to_file(url, f, progress=True)  # torch download
        if unzip and f.suffix in ('.zip', '.gz'):
            print(f'Unzipping {f}...')
            if f.suffix == '.zip':
                s = f'unzip -qo {f} -d {dir}'  # unzip -quiet -overwrite
            elif f.suffix == '.gz':
                s = f'tar xfz {f} --directory {f.parent}'  # unzip
            if delete:  # delete zip file after unzip
                s += f' && rm {f}'
            os.system(s)

    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)  # make directory
    if threads > 1:
        pool = ThreadPool(threads)
        pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))  # multi-threaded
        pool.close()
        pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            download_one(u, dir)


@try_except
def check_requirements(requirements='requirements.txt', exclude=()):
    # Check installed dependencies meet requirements (pass *.txt file or list of packages)
    prefix = colorstr('red', 'bold', 'requirements:')
    check_python()  # check python version
    if isinstance(requirements, (str, Path)):  # requirements.txt file
        file = Path(requirements)
        assert file.exists(), f"{prefix} {file.resolve()} not found, check failed."
        requirements = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements(file.open()) if x.name not in exclude]
    else:  # list or tuple of packages
        requirements = [x for x in requirements if x not in exclude]

    n = 0  # number of packages updates
    for r in requirements:
        try:
            pkg.require(r)
        except Exception as e:  # DistributionNotFound or VersionConflict if requirements not met
            print(f"{prefix} {r} not found and is required by YOLOv5, attempting auto-update...")
            try:
                assert check_online(), f"'pip install {r}' skipped (offline)"
                print(check_output(f"pip install '{r}'", shell=True).decode())
                n += 1
            except Exception as e:
                print(f'{prefix} {e}')

    if n:  # if packages updated
        source = file.resolve() if 'file' in locals() else requirements
        s = f"{prefix} {n} package{'s' * (n > 1)} updated per {source}\n" \
            f"{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
        print(emojis(s))


def emojis(str=''):
    # Return platform-dependent emoji-safe version of string
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']
