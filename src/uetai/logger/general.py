"""uetai utilities"""
import sys
import subprocess
import argparse
import socket
import platform
from pathlib import Path
from subprocess import check_output

import pkg_resources as pkg


def parse_opt():
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
    parser.add_argument(
        '--weights', type=str, default='', help='initial weights path')
    parser.add_argument(
        '--data', type=str, default='data/data.yaml', help='dataset.yaml path')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument(
        '--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--project', default='', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='', help='save to project/name')
    parser.add_argument(
        '--upload_dataset', action='store_true',
        help='Upload dataset as W&B artifact table')
    parser.add_argument('--artifact_alias', type=str, default="latest",
                        help='version of dataset artifact to be used')

    opt = parser.parse_args()
    return parser, opt


def try_except(func):
    """try-except function
    """
    # try-except function. Usage: @try_except decorator
    def handler(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as exception:
            print(exception)

    return handler


def install_package(package):
    """install `package` by `pip install -m <<package>>`
    """
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False):
    """check version vs. required version
    """
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)
    assert result, (f'{name}{minimum} required,'
                    f'but {name}{current} is currently installed')


def check_python(minimum='3.6.2'):
    """check python version vs. required python version
    """
    # Check current python version vs. required python version
    check_version(platform.python_version(), minimum, name='Python ')


def check_online():
    """Check internet connectivity"""
    try:
        socket.create_connection(("1.1.1.1", 443), 5)  # check host accessibility
        return True
    except OSError:
        return False


@try_except
def check_requirements(requirements='requirements.txt', exclude=()):
    """Check installed dependencies meet requirements
    (pass *.txt file or list of packages)
    """
    prefix = colorstr('red', 'bold', 'requirements:')
    check_python()  # check python version
    if isinstance(requirements, (str, Path)):  # requirements.txt file
        file = Path(requirements)
        assert file.exists(), f"{prefix} {file.resolve()} not found, check failed."
        requirements = [f'{x.name}{x.specifier}'
                        for x in pkg.parse_requirements(file.open())
                        if x.name not in exclude]
    else:  # list or tuple of packages
        requirements = [x for x in requirements if x not in exclude]

    number_of_package = 0  # number of packages updates
    for package in requirements:
        try:
            pkg.require(package)

        # DistributionNotFound or VersionConflict if requirements not met
        except Exception as expection:
            print(
                f"{prefix} {package} {expection} not found and is required, "
                "attempting auto-update...")
            try:
                assert check_online(), f"'pip install {package}' skipped (offline)"
                print(check_output(f"pip install '{package}'", shell=True).decode())
                number_of_package += 1
            except Exception as expection:
                print(f'{prefix} {expection}')

    if number_of_package:  # if packages updated
        source = file.resolve() if 'file' in locals() else requirements
        string = (
            f"{prefix} {number_of_package} package{'s' * (number_of_package > 1)} "
            f"updated per {source}\n"
            f"{prefix} ⚠️ "
            f"{'Restart runtime or rerun command for updates to take effect'}\n")
        print(emojis(string))


def emojis(string=''):
    """Return platform-dependent emoji-safe version of string
    """
    if platform.system() == 'Windows':
        return string.encode().decode('ascii', 'ignore')
    return string


def colorstr(*inputs):
    """return platform-dependent emoji-safe version of string

    Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  \
colorstr('blue', 'hello world')
    """
    *args, string = inputs if len(inputs) > 1 else ('blue', 'bold', inputs[0])
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
