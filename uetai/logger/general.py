"""uetai utilities"""
import socket
import platform
from pathlib import Path
from subprocess import check_output

import pkg_resources as pkg


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
        except Exception as exception:  # pragma: no cover
            print(
                f"{prefix} {package} {exception} not found and is required, "
                "attempting auto-update...")
            try:
                assert check_online(), f"'pip install {package}' skipped (offline)"
                print(check_output(f"pip install '{package}'", shell=True).decode())
                number_of_package += 1
            except Exception as exception:
                print(f'{prefix} {exception}')

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
