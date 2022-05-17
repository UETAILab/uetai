"""checking function"""
import sys
import subprocess
import warnings


def _check_version(name):
    """check package version"""
    latest_version = str(subprocess.run(
        [sys.executable, '-m', 'pip', 'install', f'{name}==random'],
        capture_output=True, text=True))
    latest_version = latest_version[latest_version.find('(from versions:')+15:]
    latest_version = latest_version[:latest_version.find(')')]
    latest_version = latest_version.replace(' ', '').rsplit(',', maxsplit=1)[-1]

    current_version = str(subprocess.run(
        [sys.executable, '-m', 'pip', 'show', f'{name}'],
        capture_output=True, text=True))
    current_version = current_version[current_version.find('Version:')+8:]
    current_version = current_version[:current_version.find('\\n')].replace(' ', '')

    if latest_version == current_version:
        return True

    return False


def check_uetai_version():
    """check uetai version

    :return: True if version is up-to-date
    """
    if not _check_version('uetai'):
        warnings.warn('current `uetai` package is out of date, please run `pip install --upgrade uetai`')
        return False
    return True
