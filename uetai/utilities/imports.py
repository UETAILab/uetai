"""Import utilities."""
import importlib
from importlib.util import find_spec


def _package_available(package_name: str) -> bool:
    """Check if a package is available in your environment.
    >>> _package_available('os')
    True
    >>> _package_available('bla')
    False
    """
    try:
        return find_spec(package_name) is not None
    except ModuleNotFoundError:
        return False


def module_available(module_path: str) -> bool:
    """Check if a module path is available in your environment.
    >>> module_available('os')
    True
    >>> module_available('os.bla')
    False
    >>> module_available('bla.bla')
    False
    """
    module_names = module_path.split(".")
    if not _package_available(module_names[0]):
        return False
    try:
        module = importlib.import_module(module_names[0])
    except ImportError:
        return False
    for name in module_names[1:]:
        if not hasattr(module, name):
            return False
        module = getattr(module, name)
    return True
