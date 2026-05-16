import os
import sys
import tempfile
from pathlib import Path

_TEST_HOME = Path(tempfile.gettempdir()) / 'neuralib-pytest-home'
_TEST_HOME.mkdir(parents=True, exist_ok=True)

# Napari installs a pytest plugin with autouse fixtures. When that plugin is
# loaded, it may build theme SVG caches before tests run, even for tests that do
# not import napari. Keep those writes inside the project test cache instead of
# the user's cache directory, which can be unavailable in sandboxed runners.
os.environ['HOME'] = str(_TEST_HOME)
os.environ['XDG_CACHE_HOME'] = str(_TEST_HOME / '.cache')
os.environ['XDG_CONFIG_HOME'] = str(_TEST_HOME / '.config')
os.environ['MPLCONFIGDIR'] = str(_TEST_HOME / '.matplotlib')

try:
    import neuralib.io.core as io_core

    io_core.CACHE_DIRECTORY = Path(os.environ['XDG_CACHE_HOME'])
    io_core.NEUROLIB_CACHE_DIRECTORY = io_core.CACHE_DIRECTORY / 'neuralib'
    io_core.NEUROLIB_DATASET_DIRECTORY = io_core.NEUROLIB_CACHE_DIRECTORY / 'dataset'
    io_core.ATLAS_CACHE_DIRECTORY = io_core.NEUROLIB_CACHE_DIRECTORY / 'atlas'
    io_core.CASCADE_MODEL_CACHE_DIRECTORY = io_core.NEUROLIB_CACHE_DIRECTORY / 'cascade'
    _TEST_CACHE = io_core.NEUROLIB_CACHE_DIRECTORY / 'napari'
except ImportError:
    _TEST_CACHE = Path(os.environ['XDG_CACHE_HOME']) / 'neuralib' / 'napari'

_TEST_CACHE.mkdir(parents=True, exist_ok=True)

try:
    from napari.utils import _appdirs

    _appdirs.user_cache_dir = lambda: str(_TEST_CACHE)
    if 'napari.resources._icons' in sys.modules:
        sys.modules['napari.resources._icons'].user_cache_dir = _appdirs.user_cache_dir
except ImportError:
    pass
