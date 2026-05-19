from .fft import *

try:
    from .svd import *
except ModuleNotFoundError as e:
    if e.name != 'sklearn':
        raise
