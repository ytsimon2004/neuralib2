from pathlib import Path

import json
import numpy as np
from typing import Any

from neuralib.typing import PathLike
from neuralib.util.verbose import print_save, print_load

__all__ = ['JsonEncodeHandler',
           'load_json',
           'save_json']


class JsonEncodeHandler(json.JSONEncoder):
    """Extend from the JSONEncoder class and handle the conversions in a default method

    **Usage**: Add kwarg, e.g., ``json_dump(..., cls=JsonEncodeHandler)``
    """

    def default(self, o: Any) -> Any:
        """handle array/Path type"""
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.bool_):
            return bool(o)
        elif isinstance(o, Path):
            return str(o)
        else:
            return json.JSONEncoder.default(self, o)


def load_json(filepath: PathLike, verbose=True, **kwargs) -> dict[str, Any]:
    """
    Load a json as dict

    :param filepath: json filepath
    :param verbose: verbose load
    :param kwargs: additional arguments to ``json.load()``
    :return:
    """
    with open(filepath, "r") as file:
        if verbose:
            print_load(filepath)
        return json.load(file, **kwargs)


def save_json(filepath: PathLike, dict_obj: dict[str, Any], verbose=True, **kwargs) -> None:
    """
    Save dict as a json file

    :param filepath: json filepath
    :param dict_obj: dictionary object
    :param verbose: verbose save output
    :param kwargs: additional arguments to ``json.dump()``
    """
    with open(filepath, "w") as outfile:
        if verbose:
            print_save(filepath)
        json.dump(dict_obj, outfile, sort_keys=True, indent=4, cls=JsonEncodeHandler, **kwargs)
