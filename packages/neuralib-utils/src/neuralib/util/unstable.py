import functools
import warnings
from typing import Any, Callable, TypeVar, cast

__all__ = ['unstable']

T = TypeVar('T')


def unstable(doc: bool = True,
             runtime: bool = True,
             mark_all: bool = False,
             method: str = '__init__') -> Callable[[T], T]:
    """
    A decorator that mark class/function unstable.

    :param doc: Add unstable message in function/class document.
    :param runtime: Add runtime warning when invoking function/initialization.
    :param mark_all: Mark all public methods when decorate on a class, If False, then only mark ``method_name``
    :param method: Name of method in class to be marked. defaults to '__init__'
    """

    def _decorator(obj: T) -> T:
        if getattr(obj, '__unstable_marker', False):
            return obj

        setattr(obj, '__unstable_marker', True)

        if doc:
            new_doc = 'UNSTABLE.'
            if obj.__doc__ is not None:
                new_doc += f'\n{obj.__doc__}'

            try:
                obj.__doc__ = new_doc
            except AttributeError:
                # AttributeError: 'wrapper_descriptor' object has no attribute
                pass

        if isinstance(obj, type):  # class
            if mark_all:
                for meth in dir(obj):
                    if callable(f := getattr(obj, meth, None)) and (not meth.startswith('_') or meth in ('__init__',)):
                        setattr(obj, meth, unstable(doc=doc, runtime=runtime, mark_all=mark_all)(f))
            else:
                f = obj.__dict__.get(method)
                if f is not None:
                    setattr(obj, method, unstable(doc=doc, runtime=runtime, mark_all=mark_all)(f))
                else:
                    raise RuntimeError(f'Method {method} in {obj.__qualname__} not found')

            return obj

        else:  # func/meth
            if runtime:
                func = cast(Callable[..., Any], obj)

                @functools.wraps(func)
                def _unstable_meth(*args: Any, **kwargs: Any) -> Any:
                    warnings.warn(f"{obj.__qualname__} is unstable and under development.", stacklevel=2)
                    return func(*args, **kwargs)

                return cast(T, _unstable_meth)
            else:
                return obj

    return _decorator
