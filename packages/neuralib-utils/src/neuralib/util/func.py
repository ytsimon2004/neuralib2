"""
Dynamic function generator
==========================


"""

import inspect
import textwrap
from collections.abc import Callable
from typing import Any

__all__ = ['create_fn',
           'get_func_default']

PARA_TYPE = str | tuple[str, str | type] | tuple[str, None | str | type, str]
PARA_TYPE_LIST = list[PARA_TYPE]
RET_TYPE = None | str | type
SIGN_TYPE = PARA_TYPE_LIST | tuple[PARA_TYPE_LIST, RET_TYPE] | Callable[..., Any]


def _create_fn_para_from_func(func: Callable[..., Any]) -> str:
    para = []

    s = inspect.signature(func)
    p = None
    for arg_name, arg in s.parameters.items():

        if arg.annotation is inspect.Parameter.empty:
            arg_type = ''
        elif isinstance(arg.annotation, str):
            arg_type = f': {arg.annotation}'
        else:
            arg_type = f': {arg.annotation.__name__}'

        if arg.kind != p:
            if p == inspect.Parameter.POSITIONAL_ONLY:
                para.append('/')
                if arg.kind == inspect.Parameter.KEYWORD_ONLY:
                    para.append('*')
            elif p == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                if arg.kind == inspect.Parameter.KEYWORD_ONLY:
                    para.append('*')

        if arg.kind == inspect.Parameter.VAR_POSITIONAL:
            arg_prefix = '*'
        elif arg.kind == inspect.Parameter.VAR_KEYWORD:
            arg_prefix = '**'
        else:
            arg_prefix = ''

        if arg.default is inspect.Parameter.empty:
            arg_default = ''
        else:
            arg_default = f'={repr(arg.default)}'

        para.append(f'{arg_prefix}{arg_name}{arg_type}{arg_default}')

        p = arg.kind

    if s.return_annotation is inspect.Parameter.empty:
        ret_ann = ''
    elif isinstance(s.return_annotation, str):
        ret_ann = f'-> {s.return_annotation}'
    else:
        ret_ann = f'-> {s.return_annotation.__name__}'

    para = ', '.join(para)
    return f'({para}){ret_ann}'


def _create_fn_para_from_list(sign: PARA_TYPE_LIST | tuple[PARA_TYPE_LIST, RET_TYPE]) -> str:
    para = []

    if isinstance(sign, list):
        ret_ann = ''
    else:
        sign, ret_type = sign
        if ret_type is None:
            ret_ann = '-> None'
        elif isinstance(ret_type, str):
            ret_ann = f'-> {ret_type}'
        elif isinstance(ret_type, type):
            ret_ann = f'-> {ret_type.__name__}'
        else:
            raise TypeError()

    for arg in sign:
        if isinstance(arg, str):
            para.append(arg)
        elif len(arg) == 2:
            arg_name, arg_type = arg
            if isinstance(arg_type, str):
                para.append(f'{arg_name}: {arg_type}')
            elif isinstance(arg_type, type):
                para.append(f'{arg_name}: {arg_type.__name__}')
            else:
                raise TypeError()
        elif len(arg) == 3:
            arg_name, arg_type, default = arg
            if arg_type is None:
                para.append(f'{arg_name}={default}')
            elif isinstance(arg_type, str):
                para.append(f'{arg_name}: {arg_type}={default}')
            elif isinstance(arg_type, type):
                para.append(f'{arg_name}: {arg_type.__name__}={default}')
            else:
                raise TypeError()
        else:
            raise TypeError()

    para = ', '.join(para)
    return f'({para}){ret_ann}'


def create_fn(name: str,
              sign: SIGN_TYPE,
              body: str = 'pass',
              *,
              globals: dict[str, Any] | None = None,
              locals: dict[str, Any] | None = None) -> Callable[..., Any]:
    """

    Example:

    >>> add = create_fn('add', (['a', 'b'], int), 'return a + b')
    >>> add(1, 2)
    3
    >>> def add_sign(a: int, b:int) -> int:
    ...     pass
    >>> add = create_fn('add', add_sign, 'return a + b')
    >>> add(1, 2)
    3

    Signature Example

    >>> def f(a, b:int, c=0, d:int=1) -> int:
    (['a', ('b', int), ('c', None, '0'), ('d', int, '1')], int)

    reference: dataclasses._create_fn

    :param name:
    :param sign: `([arg_name|(arg_name, arg_type)|(arg_name, arg_type?, str(default)),...], ret_type)`
    :param body:
    :param globals:
    :param locals:
    :return:
    """
    local_vars = locals or {}

    if callable(sign):
        signature = _create_fn_para_from_func(sign)
    else:
        signature = _create_fn_para_from_list(sign)

    func = textwrap.indent(f'def {name}{signature}:\n' + textwrap.indent(body, prefix='  '), '  ')

    local_params = ', '.join(local_vars.keys())
    code = f'def __neuralib_dynamic_generate_function__({local_params}):\n{func}\n  return {name}'

    # print(code)
    namespace: dict[str, Any] = {}
    exec(code, globals, namespace)
    return namespace['__neuralib_dynamic_generate_function__'](**local_vars)


def get_func_default(f: Callable[..., Any], arg: str) -> Any:
    import inspect
    return inspect.signature(f).parameters[arg].default
