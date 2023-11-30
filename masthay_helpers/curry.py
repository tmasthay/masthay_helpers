import re
import types

# import torch
import returns.curry
from typing import Any
import inspect


def curry_builtin(sig, module):
    # Regular expressions to parse the signature
    func_pattern = re.compile(r'(\w+)\((.*?)\) -> (\w+)')
    arg_pattern = re.compile(r'(\w+)(?:=(\w+))?')

    # Parse the function signature
    func_match = func_pattern.match(sig)
    func_name, args_str, return_type = func_match.groups()

    try:
        eval(return_type)
    except NameError:
        return_type = f'{module}.{return_type}'
        try:
            eval(return_type)
        except NameError:
            return_type = 'Any'

    args = []
    kwargs = []
    kwonly = False
    for arg in args_str.split(','):
        if arg.strip() == '*':
            kwonly = True
            continue

        arg_match = arg_pattern.match(arg.strip())
        if arg_match:
            arg_name, default_val = arg_match.groups()
            if kwonly:
                kwargs.append((arg_name, default_val))
            else:
                args.append((arg_name, default_val))

    def var_decl(name, default):
        if default:
            return f'{name}={default}'
        else:
            return name

    def multi_var_decl(d):
        return ', '.join([var_decl(name, default) for name, default in d])

    # Construct the wrapper function string
    def_str = f'import {module}\ndef {func_name}({multi_var_decl(args)}'
    if kwargs:
        def_str += ', *, ' + multi_var_decl(kwargs)
    def_str += f") -> {return_type}:\n"
    def_str += (
        "    return"
        f" {module}.{func_name}({', '.join([f'{name}={name}' for name, _ in args + kwargs])})"
    )

    # Compile and return the wrapper function
    namespace = {}
    exec(def_str, {'module': module}, namespace)
    return returns.curry.curry(namespace[func_name])


def sig_builtin(f, *, marker='.. function:: ', dispatch=0):
    lines = [e.strip() for e in f.__doc__.strip().split('\n')]
    lines = [e for e in lines if e]
    if dispatch == 0:
        return lines[0]
    else:
        res = [lines[0]] + [
            e.replace(marker, '') for e in lines if e.startswith(marker)
        ]
        if len(res) > dispatch:
            return res[dispatch]
        else:
            s = '\n    ' + '\n    '.join(res)
            raise ValueError(
                f"\ndispatch={dispatch} for {f.__name__} is out of range\n"
                f"Valid signatures below (zero-indexed):{s}"
            )


def curry(f, *, marker='.. function:: ', dispatch=0):
    try:
        return returns.curry(f)
    except Exception:
        return curry_builtin(
            sig_builtin(f, marker=marker, dispatch=dispatch), f.__module__
        )


if __name__ == "__main__":
    import torch
    import inspect

    print(inspect.signature(curry(torch.mean)))
    print(inspect.signature(curry(torch.mean, dispatch=1)))
