"""
These functions will eventually be deprecated but are still used in the
misfit_toys library, so they are included here to not break that
other project.
"""

import os
from fnmatch import fnmatch
import tabulate as tab
import re
from termcolor import colored
from subprocess import check_output as co, CalledProcessError
from functools import wraps
import argparse
from mh.core import DotDict


def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename


def cstr(s, color=None):
    if color is None:
        return s
    return colored(remove_color(s), color)


def ctab(data, *, colors=None, headers, **kw):
    if colors is None:
        colors = [None] * len(headers)

    if len(colors) != len(headers):
        raise ValueError(
            f"Number of colors ({len(colors)}) must match number of headers"
            f" ({len(headers)})."
            f"    colors = {colors}\n"
            f"    headers = {headers}"
        )
    for i in range(len(colors)):
        headers[i] = cstr(headers[i], colors[i])

    for i in range(len(data)):
        if len(data[i]) != len(colors):
            raise ValueError(
                f"Row {i} has {len(data[i])} columns, but there are"
                f" {len(colors)} colors.\n"
                f"    data[i] = {data[i]}\n"
                f"    colors = {colors}"
            )
        for j in range(len(data[i])):
            data[i][j] = cstr(data[i][j], colors[j])

    return tab(data, headers=headers, **kw)


def exec_imports_legacy(
    d: DotDict, *, root=None, delim='|', import_key='dimport'
):
    q = [('', d)]
    root = root or os.getcwd()
    while q:
        prefix, curr = q.pop(0)
        for k, v in curr.items():
            if isinstance(v, DotDict) or isinstance(v, dict):
                q.append((f'{prefix}.{k}' if prefix else k, v))
            elif isinstance(v, list) and len(v) > 0 and v[0] == import_key:
                lcl_root = os.path.join(root, *prefix.split('.'))
                d[f'{prefix}.{k}'] = cfg_import(
                    v[1], root=lcl_root, delim=delim
                )
    return d


def remove_color(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def save_metadata(*, path=None, cli=False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import os

            meta = func(*args, **kwargs)

            if cli:
                parser = argparse.ArgumentParser()
                parser.add_argument(
                    "--store_path", type=str, help="Path for storing metadata"
                )
                args = parser.parse_args()
                save_path = args.store_path
            else:
                save_path = (
                    path
                    if path is not None
                    else os.path.dirname(os.path.abspath(__file__))
                )
            print(
                f"save_metadata attempting create folder {save_path}...", end=""
            )
            os.makedirs(save_path, exist_ok=True)
            print("SUCCESS")
            full_path = os.path.join(save_path, "metadata.pydict")
            print(
                f"save_metadata attempting to save metadata to {full_path}...",
                end="",
            )
            with open(full_path, "w") as f:
                # f.write(prettify_dict(meta))
                f.write(str(meta))
            print("SUCCESS")

            return meta

        return wrapper

    return decorator


def sco(s, split=True):
    u = co(s, shell=True).decode("utf-8")
    if split:
        u = u.split("\n")[:-1]
    return u


def subdict(d, *, inc=None, exc=None):
    inc = list(d.keys()) if inc is None else inc
    exc = [] if exc is None else exc
    full_inc = set(inc).difference(exc)
    return {k: d[k] for k in full_inc}


def vco(s):
    """vanilla check output"""
    try:
        return co(s, shell=True).decode('utf-8').strip()
    except CalledProcessError as e:
        if "returned non-zero exit status 1" in str(e):
            return ""
        else:
            raise e
