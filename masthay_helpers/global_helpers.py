from subprocess import check_output as co
import os
import numpy as np
import subprocess
import re
from functools import wraps
import sys
import argparse


# global helpers tyler
def sco(s, split=True):
    u = co(s, shell=True).decode('utf-8')
    if split:
        u = u.split('\n')[:-1]
    return u


def conda_include_everything():
    inc_paths = ':'.join(sco('find $CONDA_PREFIX/inclugetde -type d'))
    c_path = os.environ.get("C_INCLUDE_PATH")
    cmd = "echo 'export C_INCLUDE_PATH=%s:%s'" % (c_path, inc_paths)
    cmd += ' | pbcopy'
    os.system(cmd)


def get_dependencies():  # Run the grep command using subprocess
    grep_command = "grep -rE '^(import|from .* import)' --include='*.py' ."
    result = sco(grep_command)

    final = []
    for line in result:
        l = line.split(':')[1].split(' ')[1]
        if l not in final:
            final.append(l)
    return final


def prettify_dict(d, jsonify=True):
    s = str(d)
    s = re.sub(r'<function (\w+) at 0x[\da-f]+>', r'\1', s)
    s = s.replace('{', '{\n')
    s = s.replace('}', '\n}')
    s = s.replace(', ', ',\n')
    lines = s.split('\n')
    idt = 4 * ' '
    idt_level = 0
    for i, l in enumerate(lines):
        if l in ['}', '},', ',']:
            idt_level -= 1
            if idt_level < 0:
                idt_level = 0
        lines[i] = idt_level * idt + l
        if l[-1] == '{':
            idt_level += 1
    res = '\n'.join(lines)
    if jsonify:
        res = res.replace("'", '"')
    return res


def save_metadata(*, path=None, cli=False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import os

            meta = func(*args, **kwargs)

            if cli:
                parser = argparse.ArgumentParser()
                parser.add_argument(
                    '--store_path', type=str, help='Path for storing metadata'
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
                f'save_metadata attempting create folder {save_path}...', end=''
            )
            os.makedirs(save_path, exist_ok=True)
            print('SUCCESS')
            full_path = os.path.join(save_path, 'metadata.pydict')
            print(
                f'save_metadata attempting to save metadata to {full_path}...',
                end='',
            )
            with open(full_path, 'w') as f:
                f.write(prettify_dict(meta))
            print('SUCCESS')

            return meta

        return wrapper

    return decorator


def add_root_package_path(*, path, pkg):
    path_tokens = path.split(os.sep)
    global_root = os.sep.join(path_tokens[: path_tokens.index(pkg)])
    sys.path.append(global_root)


def path_up(path, n=1):
    if path[-1] == os.sep:
        path = path[:-1]
    path_tokens = path.split(os.sep)
    return os.sep.join(path_tokens[:-n])


def justify_lines(lines, *, demarcator='&', align='ljust', extra_space=0):
    # Step (1)
    l = [line.split(demarcator) for line in lines]

    # Step (2)
    l = [[ee.strip() for ee in e] for e in l]

    # Step (3)
    if not all(len(e) == len(l[0]) for e in l):
        raise ValueError(
            "All lines must have the same number of elements after splitting."
        )

    # Step (4)
    if align.lower() not in ['ljust', 'rjust']:
        raise ValueError("Invalid align argument. Must be 'ljust' or 'rjust'.")

    # Determine max lengths for each column
    max_lengths = [max(len(e[j]) for e in l) for j in range(len(l[0]))]

    formatted_lines = []
    for e in l:
        formatted_line = []
        for j, text in enumerate(e):
            max_length = max_lengths[j]
            formatted_text = getattr(text, align.lower())(
                max_length + extra_space
            )
            formatted_line.append(formatted_text)
        formatted_lines.append(' '.join(formatted_line))

    return "\n".join(formatted_lines)


def printj(lines, *, demarcator='&', align='ljust', extra_space=0, **kw):
    print(
        justify_lines(
            lines, demarcator=demarcator, align=align, extra_space=extra_space
        ),
        **kw,
    )


def var_name(tmp, calling_context):
    address = id(tmp)
    res = []
    for k, v in calling_context.items():
        if id(v) == address:
            res.append(k)
    return res


def var_unique_name(tmp, calling_context):
    names = var_name(tmp, calling_context)
    if len(names) > 1:
        raise ValueError(
            'Multiple names for variable with debug info below\n'
            f'    id={id(tmp)}\n'
            f'    possible_names={names}\n'
            f'    calling_context={calling_context}'
        )
    return names[0]


def get_var(var_name, calling_context):
    return calling_context.get(var_name, None)
