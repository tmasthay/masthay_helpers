import argparse
import inspect
import os
import re
import subprocess
import sys
import textwrap
from fnmatch import fnmatch
from functools import wraps
from itertools import product
from subprocess import check_output as co

import black
import numpy as np
import pandas as pd
import torch
from returns.curry import curry, partial
from tabulate import tabulate as tab
from termcolor import colored
from rich.table import Table
from rich.console import Console
from rich.text import Text
from rich_tools import df_to_table


class GlobalHelpers:
    delimiter_choices = ["", "@", "#", "%", "^", "&", "*"]
    for a, b, c, d in product(delimiter_choices, repeat=4):
        u = a + b + c + d
        if u not in delimiter_choices:
            delimiter_choices.append(a + b + c + d)
    delimiter_choices = [e for e in delimiter_choices if e != ""]
    delimiter_choices.sort(key=len)

    @staticmethod
    def get_delimiter(s):
        for d in GlobalHelpers.delimiter_choices:
            if d not in s:
                return d
        return None


# global helpers tyler
def sco(s, split=True):
    u = co(s, shell=True).decode("utf-8")
    if split:
        u = u.split("\n")[:-1]
    return u


def vco(s):
    """'vanilla check output'"""
    return co(s, shell=True).decode('utf-8').strip()


def conda_include_everything():
    inc_paths = ":".join(sco("find $CONDA_PREFIX/include -type d"))
    c_path = os.environ.get("C_INCLUDE_PATH")
    cmd = "echo 'export C_INCLUDE_PATH=%s:%s'" % (c_path, inc_paths)
    cmd += " | pbcopy"
    os.system(cmd)


def get_dependencies():  # Run the grep command using subprocess
    grep_command = "grep -rE '^(import|from .* import)' --include='*.py' ."
    result = sco(grep_command)

    final = []
    for line in result:
        l = line.split(":")[1].split(" ")[1]
        if l not in final:
            final.append(l)
    return final


def format_with_black(
    code: str,
    line_length: int = 80,
    preview: bool = True,
    magic_trailing_comma: bool = False,
    string_normalization: bool = False,
) -> str:
    try:
        mode = black.FileMode(
            line_length=line_length,
            preview=preview,
            magic_trailing_comma=magic_trailing_comma,
            string_normalization=string_normalization,
        )
        formatted_code = black.format_str(code, mode=mode)
        return formatted_code
    except black.NothingChanged:
        return code


def function_str_dummy(s):
    s = s.replace("<locals>", "locals")
    s = re.sub(r"(<function .*? at .*?>)", r"'STRING_DUMMY: \1'", s)
    return s


def prettify_dict(
    d,
    line_length=80,
    preview=True,
    magic_trailing_comma=False,
    string_normalization=False,
):
    return format_with_black(
        function_str_dummy(str(d)),
        line_length=line_length,
        preview=preview,
        magic_trailing_comma=magic_trailing_comma,
        string_normalization=string_normalization,
    )


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
                f.write(prettify_dict(meta))
            print("SUCCESS")

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


def justify_lines(lines, *, demarcator="&", align="ljust", extra_space=0):
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
    if align.lower() not in ["ljust", "rjust"]:
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
        formatted_lines.append(" ".join(formatted_line))

    return "\n".join(formatted_lines)


def printj(lines, *, demarcator="&", align="ljust", extra_space=0, **kw):
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
            "Multiple names for variable with debug info below\n"
            f"    id={id(tmp)}\n"
            f"    possible_names={names}\n"
            f"    calling_context={calling_context}"
        )
    return names[0]


def get_var(var_name, calling_context):
    return calling_context.get(var_name, None)


def istr(*args, l=0, idt_str="    ", cpl=80):
    idt_level = l

    def dummy_endline(i, x):
        if "\n" not in x:
            return ""
        n = cpl - len(x) - idt_level * len(idt_str) - 1
        if i > 0:
            n -= len(idt_str)
        delimiter = GlobalHelpers.get_delimiter(x)
        extra = n * delimiter
        extra = extra[:n] + " "
        return extra

    args1 = []
    for e in args:
        u = e.split("\n")
        for i in range(len(u) - 1):
            u[i] = u[i] + "\n"
        args1.extend(u)

    args2 = []
    for e in args1:
        if len(e) < cpl:
            args2.append(e)
        else:
            tokens = e.split(" ")
            args2.append("")
            total = 0
            for t in tokens:
                if total + len(t) > cpl:
                    args2.append(t)
                    total = len(t)
                else:
                    args2[-1] += t
                    total += len(t)
                    if total < cpl:
                        args2[-1] += " "
                        total += 1
    delimiters = [dummy_endline(i, e) for i, e in enumerate(args2)]
    args = [e.replace("\n", "") + d for e, d in zip(args2, delimiters)]

    wrapper = textwrap.TextWrapper(
        width=cpl,
        replace_whitespace=False,
        initial_indent=idt_level * idt_str,
        subsequent_indent=(idt_level + 1) * idt_str,
    )

    if "comment" in args:
        input("yoyoyo")
        input(args)
    res = "\n".join(wrapper.wrap("".join(args)))

    for e in delimiters:
        res = res.replace(e.replace(" ", ""), "")

    return res


def iprints(*args, l=0, idt_str="    ", cpl=80, **kw):
    print(istr(*args, l=l, idt_str=idt_str, cpl=cpl), **kw)


def iprintl(*args, l=0, idt_str="    ", cpl=80, sep="\n", **kw):
    args = [istr(e, l=l, idt_str=idt_str, cpl=cpl) for e in args]
    print("\n".join(args), sep=sep, **kw)


def iprintt(*args, l=0, idt_str="    ", cpl=80, sep="\n", **kw):
    args1 = []
    for arg in args:
        if isinstance(arg, str):
            args1.append(istr(arg, l=l, idt_str=idt_str, cpl=cpl))
        elif isinstance(arg, tuple):
            if len(arg) != 2:
                raise ValueError(
                    "Tuple must have length 2. USAGE: (string,"
                    " local_indent_level)"
                )
            args1.append(istr(arg[0], l=l + arg[1], idt_str=idt_str, cpl=cpl))
        else:
            raise ValueError(
                "Arguments must be strings or tuples. USAGE: (string,"
                " local_indent_level). If only string is passed, local_indent"
                " level is assumed to be equal to kwarg l, which you have"
                f" passed l={l}."
            )
    print(*args1, sep=sep, **kw)


def iprint(*args, l=0, idt_str="    ", cpl=80, sep="\n", mode="auto", **kw):
    if mode == "auto":
        iprintt(*args, l=l, idt_str=idt_str, cpl=cpl, sep=sep, **kw)
    elif mode == "lines":
        iprintl(*args, l=l, idt_str=idt_str, cpl=cpl, sep=sep, **kw)
    elif mode == "string":
        iprints(*args, l=l, idt_str=idt_str, cpl=cpl, sep=sep, **kw)
    else:
        raise ValueError(
            f"Invalid mode {mode}. Choose from [auto, lines, string]"
        )


def iraise(error_type, *args, l=0, idt_str="    ", cpl=80):
    raise error_type(istr(*args, l=l, idt_str=idt_str, cpl=cpl))


def ireraise(e, *args, l=0, idt_str="    ", cpl=80, idt_further=True):
    msg = str(e) + "\n"
    exception_type = type(e)
    full = istr(msg, l=l, idt_str=idt_str, cpl=cpl)
    if idt_further:
        idt_level += 1
    full += (
        "\n"
        + cpl * "*"
        + istr(*args, idt_level=idt_level, idt_str=idt_str, cpl=cpl)
        + cpl * "*"
    )
    raise exception_type(full)


def nothing(*args, **kwargs):
    pass


def call_counter_obj(func):
    def wrapper(instance, *args, **kwargs):
        # Check if the counter attribute exists, if not, initialize
        if not hasattr(instance, "_call_counter"):
            instance._call_counter = {}

        # Increment the counter for the function
        instance._call_counter[func.__name__] = (
            instance._call_counter.get(func.__name__, 0) + 1
        )

        return (
            func(instance, *args, **kwargs),
            instance._call_counter[func.__name__],
        )

    return wrapper


def call_counter(verbose=0, postprocess=nothing, return_counter=False):
    def decorator(func):
        def wrapper(*args, **kwargs):
            wrapper.calls += 1
            res = func(*args, **kwargs)
            postprocess(
                return_val=res, calls=wrapper.calls, verbose=wrapper.verbose
            )
            if return_counter:
                return res, wrapper.calls
            else:
                return res

        wrapper.calls = 0
        wrapper.verbose = verbose
        return wrapper

    return decorator


def call_vars(*, update=None, **kwargs_dummy):
    if update is None:
        update = lambda *args, res, state, **kw: state

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            res = func(*args, **kwargs)
            wrapper.d = update(*args, res=res, state=wrapper.d, **kwargs)
            return res

        wrapper.d = kwargs_dummy
        return wrapper

    return decorator


class DotDict:
    def __init__(self, d):
        if type(d) is DotDict:
            self.__dict__.update(d.__dict__)
        else:
            self.__dict__.update(d)

    def set(self, k, v):
        self.__dict__[k] = v

    def get(self, k):
        return getattr(self, k)

    def __setitem__(self, k, v):
        self.set(k, v)

    def __getitem__(self, k):
        return self.get(k)

    def getd(self, k, v):
        return self.__dict__.get(k, v)

    def setdefault(self, k, v):
        self.__dict__.setdefault(k, v)

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def has(self, k):
        return hasattr(self, k)

    def has_all(self, *keys):
        return all([self.has(k) for k in keys])

    def has_all_type(self, *keys, lcl_type=None):
        return all(
            [self.has(k) and type(self.get(k)) is lcl_type for k in keys]
        )

    def update(self, d):
        self.__dict__.update(DotDict.get_dict(d))

    def str(self):
        return format_with_black('DotDict(\n' + str(self.__dict__) + '\n)')

    def dict(self):
        return self.__dict__

    def __str__(self):
        return self.str()

    def __repr__(self):
        return self.str()

    @staticmethod
    def get_dict(d):
        if isinstance(d, DotDict):
            return d.dict()
        else:
            return d


def peel_final(x):
    y = x.view(-1, x.shape[-1])
    shape = x.shape[:-1]

    def unravel(i):
        return np.unravel_index(i, shape)

    def ravel(*args):
        return np.ravel_multi_index(args, shape)

    return y, unravel, ravel


def get_print_deprecated(
    *,
    _verbose,
    l=0,
    idt_str="    ",
    cpl=80,
    sep="\n",
    mode="auto",
    demarcator="&",
    align="ljust",
    extra_space=0,
):
    def print_fn(*args, verbose=1, **kw):
        if verbose <= _verbose:
            kw["flush"] = kw.get("flush", True)
            iprint(
                *args, l=l, idt_str=idt_str, cpl=cpl, sep=sep, mode=mode, **kw
            )

    def print_col_fn(*args, verbose=1, **kw):
        if verbose <= _verbose:
            kw["flush"] = True
            printj(
                args,
                demarcator=demarcator,
                align=align,
                extra_space=extra_space,
                **kw,
            )
        return print

    return print_fn, print_col_fn


def get_print(*, _verbose):
    def print_fn(*args, verbose=1, **kw):
        if verbose <= _verbose:
            kw["flush"] = kw.get("flush", True)
            print(*args, **kw)

    return print_fn, print_fn


def summarize_tensor(tensor, *, idt_level=0, idt_str="    ", heading="Tensor"):
    stats = dict(dtype=tensor.dtype, shape=tensor.shape)
    if tensor.dtype == torch.bool:
        return str(stats)
    elif tensor.dtype in [
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ]:
        tensor = tensor.float()
    stats.update(
        {
            'mean': torch.mean(tensor).item(),
            'variance': torch.var(tensor).item(),
            'median': torch.median(tensor).item(),
            'min': torch.min(tensor).item(),
            'max': torch.max(tensor).item(),
            'stddev': torch.std(tensor).item(),
            'RMS': torch.sqrt(torch.mean(tensor**2)).item(),
            'L2': torch.norm(tensor).item(),
        }
    )

    # Prepare the summary string with the desired indentation
    indent = idt_str * idt_level
    summary = [f"{heading}:"]
    for key, value in stats.items():
        summary.append(f"{indent}{idt_str}{key} = {value}")

    return "\n".join(summary)


def print_tensor(tensor, print_fn=print, print_kwargs=None, **kwargs):
    if print_kwargs is None:
        print_kwargs = {"flush": True}
    print_fn(summarize_tensor(tensor, **kwargs), **print_kwargs)


def filter_kw(func, kwargs):
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def clean_kw(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        filtered_kwargs = filter_kw(func, kwargs)
        return func(*args, **filtered_kwargs)

    return wrapper


def pandify(data, column_names):
    # Check input dimensions
    if len(data.shape) < 2:
        raise ValueError("Data should have at least 2 dimensions.")

    if len(column_names) != len(data.shape) - 1:
        raise ValueError(
            f"column_names should have {len(data.shape) - 1} names, got"
            f" {len(column_names)}"
        )

    # Create a multi-level column index using the provided column names
    columns = pd.MultiIndex.from_product(
        [range(dim_size) for dim_size in data.shape[1:]], names=column_names
    )

    # Create and return a DataFrame
    data_frame = pd.DataFrame(data.reshape(data.shape[0], -1), columns=columns)

    return data_frame


def depandify(data_frame):
    df_shape = [len(data_frame.index)] + [
        len(e) for e in data_frame.columns.levels
    ]
    data = data_frame.values.reshape(df_shape)
    column_names = data_frame.columns.names
    return data, column_names


def get_full_slices(indices):
    return list(np.where(np.array(indices) == slice(None))[0])


def subdict(d, *, include=None, exclude=None):
    include = list(d.keys()) if include is None else include
    exclude = [] if exclude is None else exclude
    full_include = set(include).difference(exclude)
    return {k: d[k] for k in full_include}


def kw_builder(key_builder=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kw):
            if "key_builder" in kw:
                raise KeyError(
                    "The keyword argument 'key_builder' is reserved by the"
                    " decorator. Consider: (a) Omitting the decorator, (b)"
                    " Refactoring the function so that it has different keyword"
                    " arguments, or (c) Checking your passed keyword arguments."
                )

            kw = kw if key_builder is None else key_builder(kw)
            return func(*args, **kw)

        return wrapper

    return decorator


def dupe(stdout=True, stderr=True):
    """
    Duplicate stdout and stderr streams to /dev/null.
    """
    if stdout:
        dupe.original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
    if stderr:
        dupe.original_stderr = sys.stderr
        sys.stderr = open(os.devnull, "w")


def undupe(stdout=True, stderr=True):
    """
    Restore the original stdout and stderr streams if they were duplicated.
    """
    if stdout and hasattr(dupe, "original_stdout"):
        sys.stdout.close()
        sys.stdout = dupe.original_stdout
        delattr(dupe, "original_stdout")

    if stderr and hasattr(dupe, "original_stderr"):
        sys.stderr.close()
        sys.stderr = dupe.original_stderr
        delattr(dupe, "original_stderr")


def dynamic_expand(src, target_shape):
    # Determine the indices where the shape of `src` matches the shape of `target`
    common_shape = list(src.shape)
    expand_shape = [1] * len(target_shape)

    # Loop to place the true shape and put ones everywhere else
    j = len(common_shape) - 1
    for i in range(len(target_shape) - 1, -1, -1):
        if j >= 0:
            if common_shape[j] == target_shape[i]:
                expand_shape[i] = common_shape[j]
                j -= 1
            elif common_shape[j] == 1:
                j -= 1

    # Reshape src to be compatible for broadcasting
    src = src.view(*expand_shape)

    # Expand to match target_shape
    return src.expand(target_shape)


def kw_peeler(**peel):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kw):
            kw = {k: kw[k] for k in kw if k not in peel}
            return func(*args, **kw)

        return wrapper

    return decorator


def extend_dict(d, *, add=None, sub=None):
    add = {} if add is None else add
    sub = [] if sub is None else sub
    u = {**d, **add}
    return {k: v for k, v in u.items() if k not in sub}


def bstr(*args):
    return "".join(args)


def remove_color(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def cstr(s, color=None):
    if color is None:
        return s
    return colored(remove_color(s), color)


def cprint(s, color="red", **kw):
    print(cstr(s, color), **kw)


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


def install(**kwargs):
    traceback_defaults = {
        'show_locals': True,
        'width': 160,
        'locals_max_length': 20,
        'locals_max_string': 80,
    }
    kwargs = {**traceback_defaults, **kwargs}
    if os.environ.get('RICH_TRACEBACK', False):
        from rich.traceback import install

        install(**kwargs)


def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename


def ismath(x, *, ext=None):
    symbols = ['+', '-', 'e', '.', 'E', 'j']
    if ext is not None:
        symbols.extend(ext)

    for s in symbols:
        x = x.replace(s, '')
    return x.isnumeric()


def rich_tensor(
    tensor, *, name='Tensor', filename=None, max_width=None, strip=True, sep=','
):
    stats = dict(dtype=tensor.dtype, shape=tensor.shape)
    if tensor.dtype == torch.bool:
        return str(stats)
    elif tensor.dtype in [
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ]:
        tensor = tensor.float()
    stats.update(
        {
            'mean': torch.mean(tensor).item(),
            'variance': torch.var(tensor).item(),
            'median': torch.median(tensor).item(),
            'min': torch.min(tensor).item(),
            'max': torch.max(tensor).item(),
            'stddev': torch.std(tensor).item(),
            'RMS': torch.sqrt(torch.mean(tensor**2)).item(),
            'L2': torch.norm(tensor).item(),
        }
    )
    d1 = {}
    for k, v in stats.items():
        if type(v) == float:
            d1[k] = f'{v:.4f}'
        else:
            d1[k] = str(v)

    # Create DataFrame
    df = pd.DataFrame(stats)
    table = df_to_table(df)

    if filename is None:
        return table, df
    else:
        df_filename = filename + '.csv'
        table_filename = filename + '.txt'
        if filename == 'stdout':
            console = Console()
        else:
            console = Console(file=open(table_filename, 'w'), width=10000)
        console.print(table)
        csv_str = df.to_csv(sep=sep)
        if strip:
            csv_str = '\n'.join(
                [sep.join(e.split(sep)[1:]) for e in csv_str.split('\n')]
            )
        with open(df_filename, 'w') as f:
            f.write(csv_str)


def torch_dir_see(
    dir1=None,
    *,
    filename=None,
    include_only=None,
    make_figs=False,
    max_width=None,
    strip=True,
    sep=',',
):
    dir1 = dir1 if dir1 else os.getcwd()

    if dir1 is None:
        dir1 = os.getcwd()
    import matplotlib.pyplot as plt
    from masthay_helpers.typlotlib import plot_tensor2d_fast as plot

    u = [e.split('/')[-1] for e in os.listdir(dir1) if e.endswith('.pt')]
    if include_only is not None:
        include_only = [e.replace('.pt', '') + '.pt' for e in include_only]
        u = set(u).intersection(set(include_only))

    if u:
        print(f'Files in Directory: {dir1}')
        for e in u:
            a = torch.load(os.path.join(dir1, e))
            rich_tensor(
                a,
                name=e.replace('.pt', ''),
                filename=filename,
                max_width=max_width,
                strip=strip,
                sep=sep,
            )
            if make_figs:
                curr_path = os.path.join(dir1, 'figs', e.replace('.pt', ''))
                plot(
                    tensor=a,
                    labels=[e],
                    print_freq=25,
                    path=curr_path,
                    verbose=True,
                )


def torch_dir_compare(
    dir1, dir2, out_dir, *, include_only=None, make_figs=True
):
    import matplotlib.pyplot as plt

    from masthay_helpers.typlotlib import plot_tensor2d_fast as plot

    u1 = [e.split('/')[-1] for e in os.listdir(dir1) if e.endswith('.pt')]
    u2 = [e.split('/')[-1] for e in os.listdir(dir2) if e.endswith('.pt')]
    u3 = set(u1).intersection(set(u2))
    if include_only is not None:
        include_only = [e.replace('.pt', '') + '.pt' for e in include_only]
        u3 = set(u3).intersection(set(include_only))
    # raise ValueError('debug')
    if not u3:
        raise ValueError(
            f'No common files found between directories dir1={dir1},'
            f' dir2={dir2}'
        )

    @curry
    def config(title, *, min_val=None, max_val=None):
        plt.title(title)
        plt.set_cmap('seismic')
        plt.colorbar()
        if min_val is not None:
            plt.clim(vmin=min_val, vmax=max_val)

    for u in u3:
        a = torch.load(os.path.join(dir1, u))
        b = torch.load(os.path.join(dir2, u))
        path = f'{out_dir}/figs'
        os.makedirs(path, exist_ok=True)
        try:
            difference = a - b
            pt_file = os.path.join(out_dir, u)
            txt_file = os.path.join(out_dir, u.replace('.pt', '.txt'))
            torch.save(difference, pt_file)
            # summary = summarize_tensor(
            #     difference,
            #     heading=f'Difference between {dir1}/{u} and {dir2}/{u}',
            # )
            # with open(txt_file, 'w') as f:
            #     f.write(summary)
            rich_tensor(
                difference,
                name=f'Difference between versions of {u}',
                console=Console(file=open(txt_file, 'w'), width=10000),
            )

            def my_flatten(x):
                while len(x.shape) > 3:
                    new = x.shape[0] * x.shape[1]
                    x = x.view(new, *x.shape[2:])
                if len(x.shape) == 3:
                    x = x.permute(2, 1, 0)
                return x

            if make_figs:
                curr_path = os.path.join(path, u.replace('.pt', ''))
                name0 = dir1.split('/')[-1]
                name1 = dir2.split('/')[-1]
                if name0 == name1:
                    name0 = name0 + '0'
                    name1 = name1 + '1'
                plot(
                    tensor=my_flatten(difference),
                    labels=[f'Difference {u}'],
                    print_freq=25,
                    path=curr_path,
                    config=config(
                        min_val=difference.min(), max_val=difference.max()
                    ),
                    verbose=True,
                    name='difference',
                )
                plot(
                    tensor=my_flatten(a),
                    labels=[f'{dir1}/{u}'],
                    print_freq=25,
                    path=curr_path,
                    config=config(min_val=a.min(), max_val=a.max()),
                    verbose=True,
                    name=name0,
                )
                plot(
                    tensor=my_flatten(b),
                    labels=[f'{dir2}/{u}'],
                    print_freq=25,
                    path=curr_path,
                    config=config(min_val=b.min(), max_val=b.max()),
                    verbose=True,
                    name=name1,
                )
        except Exception as e:
            print(
                f'Could not compare {u}\n    {dir1} -> {a.shape}\n   '
                f' {dir2} -> {b.shape}\n    {str(e)}',
                flush=True,
            )

    c = '\n    '
    u4 = [e.replace('.pt', '.txt') for e in u3]
    u5 = [f'{e} -> {f}' for e, f in zip(u3, u4)]
    print(f'Files written to Directory: {out_dir}\n    {c.join(u5)}')


def torch_math(f, *args):
    grids = torch.meshgrid(*[torch.arange(len(arg)) for arg in args])
    values = [arg[grid] for arg, grid in zip(args, grids)]
    result = f(*values)
    return result


def flip_dict(d):
    try:
        u = {}
        for outer_key, inner_dict in d.items():
            for inner_key, value in inner_dict.items():
                u.setdefault(inner_key, {})
                u[inner_key][outer_key] = value
    except Exception as e:
        print(f'Could not flip dictionary due to error. Here is the dict:')
        # print(prettify_dict(d))
        raise e
    return u
