from subprocess import check_output as co
import os
import numpy as np
import subprocess
import re
from functools import wraps
import sys
import argparse
import textwrap
from itertools import product
import black


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


def conda_include_everything():
    inc_paths = ":".join(sco("find $CONDA_PREFIX/inclugetde -type d"))
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


# def prettify_dict(d, jsonify=True):
#     s = str(d)
#     s = re.sub(r"<function (\w+) at 0x[\da-f]+>", r"\1", s)
#     s = s.replace("{", "{\n")
#     s = s.replace("}", "\n}")
#     s = s.replace(", ", ",\n")
#     lines = s.split("\n")
#     idt = 4 * " "
#     idt_level = 0
#     for i, l in enumerate(lines):
#         if l in ["}", "},", ","]:
#             idt_level -= 1
#             if idt_level < 0:
#                 idt_level = 0
#         lines[i] = idt_level * idt + l
#         if l[-1] == "{":
#             idt_level += 1
#     res = "\n".join(lines)
#     if jsonify:
#         res = res.replace("'", '"')
#     return res


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


def prettify_dict(
    d,
    line_length=80,
    preview=True,
    magic_trailing_comma=False,
    string_normalization=False,
):
    return format_with_black(
        str(d),
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
    print(
        istr(*args, l=l, idt_str=idt_str, cpl=cpl),
        **kw,
    )


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
            args1.append(
                istr(
                    arg[0],
                    l=l + arg[1],
                    idt_str=idt_str,
                    cpl=cpl,
                )
            )
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


class DotDict:
    def __init__(self, d):
        for k, v in d.items():
            self.set(k, v)

    def set(self, k, v):
        self.__dict__[k] = v

    def get(self, k):
        return getattr(self, k)

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

    def str(self):
        return prettify_dict(self.__dict__)


def peel_final(x):
    y = x.view(-1, x.shape[-1])
    shape = x.shape[:-1]

    def unravel(i):
        return np.unravel_index(i, shape)

    def ravel(*args):
        return np.ravel_multi_index(args, shape)

    return y, unravel, ravel


def get_print(
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
            kw["flush"] = True
            iprint(
                *args,
                l=l,
                idt_str=idt_str,
                cpl=cpl,
                sep=sep,
                mode=mode,
                **kw,
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
