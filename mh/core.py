import copy
import importlib
import inspect
import os
import random
import sys
import traceback
from functools import wraps
from typing import Any, Iterable, get_type_hints

import hydra
import yaml
from hydra import compose, initialize
from omegaconf import OmegaConf


def dict_dump(d, max_length=160):
    def sdict(input_dict):
        """
        Recursively convert all values in the dictionary to strings, ensuring proper serialization without
        including unwanted Python object references in the YAML output.
        """
        if isinstance(input_dict, dict) or isinstance(input_dict, DotDict):
            return {str(k): sdict(v) for k, v in input_dict.items()}
        elif isinstance(input_dict, list):
            return [sdict(element) for element in input_dict]
        else:
            if type(input_dict) in [int, float, bool]:
                return input_dict
            s = str(input_dict)
            if len(s) > max_length:
                s = s[:max_length] + '...'
            return s

    return yaml.dump(sdict(d))


def like_dict(obj: Any) -> bool:
    required_methods = ['keys', 'items', 'get']
    return all(hasattr(obj, method) for method in required_methods)


def like_list(obj: Any) -> bool:
    required_methods = [
        '__getitem__',
        '__setitem__',
        '__len__',
        'append',
        'extend',
        'pop',
    ]
    return all(hasattr(obj, method) for method in required_methods)


def gen_items(obj: Any) -> Iterable:
    if like_dict(obj):
        return obj.items()
    elif like_list(obj):
        return enumerate(obj)
    else:
        raise ValueError(f'Object {obj} is neither a list nor a dictionary')


class TraceError(Exception):
    def __init__(self, obj, msg, stack_trace=False):
        super().__init__(msg)
        self.msg = msg + '\n' + f'See {inspect.getfile(obj)} for more info'
        if stack_trace:
            self.msg += '\nStack trace below' + str(inspect.stack())

    def __str__(self):
        return self.msg


class TieredError(Exception):
    def __init__(self, passed_locals, msg, severity=0):
        super().__init__(msg)
        self.msg = msg + '\n' + f'locals in call: {passed_locals}\n'
        self.severity = severity

    def __str__(self):
        return self.msg


class DotDict:
    def __init__(self, d=None, self_ref_resolve=False, deep=False):
        if isinstance(d, DotDict):
            self.__init__(
                d.dict(), self_ref_resolve=self_ref_resolve, deep=deep
            )
            return
        if d is None:
            d = {}
        if deep:
            D = copy.deepcopy(d)
        else:
            D = d
        for k, v in D.items():
            if type(v) is dict:
                D[k] = DotDict(v, self_ref_resolve=False)
            elif type(v) is list:
                D[k] = [
                    (
                        DotDict(e, self_ref_resolve=False)
                        if type(e) is dict
                        else e
                    )
                    for e in v
                ]
        self.__dict__.update(D)
        if self_ref_resolve:
            self.self_ref_resolve()

    def set(self, k, v):
        self.deep_set(k, v)

    def get(self, k, default_val=None):
        try:
            return self.deep_get(k)
        except KeyError:
            return default_val

    def simple_dict(self):
        u = {}
        for k, v in self.items():
            if isinstance(v, DotDict):
                u[k] = v.__dict__
            elif isinstance(v, list):
                u[k] = [e.__dict__ if isinstance(e, DotDict) else e for e in v]
            else:
                u[k] = v
        return u

    def __setitem__(self, k, v):
        self.deep_set(k, v)

    def __getitem__(self, k):
        return self.deep_get(k)

    # def __getattr__(self, k):
    #     return self.deep_get(k)

    def __setattr__(self, k, v):
        if isinstance(v, dict):
            v = DotDict(v)
        self.__dict__[k] = v

    def __iter__(self):
        return iter(self.__dict__)

    def __delitem__(self, k):
        del self.__dict__[k]

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

    def update(self, d):
        self.__dict__.update(DotDict(d).dict())

    def str(self):
        # return str(self.__dict__)
        return self.pretty_str()
        # return str(self.dict())

    def dict(self):
        tmp = {}
        for k, v in self.items():
            if isinstance(v, DotDict):
                tmp[k] = v.dict()
            elif like_list(v):
                tmp[k] = [e.dict() if isinstance(e, DotDict) else e for e in v]
            else:
                tmp[k] = v
        return tmp

    def __str__(self):
        return self.str()

    def __repr__(self):
        return self.str()

    def deep_get(self, k):
        d = self.__dict__
        if type(k) != str:
            return d[k]
        keys = k.split('.')
        for i, key in enumerate(keys):
            d = d[key]
            # try:
            # except KeyError as e:
            #     partial_key = '.'.join(keys[:(i+1)])
            #     msg = (
            #         f'KeyError {e} found\n\n'
            #         f'Error getting {k}@{partial_key} from DotDict with keys below\n\n'
            #         f'{DotDict.indent_keys(self.flat_keys())}'
            #     )
            #     raise AttributeError(msg)
        return d

    def deep_set(self, k, v):
        d = self.__dict__
        if type(k) != str:
            return d[k]
        keys = k.split('.')
        for key in keys[:-1]:
            try:
                d = d[key]
            except KeyError:
                d[key] = DotDict({})
                d = d[key]
        d[keys[-1]] = v

    def has_self_ref(self, *, self_key='self'):
        d = self.__dict__
        q = [d]
        while q:
            d = q.pop()
            if like_dict(d):
                for k, v in d.items():
                    if like_dict(v) or like_list(v):
                        q.append(v)
                    elif isinstance(v, str):
                        if self_key in v or 'eval(' in v:
                            return True
            elif like_list(d):
                for e in d:
                    if like_dict(e) or like_list(e):
                        q.append(e)
                    elif isinstance(e, str):
                        if self_key in e or 'eval(' in e:
                            return True
        return False

    def pretty_str(self, max_length=160):
        return dict_dump(self, max_length=max_length)

    def self_ref_resolve(
        self, *, self_key='self', max_passes=5, gbl=None, lcl=None, relax=False
    ):
        lcl = lcl or {}
        gbl = gbl or {}
        lcl.update(locals())
        gbl.update(globals())
        passes = 0
        while passes < max_passes and self.has_self_ref(self_key=self_key):
            d = self.__dict__
            q = [d]
            while q:
                d = q.pop()
                for k, v in gen_items(d):
                    if isinstance(v, DotDict) or like_list(v):
                        q.append(v)
                    elif isinstance(v, dict):
                        d[k] = DotDict(v)
                        q.append(d[k])
                    elif isinstance(v, str):
                        try:
                            if 'eval(' in v:
                                d[k] = eval(
                                    v[5:-1].replace(self_key, 'self'), gbl, lcl
                                )
                            elif v.startswith(f'{self_key}.'):
                                # input(f'Attempting eval of {v=}')
                                d[k] = eval(
                                    v.replace(self_key, 'self'), gbl, lcl
                                )
                                # input('SUCCESSS')

                        except AttributeError:
                            msg = (
                                f"{self}Could not resolve self reference for"
                                f" {k}={v}, full self above"
                            )
                            subkeys = v.split('.')
                            debug_res = []
                            sub_msg = ''
                            for i in range(len(subkeys)):
                                subkey = '.'.join(subkeys[: i + 1])
                                try:
                                    tmp = eval(subkey, gbl, lcl)
                                    sub_msg += f'{subkey} -> {tmp}'
                                except Exception as e:
                                    sub_msg += f'{subkey} -> {e}'
                                sub_msg += '\n'
                            msg += (
                                "\nAttempted debug info with subkey resolution"
                                f" below\n{sub_msg}"
                            )
                            msg += (
                                "\n\nFull"
                                f" preview={DotDict.summarize_keys(self)}\n\n"
                            )

                            if not relax:
                                raise AttributeError(msg)
                            else:
                                UserWarning(msg)
                        except Exception as e:
                            msg = str(e)
                            final_msg = (
                                f'Error evaluating {v} of type {type(v)}\n{msg}'
                            )
                            raise RuntimeError(final_msg)

            passes += 1

        if passes == max_passes:
            msg = (
                f"Max passes ({max_passes}) reached. self_ref_resolve failed."
                " Debug info"
                " below."
                f"\nDirect key preview: {DotDict.summarize_keys(self)}\n"
                f"\n{self.get_self_ref_subdict(self_key=self_key)=}"
            )
            if not relax:
                raise ValueError(msg)
            else:
                further_msg = (
                    '. Continuing...set relax=False to raise error if '
                    ' this behavior is unexpected.'
                )
                UserWarning(f'{msg}...{further_msg}')
        return self

    def get_self_ref_keys(self, *, self_key='self'):
        d = self.__dict__
        q = [('root', d)]
        keys = []
        while q:
            base_key, d = q.pop()
            for k, v in gen_items(d):
                if isinstance(v, DotDict) or like_list(v):
                    q.append((f'{base_key}.{k}', v))
                elif isinstance(v, dict):
                    q.append((f'{base_key}.{k}', DotDict(v)))
                elif isinstance(v, str):
                    if self_key in v:
                        keys.append(f'{base_key}.{k}')
        return keys

    def get_self_ref_subdict(self, *, self_key='self'):
        def get_key(a, b):
            return f'{a}.{b}' if a else b

        q = [('', self)]
        final = {}
        while q:
            prefix, curr = q.pop(0)
            for k, v in gen_items(curr):
                K = get_key(prefix, k)
                if isinstance(v, DotDict) or like_list(v):
                    q.append((K, v))
                elif isinstance(v, dict):
                    q.append((K, DotDict(v)))
                elif isinstance(v, str):
                    if self_key in v:
                        final[K] = v

        eval_final = {}
        for k, v in final.items():
            try:
                final[k] = 'eval(' + v.replace(self_key, 'self') + ')'
                eval_final[k] = eval(final[k], globals(), locals())
            except Exception as e:
                eval_final[k] = e
        return final, eval_final

    def filter(self, exclude=None, include=None, relax=False):
        keys = set(self.keys())
        exclude = set() if exclude is None else set(exclude)
        include = keys if include is None else set(include)
        if not relax:
            if not include.issubset(keys):
                raise ValueError(
                    f"include={include} contains keys not in d={keys}"
                )
            if not exclude.issubset(keys):
                raise ValueError(
                    f"exclude={exclude} contains keys not in d={keys}...use"
                    " relax=True to ignore this error"
                )
            return DotDict({k: self[k] for k in include.difference(exclude)})
        else:
            include = include.intersection(keys)
            exclude = exclude.intersection(include)
            return DotDict(
                {k: self.get(k, None) for k in include.difference(exclude)}
            )

    def flat_keys(self, sort_by_depth=True):
        q = [(self, "")]  # Queue with (current_dict, current_prefix)
        keys = []

        while q:
            curr, prefix = q.pop(0)  # Dequeue from the front for BFS
            for k, v in curr.items():
                new_key = f"{prefix}.{k}" if prefix else k
                keys.append(new_key)
                if DotDict.is_like_dict(v):
                    q.append(
                        (v, new_key)
                    )  # Enqueue the nested dictionary with updated prefix
        if sort_by_depth:
            keys.sort(key=lambda x: (x.count('.'), x))

        return keys

    def validate_keys(self, keys: list[str]) -> list[str]:
        flattened_keys = self.flat_keys()
        invalid_keys = [k for k in keys if k not in flattened_keys]
        return invalid_keys

    def assert_keys_present(self, keys: list[str]) -> None:
        invalid_keys = self.validate_keys(keys)
        if invalid_keys:
            expected = DotDict.indent_keys(keys)
            invalid = DotDict.indent_keys(invalid_keys)
            flat_keys = DotDict.indent_keys(self.flat_keys())
            raise ValueError(
                f"Expected keys below:\n{expected}\n\nInvalid"
                f" keys:\n{invalid}\n\nFlat keys:\n{flat_keys}"
            )

    @staticmethod
    def summarize_keys(data, indent=0):
        """
        Recursively generate a YAML-style summary of keys and list indices,
        omitting large content and only displaying the structure.
        """
        lines = []
        indent_str = "  " * indent

        if isinstance(data, dict) or isinstance(data, DotDict):
            for key, value in data.items():
                # Print the key
                lines.append(f"{indent_str}{key}:")
                # If the value is a dictionary, summarize its keys recursively
                if isinstance(value, dict) or isinstance(value, DotDict):
                    lines.append(DotDict.summarize_keys(value, indent + 1))
                # If the value is a list, iterate each element
                elif isinstance(value, list):
                    for idx, item in enumerate(value):
                        # Only recurse into elements that are dict-like
                        if isinstance(item, dict) or isinstance(item, DotDict):
                            lines.append(f"{indent_str}  [{idx}]:")
                            lines.append(
                                DotDict.summarize_keys(item, indent + 2)
                            )
                        else:
                            # For non-dict list items, simply indicate their type
                            lines.append(
                                f"{indent_str}  [{idx}]:"
                                f" (\033[31m{type(item).__name__}\033[0m)"
                            )
                else:
                    # For non-dict and non-list values, indicate their type
                    if len(lines) > 0:
                        lines[-1] += f" (\033[31m{type(value).__name__}\033[0m)"

        elif isinstance(data, list):
            # If the top-level data is a list, iterate its elements.
            for idx, item in enumerate(data):
                lines.append(f"{indent_str}[{idx}]:")
                if (
                    isinstance(item, dict)
                    or isinstance(item, list)
                    or isinstance(item, DotDict)
                ):
                    lines.append(DotDict.summarize_keys(item, indent + 1))
                else:
                    lines.append(f"{indent_str}  ({type(item).__name__})")
        else:
            # If not a dict or list, simply show the type.
            lines.append(f"{indent_str}({type(data).__name__})")

        return "\n".join(lines)

    @staticmethod
    def indent_keys(tokens, level=1, indent_str='    ', sep='\n'):
        indenter = sep + indent_str * level
        return indenter + indenter.join(tokens)

    @staticmethod
    def is_like_dict(obj: Any) -> bool:
        return isinstance(obj, dict) or isinstance(obj, DotDict)


class DotDictImmutable(DotDict):
    def __init__(self, d=None, self_ref_resolve=False, deep=False):
        if isinstance(d, DotDict):
            self.__init__(
                d.dict(), self_ref_resolve=self_ref_resolve, deep=deep
            )
            return
        if d is None:
            d = {}
        if deep:
            D = copy.deepcopy(d)
        else:
            D = d
        for k, v in D.items():
            if type(v) is dict:
                D[k] = DotDictImmutable(v, self_ref_resolve=False)
            elif type(v) is list:
                D[k] = [
                    (
                        DotDictImmutable(e, self_ref_resolve=False)
                        if type(e) is dict
                        else e
                    )
                    for e in v
                ]
        self.__dict__.update(D)
        if self_ref_resolve:
            self.self_ref_resolve()

    def __reject__(self, *args, **kwargs):
        raise AttributeError(
            "DotDictImmutable is immutable. Use DotDict instead if you intend"
            " to modify the object."
        )

    def __setitem__(self, k, v):
        self.__reject__(k, v)

    def __setattr__(self, k, v):
        self.__reject__(self, k, v)

    def deep_set(self, k, v):
        self.__reject__(k, v)

    def update(self, d):
        self.__reject__(d)

    def self_ref_resolve(self, *args, **kwargs):
        self.__reject__(*args, **kwargs)

    def setdefault(self, k, v):
        self.__reject__(k, v)

    def __delitem__(self, k):
        self.__reject__(k)

    def __delattr__(self, k):
        self.__reject__(k)


def cfg_import(s, *, root=None, delim='|'):
    info = s.replace('^^', '').split(delim)
    root = root or os.getcwd()

    # ^^module
    if len(info) == 1:
        path, mod, func = root, info[0], None
    # ^^module|function
    elif len(info) == 2:
        path, mod, func = root, info[0], info[1]
    # ^^import_path|module|function
    else:
        path, mod, func = info

    if path.startswith('cwd'):
        tokens = path.split(os.sep)
        tokens[0] = os.getcwd()
        path = os.sep.join(tokens)

    # when function not specified, then we use a module import
    if func is not None and func.lower() in ['none', 'null', '']:
        func = None

    if path not in ['null', 'none', '']:
        path = os.path.abspath(path)

    return dyn_import(path=path, mod=mod, func=func)


def cfg_import_constrained(
    s,
    *,
    key_path,
    config_path,
    delim="^",
    namespace_key="namespace",
    verbose=False,
    ignore_numeric_keys=True,
):
    """
    Imports a constrained module attribute based on a key path and reference string.

    YAML example:
      a:
         b:
            c:
               d:
                  __call__: ^key

    Navigation rules:
      - '^key' or '^.key' (one dot) use the current module directory.
      - '^..key' uses one level upward.
      - '^...key' uses two levels upward, and so on.
    """
    if not s.startswith(delim):
        raise ValueError(
            f"s={s} does not conform to delim={delim} for"
            " cfg_import_constrained"
        )
    ref = s[len(delim) :].strip()
    dot_count = 0
    while ref.startswith("."):
        dot_count += 1
        ref = ref[1:]
    up_levels = dot_count - 1 if dot_count > 0 else 0

    # must now start with an letter
    if not ref[0].isalpha():
        raise TieredError(
            locals(),
            f'Invalid ref={ref} full string={s}...possible overload of'
            ' constrained and unconstrained import keys',
            severity=0,
        )

    key_parts = key_path.split(".")
    if len(key_parts) < up_levels + 1:
        raise ValueError(
            "Up-level count exceeds available key parts in key_path"
        )

    def apply_special_cases(arr):
        call_case = lambda x: x if not x.startswith("__call") else "__call__"
        special_cases = [call_case]
        for case in special_cases:
            arr = [case(e) for e in arr]
        if ignore_numeric_keys:
            arr = [e for e in arr if not e.isnumeric()]
        return arr

    base_parts = key_parts[: len(key_parts) - 1 - up_levels]
    attr_chain = key_parts[len(base_parts) :]

    base_parts = apply_special_cases(base_parts)
    attr_chain = apply_special_cases(attr_chain)

    namespace_dir = os.path.join(config_path, *base_parts, namespace_key)
    init_path = os.path.join(namespace_dir, "__init__.py")
    spec = importlib.util.spec_from_file_location(namespace_key, init_path)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load module '{namespace_key}' from {init_path}"
        )

    ns = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ns)

    obj = ns
    for attr in attr_chain:
        if not hasattr(obj, attr):
            raise TraceError(
                obj,
                f"Module '{namespace_key}' in '{namespace_dir}' lacks attribute"
                f" '{attr}'",
            )
        obj = getattr(obj, attr)
    if not hasattr(obj, "get"):
        raise TraceError(
            obj, f"Final object '{attr_chain[-1]}' requires a 'get' method."
        )
    return obj.get(ref)


def clean_kwargs(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        valid_keys = set(inspect.signature(func).parameters.keys())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        return func(*args, **filtered_kwargs)

    return wrapper


def colorize_yaml(lines, depth_color_map, value_color, max_length=160):
    from rich.text import Text

    colored_lines = Text()
    for line in lines:
        if len(line) > max_length:
            line = line[:max_length] + '...'
        # Detect the indentation level
        stripped_line = line.lstrip()
        indent_level = (len(line) - len(stripped_line)) // 2

        # Determine the color based on the indentation level
        color = depth_color_map.get(indent_level, 'white')

        # Apply the color to the line without changing the indentation
        if ':' in stripped_line:  # Key-value pair
            key, value = stripped_line.split(':', 1)
            colored_lines.append(
                ' ' * (indent_level * 2)
            )  # Preserve indentation
            colored_lines.append(key + ': ', style=color)
            colored_lines.append(value.strip(), style=value_color)
        elif stripped_line.startswith('-'):  # List item
            colored_lines.append(
                ' ' * (indent_level * 2)
            )  # Preserve indentation
            colored_lines.append(stripped_line, style=value_color)
        else:  # Handle keys without a value
            colored_lines.append(
                ' ' * (indent_level * 2)
            )  # Preserve indentation
            colored_lines.append(stripped_line, style=color)
        colored_lines.append('\n')
    return colored_lines


def convert_dictconfig(obj, self_ref_resolve=False, mutable=True):
    d = OmegaConf.to_container(obj, resolve=True)
    if mutable:
        return DotDict(d, self_ref_resolve=self_ref_resolve)
    else:
        return DotDictImmutable(d, self_ref_resolve=self_ref_resolve)


def depandify(data_frame):
    df_shape = [len(data_frame.index)] + [
        len(e) for e in data_frame.columns.levels
    ]
    data = data_frame.values.reshape(df_shape)
    column_names = data_frame.columns.names
    return data, column_names


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


def dyn_import(*, path, mod, func=None):
    path = path or 'null'
    if '.' in mod or path.lower()[-4:] in ['null', 'none']:
        obj = importlib.import_module(mod)
    else:
        if not path.startswith('/'):
            path = os.path.join(os.getcwd(), path)
        if not path.endswith('.py'):
            path = os.path.join(path, f'{mod}.py')

        spec = importlib.util.spec_from_file_location(mod, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod] = module
        spec.loader.exec_module(module)
        obj = module
    if func is not None:
        obj = getattr(obj, func)
    return obj


def easy_cfg(
    config_path: str = 'cfg', config_name: str = 'cfg.yaml'
) -> DotDict:
    cfg = OmegaConf.load(os.path.join(config_path, config_name))
    return OmegaConf.to_container(cfg, resolve=True)


def exec_imports(
    d: DotDict, *, root=None, delim='|', import_key='^^', ignore_spaces=True
):
    q = [('', d)]
    root = os.getcwd() if root is None else root
    while q:
        prefix, curr = q.pop(0)
        for k, v in curr.items():
            if isinstance(v, DotDict) or isinstance(v, dict):
                q.append((f'{prefix}.{k}' if prefix else k, v))
            elif isinstance(v, str):
                if ignore_spaces:
                    v = v.replace(' ', '').replace('\t', '')
                if v.startswith(import_key) and delim in v:
                    lcl_root = os.path.join(root, *prefix.split('.'))
                    full_key = f'{prefix}.{k}' if prefix else k
                    try:
                        d[full_key] = cfg_import(
                            v[len(import_key) :], root=lcl_root, delim=delim
                        )
                    except Exception as e:
                        msg = f'Error importing {v} at {full_key}\n{e}'
                        raise ImportError(msg)

    return d


def get_hydra_config_path_name():
    cfg = hydra.core.hydra_config.HydraConfig.get()
    config_name = cfg.job.config_name
    config_path = [
        path["path"]
        for path in cfg.runtime.config_sources
        if path["schema"] == "file"
    ][0]
    return config_path, config_name


def exec_imports_constrained(
    d: DotDict,
    *,
    config_path=None,
    delim='^',
    namespace_key='namespace',
    ignore_spaces=True,
    verbose=False,
):
    if config_path is None:
        config_path, _ = get_hydra_config_path_name()

    q = [('', d)]

    while q:
        prefix, curr = q.pop(0)
        for k, v in curr.items():
            if isinstance(v, (DotDict, dict)):
                q.append((f'{prefix}.{k}' if prefix else k, v))
            elif isinstance(v, str):
                if ignore_spaces:
                    v = v.replace(' ', '').replace('\t', '')
                if v.startswith(delim) and delim in v:
                    full_key = f'{prefix}.{k}' if prefix else k
                    try:
                        d[full_key] = cfg_import_constrained(
                            v,
                            key_path=full_key,
                            config_path=config_path,
                            delim=delim,
                            namespace_key=namespace_key,
                        )
                    except TieredError as e:
                        if e.severity == 0:
                            if verbose:
                                print(f'Warning: {e}', file=sys.stderr)
                        else:
                            raise e
                    except Exception as e:
                        msg = (
                            'Unexpected error during constrained importing'
                            f' {v} at {full_key}\n{e}'
                        )
                        raise ImportError(msg) from e
    return d


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


def get_print(*, _verbose):
    def print_fn(*args, verbose=1, **kw):
        if verbose <= _verbose:
            kw["flush"] = kw.get("flush", True)
            print(*args, **kw)

    return print_fn, print_fn


def hydra_kw(*, use_cfg=False, protect_kw=True, transform_cfg=None):
    if not use_cfg and transform_cfg:
        UserWarning(
            'use_cfg is False with non-null transform_cfg -> transform_cfg will'
            ' be ignored'
        )

    def decorator(f):
        @wraps(f)
        def wrapper(
            *args,
            config_path=None,
            config_name=None,
            version_base=None,
            overrides=None,
            return_hydra_config=False,
            **kw,
        ):
            if config_path is None or config_name is None:
                cfg = {}
            else:
                config_path = os.path.relpath(
                    config_path, os.path.dirname(__file__)
                )
                with initialize(
                    config_path=config_path, version_base=version_base
                ) as cfg:
                    cfg = compose(
                        config_name=config_name,
                        overrides=overrides,
                        return_hydra_config=return_hydra_config,
                    )

            overlapping_keys = set(cfg.keys()).intersection(set(kw.keys()))
            for key in overlapping_keys:
                kw[key] = cfg[key]
                if protect_kw:
                    del cfg[key]
            if use_cfg:
                if transform_cfg is not None:
                    cfg = transform_cfg(cfg)
                return f(cfg, *args, **kw)
            else:
                return f(*args, **kw)

        return wrapper

    return decorator


def hydra_out(name: str = '') -> str:
    out = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    s = os.path.join(out, name)
    base_folder = os.path.dirname(s)
    os.makedirs(base_folder, exist_ok=True)
    return s


def pandify(data, column_names):
    import pandas as pd

    if len(data.shape) < 2:
        raise ValueError("Data should have at least 2 dimensions.")

    if len(column_names) != len(data.shape) - 1:
        raise ValueError(
            f"column_names should have {len(data.shape) - 1} names, got"
            f" {len(column_names)}\n"
            f"{data.shape=}"
            f"{column_names=}"
        )

    columns = pd.MultiIndex.from_product(
        [range(dim_size) for dim_size in data.shape[1:]], names=column_names
    )

    data_frame = pd.DataFrame(data.reshape(data.shape[0], -1), columns=columns)

    return data_frame


def rand_slices(*shape, none_dims=None, N=1):
    """
    Generate random slices for specified dimensions of a given shape.

    Parameters:
    - shape: The shape of the array(s) as a tuple.
    - none_dims: A list of dimensions to keep unchanged (zero-based indexing, supports negative values).
    - N: The number of random elements to select.

    Returns:
    - A list of tuples, each tuple containing slice objects or integers for indexing into an array.
    """
    none_dims = none_dims or []
    # Normalize negative indices in none_dims
    none_dims = [d % len(shape) for d in none_dims]
    # Initialize a list to hold the generated slices
    generated_slices = []

    for _ in range(N):
        # Start with a list of slice(None) for each dimension
        current_slices = [slice(None)] * len(shape)

        # For dimensions not in none_dims, select a random index
        for dim in range(len(shape)):
            if dim not in none_dims:
                random_index = random.randint(0, shape[dim] - 1)
                current_slices[dim] = random_index

        generated_slices.append(tuple(current_slices))

    return generated_slices


def rich_tensor(
    tensor, *, name='Tensor', filename=None, max_width=None, strip=True, sep=','
):
    import torch

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


def torch_stats(report=None, black_pipable=True):
    """
    Calculate statistics of a torch tensor.

    Args:
        report (list or str, optional): Specifies which statistics to include in the report.
            If not provided, only the shape of the tensor will be included.
            Defaults to None.
            Available options:
                - 'shape': Shape of the tensor
                - 'device': Device of the tensor
                - 'dtype': Data type of the tensor
                - 'mean': Mean value of the tensor
                - 'variance': Variance of the tensor
                - 'median': Median value of the tensor
                - 'min': Minimum value of the tensor
                - 'max': Maximum value of the tensor
                - 'stddev': Standard deviation of the tensor
                - 'RMS': Root mean square of the tensor
                - 'L2': L2 norm of the tensor

        black_formattable (bool, optional): Specifies whether to format the output using black.
            If True, the output will be formatted using black_str function.
            If False, the output will be a plain string.
            Defaults to True.

    Returns:
        function: A helper function that calculates the requested statistics of a torch tensor.

    Raises:
        ModuleNotFoundError: If the torch module is not found, it raises an error with installation instructions.

    Example:
        >>> import torch
        >>> stats = torch_stats(['shape', 'mean', 'min'])
        >>> tensor = torch.tensor([1, 2, 3, 4, 5])
        >>> print(stats(tensor))
        shape: torch.Size([5])
        mean: 3.0
        min: 1
    """

    try:
        import torch

        all = [
            'shape',
            'dtype',
            'mean',
            'variance',
            'median',
            'min',
            'max',
            'stddev',
            'RMS',
            'L2',
        ]
        if not report:
            report = ['shape', 'device', 'dtype']
        if report in ['all', ['all']]:
            report = all
        report = set(report)

        get = lambda y: str(y) if black_pipable else y

        def helper(x):
            y = x.float()
            stats = {}
            if 'shape' in report:
                stats['shape'] = get(x.shape)
            if 'device' in report:
                stats['device'] = get(x.device)
            if 'dtype' in report:
                stats['dtype'] = get(x.dtype)
            if 'mean' in report:
                stats['mean'] = torch.mean(y).item()
            if 'variance' in report:
                stats['variance'] = torch.var(y).item()
            if 'median' in report:
                stats['median'] = torch.median(y).item()
            if 'min' in report:
                stats['min'] = torch.min(y).item()
            if 'max' in report:
                stats['max'] = torch.max(y).item()
            if 'stddev' in report:
                stats['stddev'] = torch.std(y).item()
            if 'RMS' in report:
                stats['RMS'] = torch.sqrt(torch.mean(y**2)).item()
            if 'L2' in report:
                stats['L2'] = torch.norm(y).item()
            # return str(stats)
            return yaml.dump(stats)

    except ModuleNotFoundError as e:
        msg = f'{e}\nIn order to use torch_stats, you need to install torch'
        msg = f'{msg} with "pip install torch"'
        raise ModuleNotFoundError(msg)

    return helper


def black_str(d: DotDict):
    try:
        import black

        def stringify(curr):
            for k, v in curr.items():
                if isinstance(v, DotDict) or isinstance(v, dict):
                    s = stringify(v)
                else:
                    s = str(v)
                if type(s) == str and s.startswith('<') and s.endswith('>'):
                    curr[k] = f'"{s}"'
            return curr

        s = black.format_str(str(stringify(d)), mode=black.FileMode())
        return s
    except ModuleNotFoundError as e:
        msg = (
            f'{e}\nIn order to use black_str, you need to install black'
            ' formatter'
        )
        msg = f'{msg} with "pip install black"'
        raise ModuleNotFoundError(msg)


def set_print_options(
    *,
    precision=None,
    threshold=None,
    edgeitems=None,
    linewidth=None,
    profile=None,
    sci_mode=None,
    callback=None,
):
    try:
        import torch

        torch.set_printoptions(
            precision=precision,
            threshold=threshold,
            edgeitems=edgeitems,
            linewidth=linewidth,
            profile=profile,
            sci_mode=sci_mode,
            callback=callback,
        )
    except TypeError:
        print(
            'Ignoring callback and using standard torch print options.\n'
            'Modify your local pytorch distribution according to the following'
            ' link to use callback.\n'
            '    https://github.com/tmasthay/Experiments/tree/main/custom_torch_print'
        )
        torch.set_printoptions(
            precision=precision,
            threshold=threshold,
            edgeitems=edgeitems,
            linewidth=linewidth,
            profile=profile,
            sci_mode=sci_mode,
        )


def yamlfy(c: DotDict, lcls, gbls) -> str:
    try:
        import yaml

        u = eval(black_str(c), lcls, gbls)

        def helper(d):
            for k, v in d.items():
                if isinstance(v, dict):
                    d[k] = helper(v)
                else:
                    d[k] = str(v)
            return d

        u = helper(u)
        return yaml.dump(u)
    except ModuleNotFoundError as e:
        msg = f'{e}\nIn order to use yamlfy, you need to install pyyaml'
        msg = f'{msg} with "pip install pyyaml"'
        raise ModuleNotFoundError(msg)


def draise(*args, sep='\n'):
    s = sep
    for x in args:
        s += str(x) + sep
    raise ValueError(s)


def enforce_types(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        type_hints = get_type_hints(func)

        for arg_name, value in bound_args.arguments.items():
            expected_type = type_hints.get(arg_name)
            if expected_type and not isinstance(value, expected_type):
                raise ValueError(
                    f"{arg_name} must be of type {expected_type}, got"
                    f" {type(value)}"
                )

        return func(*args, **kwargs)

    return wrapper


class StaticClass(type):
    def __new__(cls, name, bases, namespace):
        for k, v in namespace.items():
            if callable(v) and not k.startswith("__"):
                namespace[k] = staticmethod(v)
        return super().__new__(cls, name, bases, namespace)


class AutoStatic(metaclass=StaticClass):
    _allowed_special = {"__call__", "__init__"}

    @classmethod
    def _is_valid_attr(cls, name: str) -> bool:
        return (not name.startswith("_")) or (name in cls._allowed_special)

    @classmethod
    def get(cls, key: str):
        available = {
            name: getattr(cls, name)
            for name in dir(cls)
            if cls._is_valid_attr(name)
        }
        if key in available:
            return available[key]
        elif key.startswith("_"):
            raise TraceError(
                cls,
                f"Attribute '{key}' is invalid. Only the following special"
                f" attributes are allowed: {sorted(cls._allowed_special)}."
                " Ensure you have not misspelled a special method.",
            )
        else:
            raise TraceError(
                cls, f"Attribute '{key}' not found in {cls.__name__}."
            )


def build_local_getters(ns, omit=None):
    if omit is None:
        omit = []
    return {
        name: getattr(obj, 'get')
        for name, obj in ns.items()
        if name not in omit and isinstance(obj, type) and hasattr(obj, 'get')
    }


def build_module_getter_callback(ns, omit=None):
    getters = build_local_getters(ns, omit=omit)

    def helper(key):
        if key in getters:
            return getters[key]
        else:
            raise TraceError(ns, f"Key '{key}' not found in {ns.__name__}.")

    return helper


class Tee:
    def __init__(self, outputs=None, mode="w"):
        """
        Initialize a Tee context manager.

        :param outputs: A list (or a single string/file handle) of targets to duplicate output.
                        If None, then no duplication occurs.
        :param mode: Mode to open files passed as strings.
        """
        # Normalize outputs: if None, we use an empty list (i.e. no duplication).
        if outputs is None:
            self.outputs = []
        else:
            # If a single string or bytes is given, make it a list.
            if isinstance(outputs, (str, bytes)):
                outputs = [outputs]
            # For each target in the list, if it's a string, open the file; otherwise assume it’s a file-like object.
            self.outputs = [
                (
                    open(target, mode)
                    if isinstance(target, (str, bytes))
                    else target
                )
                for target in outputs
            ]
        # Save the original stdout and stderr.
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def write(self, stream, data):
        # Write to the original stream.
        stream.write(data)
        # Write to each target.
        for out in self.outputs:
            out.write(data)

    def flush(self, stream):
        stream.flush()
        for out in self.outputs:
            out.flush()

    class TeeStream:
        """
        A helper class that wraps an original stream (stdout or stderr) and duplicates writes to
        all configured outputs via the parent Tee instance.
        """

        def __init__(self, parent, stream):
            self.parent = parent
            self.stream = stream

        def write(self, data):
            self.parent.write(self.stream, data)

        def flush(self):
            self.parent.flush(self.stream)

        def __getattr__(self, attr):
            # Delegate attribute access to the underlying stream.
            return getattr(self.stream, attr)

    def __enter__(self):
        # If there are no outputs to duplicate to, do nothing.
        if not self.outputs:
            return self
        # Save original streams.
        self._saved_stdout = sys.stdout
        self._saved_stderr = sys.stderr
        # Replace sys.stdout and sys.stderr with TeeStream instances.
        sys.stdout = Tee.TeeStream(self, self.stdout)
        sys.stderr = Tee.TeeStream(self, self.stderr)
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        # Restore original streams if we replaced them.
        if self.outputs:
            sys.stdout = self._saved_stdout
            sys.stderr = self._saved_stderr
            # Close any targets we opened (but not if they are the original streams).
            for out in self.outputs:
                if out not in (self.stdout, self.stderr):
                    out.close()

    @staticmethod
    def tee_output(outputs, mode="w"):
        """
        A decorator factory that wraps a function so that its output is duplicated
        to the specified outputs while it runs.

        :param outputs: List of targets (strings or file handles). If None, no tee occurs.
        :param mode: Mode used if any target is a string (filename).
        """

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if outputs is None:
                    # No duplication; run normally.
                    return func(*args, **kwargs)
                with Tee(outputs, mode):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    @staticmethod
    def hydra_tee(func):
        """
        A decorator for Hydra-style functions with signature func(cfg, ...).
        It looks for cfg.dupe. If cfg.dupe is truthy, it wraps the function call within a Tee context
        that duplicates output to the specified target(s). If cfg.dupe is None/false, it runs normally.
        """

        @wraps(func)
        def wrapper(cfg, *args, **kwargs):
            dupe = getattr(cfg, "dupe", None)
            if not dupe:
                return func(cfg, *args, **kwargs)
            # Normalize dupe into a list if it's a string/file handle.
            outputs = dupe
            if isinstance(outputs, (str, bytes)):
                outputs = [outputs]
            for i, e in enumerate(outputs):
                if isinstance(e, str) and not e.startswith('/'):
                    outputs[i] = hydra_out(e)

            with Tee(outputs):

                def exit_msg():
                    files = '\n    '.join(
                        [e for e in outputs if isinstance(e, str)]
                    )
                    return f'Output for {str(func)} written to\n    {files}'

                try:
                    res = func(cfg, *args, **kwargs)
                    print(exit_msg())
                    return res
                except Exception as e:
                    traceback.print_exc()
                    print(exit_msg())
                    raise e

        return wrapper
