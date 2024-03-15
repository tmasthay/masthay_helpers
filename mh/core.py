import copy
import importlib
import inspect
import os
import sys
import hydra
from omegaconf import OmegaConf
from functools import wraps
import random


class DotDict:
    def __init__(self, d=None, self_ref_resolve=False, deep=False):
        if d is None:
            d = {}
        if deep:
            D = copy.deepcopy(d)
        else:
            D = d
        if type(d) is DotDict:
            self.__dict__.update(d.__dict__)
        else:
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

    def __setitem__(self, k, v):
        self.deep_set(k, v)

    def __getitem__(self, k):
        return self.deep_get(k)

    def __setattr__(self, k, v):
        if isinstance(v, dict):
            v = DotDict(v)
        self.__dict__[k] = v

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
        self.__dict__.update(DotDict.get_dict(d))

    def str(self):
        return str(self.__dict__)

    def dict(self):
        return self.__dict__

    def __str__(self):
        return self.str()

    def __repr__(self):
        return self.str()

    def deep_get(self, k):
        d = self.__dict__
        if type(k) != str:
            return d[k]
        keys = k.split('.')
        for key in keys:
            d = d[key]
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

    def has_self_ref(self):
        d = self.__dict__
        q = [d]
        while q:
            d = q.pop()
            for k, v in d.items():
                if isinstance(v, DotDict):
                    q.append(v)
                elif isinstance(v, dict):
                    q.append(v)
                elif isinstance(v, str):
                    if 'self' in v or 'eval(' in v:
                        return True
        return False

    def self_ref_resolve(self, max_passes=5, gbl=None, lcl=None, relax=False):
        lcl = lcl or {}
        gbl = gbl or {}
        lcl.update(locals())
        gbl.update(globals())
        passes = 0
        while passes < max_passes and self.has_self_ref():
            d = self.__dict__
            q = [d]
            while q:
                d = q.pop()
                for k, v in d.items():
                    if isinstance(v, DotDict):
                        q.append(v)
                    elif isinstance(v, dict):
                        d[k] = DotDict(v)
                        q.append(d[k])
                    elif isinstance(v, str):
                        try:
                            if 'eval(' in v:
                                d[k] = eval(v[5:-1], gbl, lcl)
                            elif 'self.' in v:
                                d[k] = eval(v, gbl, lcl)

                        except AttributeError:

                            msg = (
                                f"Could not resolve self reference for {k}={v}"
                                f"\ngiven below\n\n{self}"
                            )
                            if not relax:
                                raise AttributeError(msg)
                            else:
                                UserWarning(msg)
                        except TypeError as e:
                            msg = str(e)
                            final_msg = (
                                f'Error evaluating {v} of type {type(v)}'
                                f'\n{msg}'
                            )
                            raise RuntimeError(final_msg)
            passes += 1
        if passes == max_passes:
            msg = f"Max passes ({max_passes}) reached. self_ref_resolve failed."
            if not relax:
                raise ValueError(msg)
            else:
                further_msg = (
                    '. Continuing...set relax=False to raise error if '
                    ' this behavior is unexpected.'
                )
                UserWarning(f'{msg}...{further_msg}')
        return self

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


def cfg_import(s, *, root=None, delim='|'):
    info = s.split(delim)
    root = root or os.getcwd()
    if len(info) == 1:
        path, mod, func = root, info[0], None
    elif len(info) == 2:
        path, mod, func = root, info[0], info[1]
    else:
        path, mod, func = info
        if path.lower() == "cwd":
            path = os.getcwd()

    if func is not None and func.lower() in ['none', 'null', '']:
        func = None

    path = os.path.abspath(path)
    return dyn_import(path=path, mod=mod, func=func)


def clean_kwargs(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        valid_keys = set(inspect.signature(func).parameters.keys())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        return func(*args, **filtered_kwargs)

    return wrapper


def convert_dictconfig(obj, self_ref_resolve=False):
    return DotDict(
        OmegaConf.to_container(obj, resolve=True),
        self_ref_resolve=self_ref_resolve,
    )


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
    if '.' in mod:
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


def exec_imports(d: DotDict, *, root=None, delim='|', import_key='^^'):
    q = [('', d)]
    root = root or os.getcwd()
    while q:
        prefix, curr = q.pop(0)
        for k, v in curr.items():
            if isinstance(v, DotDict) or isinstance(v, dict):
                q.append((f'{prefix}.{k}' if prefix else k, v))
            elif isinstance(v, str) and v.startswith(import_key):
                lcl_root = os.path.join(root, *prefix.split('.'))
                full_key = f'{prefix}.{k}' if prefix else k
                d[full_key] = cfg_import(
                    v[len(import_key) :], root=lcl_root, delim=delim
                )
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
    return os.path.join(out, name)


def pandify(data, column_names):
    if len(data.shape) < 2:
        raise ValueError("Data should have at least 2 dimensions.")

    if len(column_names) != len(data.shape) - 1:
        raise ValueError(
            f"column_names should have {len(data.shape) - 1} names, got"
            f" {len(column_names)}"
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


def torch_stats(report=None, black_formattable=True):
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
            report = ['shape']
        if report in ['all', ['all']]:
            report = all
        report = set(report)

        def helper(x):
            stats = {}
            if 'shape' in report:
                stats['shape'] = x.shape
            if 'dtype' in report:
                stats['dtype'] = x.dtype
            if 'mean' in report:
                stats['mean'] = torch.mean(x).item()
            if 'variance' in report:
                stats['variance'] = torch.var(x).item()
            if 'median' in report:
                stats['median'] = torch.median(x).item()
            if 'min' in report:
                stats['min'] = torch.min(x).item()
            if 'max' in report:
                stats['max'] = torch.max(x).item()
            if 'stddev' in report:
                stats['stddev'] = torch.std(x).item()
            if 'RMS' in report:
                stats['RMS'] = torch.sqrt(torch.mean(x**2)).item()
            if 'L2' in report:
                stats['L2'] = torch.norm(x).item()
            if black_formattable:
                s = black_str(stats)
            else:
                s = ''
                for k, v in stats.items():
                    s += f'{k}: {v}\n'
            return s

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
        msg = f'{e}\nIn order to use black_str, you need to install black formatter'
        msg = f'{msg} with "pip install black"'
        raise ModuleNotFoundError(msg)


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
