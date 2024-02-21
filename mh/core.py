import copy
import importlib
import inspect
import os
import sys
import hydra
from omegaconf import OmegaConf
from functools import wraps


class DotDict:
    def __init__(self, d, self_ref_resolve=False):
        D = copy.deepcopy(d)
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
        keys = k.split('.')
        for key in keys:
            d = d[key]
        return d

    def deep_set(self, k, v):
        d = self.__dict__
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

    def self_ref_resolve(self, max_passes=10, glb=None, lcl=None):
        lcl.update(locals())
        glb.update(globals())
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
                        if 'self' in v:
                            d[k] = eval(v, glb, lcl)
                        elif 'eval(' in v:
                            d[k] = eval(v[5:-1], glb, lcl)
            passes += 1
        if passes == max_passes:
            raise ValueError(
                f"Max passes ({max_passes}) reached. self_ref_resolve failed"
            )
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
        obj = getattr(module, func)
    return obj


def easy_cfg(
    config_path: str = 'cfg', config_name: str = 'cfg.yaml'
) -> DotDict:
    cfg = OmegaConf.load(os.path.join(config_path, config_name))
    return OmegaConf.to_container(cfg, resolve=True)


def exec_imports(d: DotDict, *, root=None, delim='|', import_key='dimport'):
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
