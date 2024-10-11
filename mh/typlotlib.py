import copy
import glob
import io
import os
import warnings
from copy import deepcopy
from itertools import product
from time import time
from typing import Any, Callable, Iterator, List, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from PIL import Image
from returns.curry import curry
from tabulate import tabulate as tab

# global pre_colors
pre_colors = list(mcolors.CSS4_COLORS.keys())
pre_colors_dict = mcolors.CSS4_COLORS


def bool_slice(
    *args,
    permute=None,
    none_dims=(),
    ctrl=None,
    strides=None,
    start=None,
    cut=None,
    verbose=False,
):
    """
    Generate boolean slices based on the given arguments.

    Args:
        *args: Variable length arguments representing the dimensions of the boolean slices.
        permute (list): A list specifying the order in which the dimensions should be permuted.
        none_dims (tuple): A tuple containing the indices of dimensions that should be treated as None.
        ctrl (function): A function that takes in the indices and arguments and returns a boolean value.
        strides (list): A list specifying the strides for each dimension.
        start (list): A list specifying the starting indices for each dimension.
        cut (list): A list specifying the cut values for each dimension.
        verbose (bool): A boolean value indicating whether to print verbose output.

    Yields:
        tuple: A tuple containing the boolean slice indices and the result of the control function.

    """
    permute = list(permute or range(len(args)))
    permute.reverse()

    # Logic is not correct here for strides, start, cut, etc. TODO: Fix
    strides = strides or [1 for _ in range(len(args))]
    start = start or [0 for _ in range(len(args))]
    cut = cut or [0 for _ in range(len(args))]
    tmp = list(args)
    for i in range(len(strides)):
        if i not in none_dims:
            tmp[i] = (tmp[i] - start[i] - cut[i]) // strides[i]

    args = list(args)
    none_dims = [i if i >= 0 else len(args) + i for i in none_dims]
    for i in range(len(args)):
        if i not in none_dims:
            args[i] = args[i] - cut[i]
    # Total number of combinations
    total_combinations = np.prod(
        [e for i, e in enumerate(tmp) if i not in none_dims]
    )

    # Initialize indices
    idx = [
        slice(None) if i in none_dims else start[i] for i in range(len(args))
    ]

    if ctrl is None:

        def ctrl_default(*args):
            return True

        ctrl = ctrl_default

    for combo in range(total_combinations):
        res = tuple([tuple(idx)]) + (ctrl(idx, args),)
        if verbose:
            print(f'bool_slice: {combo} -> {res}')
        yield res

        # Update indices
        for i in permute:
            if i in none_dims:
                continue
            idx[i] += strides[i]
            if idx[i] < args[i]:
                break
            idx[i] = start[i]


def clean_idx(idx, show_colons=True):
    """
    Cleans up the given index by converting it to a string representation.

    Args:
        idx (list): The index to clean up.
        show_colons (bool, optional): Whether to include colons in the string representation.
            Defaults to True.

    Returns:
        str: The cleaned up string representation of the index.
    """
    res = [str(e) if e != slice(None) else ':' for e in idx]
    if not show_colons:
        res = [e for e in res if e != ':']
    return f'({", ".join(res)})'


class PlotTypes:
    Index = Union[int, List[int]]
    Indices = Union[Iterator[Index], List[Index]]
    PlotHandler = Callable[[ArrayLike, Indices, Figure, List[Axes]], bool]


def colors_str(*, colors, normalize=True, ncols=5, tablefmt="plain", **kw):
    strings = []
    longest_key = max(len(k) for k in colors.keys())
    for name, hex_color in colors.items():
        if isinstance(hex_color, str):
            rgb = tuple(
                int(hex_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)
            )
        elif normalize:
            rgb = tuple(int(255 * v) for v in hex_color)
        else:
            rgb = (hex_color[0], hex_color[1], hex_color[2])
        ansi_escape = f"\x1b[48;2;{rgb[0]};{rgb[1]};{rgb[2]}m"
        strings.append(
            f"{ansi_escape}{name}{' ' * (longest_key - len(name))}\x1b[0m"
        )
    strings = np.array_split(strings, len(strings) // ncols)
    return tab(strings, tablefmt=tablefmt, **kw)


def rand_color():
    return pre_colors[int(np.round(len(pre_colors) * np.random.random()))]


def setup_gg_plot(*, clr_out="black", clr_in="white", figsize=(10, 10)):
    plt.style.use("ggplot")
    plt.rcParams["axes.facecolor"] = clr_in
    plt.rcParams["figure.facecolor"] = clr_out
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["text.usetex"] = True


def set_color_plot(
    *,
    axis_color="white",
    leg_edge_color="white",
    leg_label_color="white",
    tick_color="white",
    title_color="white",
    xlabel="",
    ylabel="",
    title="",
    use_legend=False,
    use_grid=False,
    use_colorbar=False,
    colorbar_kw={},
):
    def filt_dict(d, exclude_keys, include=False):
        if include:
            return {k: v for k, v in d.items() if k in exclude_keys}
        return {k: v for k, v in d.items() if k not in exclude_keys}

    plt.xlabel(xlabel, color=axis_color)
    plt.ylabel(ylabel, color=axis_color)
    plt.xticks(color=tick_color)
    plt.yticks(color=tick_color)
    if use_legend:
        plt.legend(edgecolor=leg_edge_color, labelcolor=leg_label_color)
    if use_colorbar:
        cbar = plt.colorbar()
        default_cbar_kw = {"label": "", "color": "white", "labelcolor": "white"}
        colorbar_kw = {**default_cbar_kw, **colorbar_kw}
        cbar.set_label(**filt_dict(colorbar_kw, ["labelcolor"]))
        plt.setp(
            plt.getp(cbar.ax.axes, "yticklabels"),
            color=colorbar_kw["labelcolor"],
        )

    plt.grid(visible=use_grid)

    plt.title(title, color=title_color)


def set_color_plot_global(**kw):
    def helper(
        the_title,
        *,
        title_color="white",
        exclude_keys=[],
        appendage={},
        commands=[],
    ):
        kwargs = {k: kw[k] for k in kw if k not in exclude_keys}
        kwargs = {**kwargs, **appendage}
        set_color_plot(title=the_title, title_color=title_color, **kwargs)
        for cmd in commands:
            cmd()

    return helper


def set_color_plot_static(
    *, title_color, exclude_keys=[], appendage={}, commands=[], **kw
):
    dummy = set_color_plot_global(**kw)

    def helper(the_title):
        dummy(
            the_title,
            title_color=title_color,
            exclude_keys=exclude_keys,
            appendage=appendage,
            commands=commands,
        )

    return helper


def get_frames(
    *,
    data: ArrayLike,
    iter: PlotTypes.Indices,
    fig: Figure,
    axes: List[Axes],
    plotter: PlotTypes.PlotHandler,
    framer: PlotTypes.PlotHandler = None,
    **kw,
):
    frames = []

    if framer is None:

        def default_frame_handler(*, data, idx, fig, axes, **kw2):
            fig.canvas.draw()
            frame = Image.frombytes(
                'RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
            )
            return frame

        frame_handler = default_frame_handler

    curr_kw = kw
    for idx in iter:
        curr_kw = plotter(data=data, idx=idx, fig=fig, axes=axes, **curr_kw)
        frames.append(
            frame_handler(data=data, idx=idx, fig=fig, axes=axes, **kw)
        )

    return frames


def get_frames_bool(
    *,
    data: ArrayLike,
    iter: PlotTypes.Indices,
    plotter: PlotTypes.PlotHandler,
    framer: PlotTypes.PlotHandler = None,
    fig: Figure = None,
    axes: List[Axes] = None,
    **kw,
):
    frames = []

    print('Plotting frames...', flush=True, end='')
    start_time = time()

    if (fig is None) ^ (axes is None):
        raise ValueError(
            'Either both fig and axes must be None or both must be not'
            f' None...got {(fig is None)=} and {(axes is None)=}'
        )
    elif fig is None:
        fig, axes = plt.subplots(1, 1, figsize=(10, 10))

    if framer is None:

        def default_frame_handler(*, data, idx, fig, axes, **kw2):
            fig.canvas.draw()
            frame = Image.frombytes(
                'RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
            )
            return frame

        frame_handler = default_frame_handler

    curr_kw = deepcopy(kw)
    for idx, plot_frame in iter:
        # iter_time = time()
        curr_kw = plotter(data=data, idx=idx, fig=fig, axes=axes, **curr_kw)
        curr_kw = curr_kw if curr_kw is not None else kw
        if plot_frame:
            frames.append(
                frame_handler(data=data, idx=idx, fig=fig, axes=axes, **kw)
            )
        # iter_time = time() - iter_time
        # print(
        #     f'idx={idx} took {iter_time:.2f} seconds:'
        #     f' len(frames)=={len(frames)}',
        #     flush=True,
        # )
    print(f'done in {time() - start_time:.2f} seconds.', flush=True)

    return frames


def count_frames(gif_path):
    with Image.open(gif_path) as img:
        frame_count = 0
        try:
            while True:
                img.seek(frame_count)  # Move to the next frame
                frame_count += 1
        except EOFError:
            # End of sequence: no more frames
            pass
        return frame_count


def save_frames(
    frames,
    *,
    path,
    movie_format='gif',
    duration=100,
    verbose=False,
    loop=0,
    verify_frame_count=False,
):
    target_frame_count = len(frames)
    if len(frames) == 0:
        raise ValueError("No frames to save.")
    dir, name = os.path.split(os.path.abspath(path))
    os.makedirs(dir, exist_ok=True)
    name = name.replace(f".{movie_format}", "")
    plot_name = f'{os.path.join(dir, name)}.{movie_format}'
    if verbose:
        print(f"Creating GIF at {plot_name} ...", end="")

    kw = {'duration': duration, 'loop': loop}
    kw = {k: v for k, v in kw.items() if v is not None}
    if movie_format.upper() == "PDF":
        kw = {}
    frames[0].save(
        plot_name,
        format=movie_format.upper(),
        append_images=frames[1:],
        save_all=True,
        **kw,
    )

    if verify_frame_count:
        plot_name = plot_name.replace(f'.{movie_format}', '')
        final_name = f'{plot_name}.{movie_format}'
        true_count = count_frames(final_name)
        if true_count != target_frame_count:
            # raise warning
            warnings.warn(
                f'Frame count mismatch: {true_count=},'
                f' {target_frame_count=}'
                '\nThis may be due to redundant frames'
                ' being compressed. If you believe this is a bug, please'
                ' report an issue on GitHub.',
                UserWarning,
            )


def slice_iter(*, dims, shape=None, arr=None, enum=True):
    assert (shape is None) ^ (arr is None)
    shape = shape or arr.shape
    indices = product(
        *[[slice(None)] if i in dims else range(s) for i, s in enumerate(shape)]
    )

    if arr is None:
        return indices

    if enum:

        def generator():
            for idx in indices:
                yield (idx, arr[idx])

    else:

        def generator():
            for idx in indices:
                yield arr[idx]

    return generator()


def slice_iter_bool(*, bool_gen, shape, dims):
    indices = product(
        *[[slice(None)] if i in dims else range(s) for i, s in enumerate(shape)]
    )

    for idx in indices:
        yield idx, bool_gen(idx)


def make_gifs(
    *,
    in_dir,
    out_dir=None,
    targets='all',
    opts,
    config=None,
    config1d=None,
    **kw,
):
    import torch

    out_dir = out_dir if out_dir else in_dir
    if targets == 'all':
        torch_files = glob.glob(os.path.join(in_dir, '*.pt'))
    else:

        def expand_target(x):
            if x.startswith(os.sep):
                return x.replace('.pt', '') + '.pt'
            return os.path.join(in_dir, x).replace('.pt', '') + '.pt'

        torch_files = list(map(expand_target, targets))
        if not all(os.path.exists(e) for e in torch_files):
            raise ValueError(f'Not all files in targets exist: {torch_files}')

    file_map = {os.path.basename(e).replace('.pt', ''): e for e in torch_files}
    if config is None:

        @curry
        def config(title, *, labels):
            plt.title(title)
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])
            plt.colorbar()

    if config1d is None:

        def config1d(title, *, labels):
            plt.title(title)
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])

    for k, v in file_map.items():
        if k not in opts.keys():
            raise ValueError(f'k={k} not in opts.keys()={opts.keys()}')
        u = torch.load(v)
        opts[k]['permute'] = opts[k].get('permute', list(range(u.ndim)))
        u = u.permute(opts[k]['permute'])
        opts[k]['tensor'] = u

    for k in opts.keys():
        if 'permute' in opts[k].keys():
            del opts[k]['permute']

    os.makedirs(out_dir, exist_ok=True)

    print(f'Dumping plots to {os.path.abspath(out_dir)}')
    for k, v in opts.items():
        print(f'Plotting {k}...', end='')
        if opts[k]['tensor'].ndim == 1:
            config1d(title=k, labels=v['labels'])
            y = opts[k]['tensor']
            x = torch.arange(len(y))
            plt.plot(x, y)
            plt.savefig(os.path.join(out_dir, f'{k}.jpg'))
            plt.clf()
        else:
            # plot_tensor2d_fast(
            #     **v, config=config(labels=v['labels']), name=k, **kw
            # )
            raise NotImplementedError(
                'plot_tensor2d_fast has been deprecated...look through git'
                ' history to find if you really need it.'
            )
        print('DONE')


@curry
def apply_subplot_legacy(
    data,
    sub,
    i_subplot,
    i_overlay,
    *,
    xlabel=("", ""),
    ylabel=("", ""),
    title=("", ""),
    ind_var=None,
    xlim=None,
    ylim=None,
):
    import torch  # noqa: E402

    curr_subplot = sub.order[i_subplot - 1]
    plt.subplot(*sub.shape, curr_subplot)
    specs = sub.plts[curr_subplot - 1]
    plot_type = specs.opts[i_overlay - 1].get('type', 'plot')
    if plot_type == 'plot':
        callback = plt.plot
    elif plot_type == 'scatter':
        callback = plt.scatter
    elif plot_type == 'imshow':
        callback = plt.imshow
    elif plot_type == 'hist':
        callback = plt.hist
    elif plot_type == 'bar':
        callback = plt.bar
    else:
        raise ValueError(f'Unknown plot_type: {plot_type}')

    if ind_var is not None:
        callback(
            ind_var,
            data if type(data) != torch.Tensor else data.detach().cpu(),
            **{
                k: v
                for k, v in specs.opts[i_overlay - 1].items()
                if k != 'type'
            },
        )
    else:
        callback(
            data if type(data) != torch.Tensor else data.detach().cpu(),
            **{
                k: v
                for k, v in specs.opts[i_overlay - 1].items()
                if k != 'type'
            },
        )
    xlabel_hat = specs.get('xlabel', '')
    ylabel_hat = specs.get('ylabel', '')
    title_hat = specs.get('title', '')
    ylim_hat = specs.get('ylim', 'static')
    xlim_hat = specs.get('xlim', 'static')
    ylim_hat = ylim if ylim_hat == 'static' else ylim_hat
    xlim_hat = xlim if xlim_hat == 'static' else xlim_hat
    ylim_hat = None if ylim_hat == 'dynamic' else ylim_hat
    xlim_hat = None if xlim_hat == 'dynamic' else xlim_hat

    if xlabel is not None:
        plt.xlabel(f"{xlabel[0]}{xlabel_hat}{xlabel[1]}")
    if ylabel is not None:
        plt.ylabel(f"{ylabel[0]}{ylabel_hat}{ylabel[1]}")
    if ylim_hat:
        plt.ylim(ylim_hat)
    if xlim_hat:
        plt.xlim(xlim_hat)
    if specs.get('legend', None) is not None:
        plt.legend(**specs.legend)
    if specs.get('grid', None) is not None:
        plt.grid(**specs.grid)
    if specs.get('colorbar', False):
        plt.colorbar()
    if title is not None:
        plt.title(f"{title[0]}{title_hat}{title[1]}")
    if specs.get('xticks', None) is not None:
        plt.xticks(**specs.xticks)
    if specs.get('yticks', None) is not None:
        plt.yticks(**specs.yticks)
    if specs.get('tight_layout', False):
        plt.tight_layout()


@curry
def apply_subplot(
    *,
    data,
    cfg,
    name,
    layer,
    idx=None,
    xlabel=("", ""),
    ylabel=("", ""),
    title=("", ""),
    ind_var=None,
    xlim=None,
    ylim=None,
    **other_opts,
):
    specs = cfg.plts[name]
    lyr = specs[layer]
    opts = lyr.get('opts', {})

    if lyr.get('filt', None) is not None:
        if lyr.filt == 'transpose':
            data = data.T
        else:
            data = lyr.filt(data)

    order = cfg.get('order', list(cfg.plts.keys()))
    specs_idx = order.index(name) + 1
    # input(f'order={order}, specs_idx={specs_idx}, name={name}, layer={layer}')
    plt.subplot(*cfg.sub.shape, specs_idx)

    lyr.type = lyr.get('type', 'plot')
    if lyr.type == 'plot':
        callback = plt.plot
    elif lyr.type == 'scatter':
        callback = plt.scatter
    elif lyr.type == 'imshow':
        callback = plt.imshow
    elif lyr.type == 'hist':
        callback = plt.hist
    elif lyr.type == 'bar':
        callback = plt.bar
    else:
        raise ValueError(f'Unknown plot_type: {specs.type}')

    idx = slice(None) if idx in [None, 'all', ':'] else idx
    data_slice = data[idx]
    if ind_var is not None:
        callback(ind_var, data_slice, **opts, **other_opts)
    else:
        callback(data_slice, **opts, **other_opts)

    if lyr.get('xlabel', None) is not None:
        plt.xlabel(f"{xlabel[0]}{lyr.xlabel}{xlabel[1]}")
    if lyr.get('ylabel', None) is not None:
        plt.ylabel(f"{ylabel[0]}{lyr.ylabel}{ylabel[1]}")
    if ylim not in [None, 'dynamic']:
        plt.ylim(ylim)
    if xlim:
        plt.xlim(xlim)
    if lyr.get('legend', None) is not None:
        plt.legend(**lyr.legend)
    if lyr.get('grid', None) is not None:
        plt.grid(**lyr.grid)
    if lyr.get('colorbar', False):
        plt.colorbar()
    if lyr.get('title', None) is not None:
        plt.title(f"{title[0]}{lyr.title}{title[1]}")
    if lyr.get('xticks', None) is not None:
        plt.xticks(**lyr.xticks)
    if lyr.get('yticks', None) is not None:
        plt.yticks(**specs.yticks)
    if specs.get('tight_layout', False):
        plt.tight_layout()
