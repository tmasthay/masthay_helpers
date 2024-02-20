import io
import os
from itertools import product

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tabulate import tabulate as tab
from returns.curry import curry
import glob
import copy
from time import time
import torch
from numpy.typing import ArrayLike
from typing import Iterator, Any, List, Callable, Union
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# global pre_colors
pre_colors = list(mcolors.CSS4_COLORS.keys())
pre_colors_dict = mcolors.CSS4_COLORS


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


def plot_tensor2d(
    *, tensor, labels, config=None, verbose=False, print_freq=10, delay=10, **kw
):
    """
    Plot 2D slices of a tensor and save them as JPG files. Convert the JPG files to a GIF.

    Args:
        tensor (torch.Tensor): The input tensor.
        labels (List[str]): The labels for each dimension of the tensor.
        config (Callable): A function to configure the plot title. Defaults to None.
        verbose (bool): Whether to print progress messages. Defaults to False.
        print_freq (int): The frequency at which to print progress messages. Defaults to 10.
        delay (int): The delay between frames in the generated GIF. Defaults to 10.
        **kw: Additional keyword arguments to pass to the `imshow` function.

    Returns:
        None


    """
    if config is None:
        config = lambda x: plt.title(x)

    def printv(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    # Check tensor shape
    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0)

    # Ensure tensor has more than 2 dimensions for the following
    assert len(tensor.shape) > 2

    # Create a list of indices for each dimension after the first two
    dims = [range(s) for s in tensor.shape[2:]]

    num_jpgs = 0
    N = np.prod(tensor.shape[2:])
    # Iterate over all combinations of indices
    for indices in product(*dims):
        num_jpgs += 1
        if num_jpgs % print_freq == 0:
            printv(f"Finished {num_jpgs} out of {N} slices...")
        # Use slicing to extract the appropriate 2D slice
        slices = [slice(None), slice(None)] + list(indices)
        slice_ = tensor[slices]

        # Generate title based on labels and indices
        curr_title = (
            "("
            + ", ".join([f"{e}={indices[i]}" for i, e in enumerate(labels)])
            + ")"
        )

        # Plot the slice
        plt.imshow(slice_, aspect="auto", **kw)
        config(curr_title)

        # Save the plot to a file
        filename = f"{num_jpgs:04d}.jpg"
        plt.savefig(filename)

        # Clear the figure
        plt.clf()

    # Convert the saved JPG files to a GIF
    printv("Converting JPG files to GIF...", end="", flush=True)
    os.system(f"convert -delay {delay} *.jpg movie.gif")
    printv("DONE", flush=True)

    # Remove the JPG files
    printv("Removing JPG files...", end="")
    for indices in product(*dims):
        filename = "_".join(map(str, indices)) + ".jpg"
        os.system(f"rm {filename}")
    printv("DONE")


def plot_tensor2d_fast(
    *,
    tensor,
    labels,
    config=None,
    verbose=False,
    print_freq=10,
    duration=100,
    frame_format="png",
    movie_format="gif",
    path=".",
    name="movie",
    **kw,
):
    """
    Plot a 2D slice from a 3D or higher-dimensional tensor and save it as a movie.

    Args:
        tensor (torch.Tensor): The input tensor of shape (..., H, W).
        labels (List[str]): The labels for each dimension after the first two.
        config (Callable): A function to configure the plot title. Default is None.
        verbose (bool): Whether to print progress messages. Default is False.
        print_freq (int): The frequency of progress messages. Default is 10.
        duration (int): The duration (in milliseconds) of each frame in the movie. Default is 100.
        frame_format (str): The format of the individual frames. Default is "png".
        movie_format (str): The format of the final movie. Default is "gif".
        path (str): The directory to save the movie. Default is current directory.
        name (str): The name of the movie file. Default is "movie".
        **kw: Additional keyword arguments to pass to plt.imshow().

    Returns:
        None

    Example:
        # Example usage with 3D data
        import torch
        import matplotlib.pyplot as plt

        # Create a 3D tensor
        tensor = torch.randn(10, 20, 30)

        # Define labels for each dimension
        labels = ["X", "Y", "Z"]

        # Configure the plot
        def config(title):
            plt.title(title)
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])

        # Plot and save the movie
        plot_tensor2d_fast(tensor=tensor, labels=labels, config=config)
    """

    def printv(*args, **kwargs):
        kwargs["flush"] = True
        if verbose:
            print(*args, **kwargs)

    printv("Setting up config...", end="")
    if config is None:
        config = lambda x: plt.title(x)

    frame_format = frame_format.lower().replace(".", "")
    movie_format = movie_format.lower().replace(".", "")

    # Check tensor shape
    if len(tensor.shape) == 2:
        # Directly plot the tensor and save
        # plt.imshow(tensor, aspect="auto", **kw)
        # config(labels)
        # plt.savefig(f"tensor_plot.{frame_format}")
        # plt.clf()
        # return  # exit function after handling this casegb
        tensor = tensor.unsqueeze(-1)

    # Ensure tensor has more than 2 dimensions for the following
    assert len(tensor.shape) > 2

    # Create a list of indices for each dimension after the first two
    dims = [range(s) for s in tensor.shape[2:]]

    frames = []  # list to store each frame in memory
    N = np.prod(tensor.shape[2:])
    y_label, x_label = labels[0:2]
    labels = labels[2:]
    printv("DONE")
    # Iterate over all combinations of indices
    for indices in product(*dims):
        # Use slicing to extract the appropriate 2D slice
        slices = [slice(None), slice(None)] + list(indices)
        slice_ = tensor[slices]

        # Generate title based on labels and indices
        curr_title = (
            "("
            + ", ".join([f"{e}={indices[i]}" for i, e in enumerate(labels)])
            + ")"
        )

        # Plot the slice
        plt.imshow(slice_, aspect="auto", **kw)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        config(curr_title)

        # Convert plot to PIL Image and append to frames
        # plt.axis('off')  # turn off the axis
        # plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format=frame_format)
        buf.seek(0)
        frames.append(Image.open(buf))

        # Clear the figure
        plt.clf()

        if (len(frames) % print_freq) == 0:
            printv(f"Processed {len(frames)} out of {N} slices")

    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    name = name.replace(f".{movie_format}", "")
    plot_name = os.path.join(abs_path, name) + f".{movie_format}"
    printv(f"Creating GIF at {plot_name}...", end="")
    frames[0].save(
        plot_name,
        format=movie_format.upper(),
        append_images=frames[1:],
        save_all=True,
        duration=duration,
        loop=0,
    )
    printv("DONE")

    buf.close()


def plot_tensor2d_subplot(
    *,
    tensor,
    labels,
    config=None,
    verbose=False,
    print_freq=10,
    duration=100,
    frame_format="png",
    movie_format="gif",
    path=".",
    name="movie",
    subplot_shape='horizontal',
    fig_size=(18.5, 10.5),
    layout_args=None,
    **kw,
):
    def printv(*args, **kwargs):
        kwargs["flush"] = True
        if verbose:
            print(*args, **kwargs)

    if type(tensor) != list:
        printv('Converting tensor to list')
        tensor = [e for e in tensor]

    if type(tensor[0]) != torch.Tensor:
        printv('Converting tensor to torch.Tensor')
        tensor = [torch.from_numpy(e) for e in tensor]

    for i, e in enumerate(tensor):
        while len(tensor[i].shape) <= 2:
            printv(f'Expanding tensor[{i}] to shape {e.shape + (1,)}')
            tensor[i] = tensor[i].unsqueeze(-1)

    loop_indices = np.array([e.shape[2:] for e in tensor])
    if len(np.where(loop_indices != loop_indices[0])[0]) > 0:
        raise ValueError(
            f'All dimensions except the first two must be the same for all'
            f' tensors.'
        )

    layout_args = layout_args or {}
    if config is None:

        def config_helper(fig, ax, im, title):
            ax.set_title(title)
            fig.colorbar(im, ax=ax)

        config = config_helper

    printv("Setting up config...", end="")
    if config is None:
        config = lambda x: plt.title(x)

    frame_format = frame_format.lower().replace(".", "")
    movie_format = movie_format.lower().replace(".", "")

    if subplot_shape == 'horizontal':
        subplot_shape = (1, len(tensor))
    elif subplot_shape == 'vertical':
        subplot_shape = (len(tensor), 1)

    if subplot_shape == (1, 1):
        fig, axs = plt.subplots()
    else:
        fig, axs = plt.subplots(*subplot_shape)
    if type(axs) != np.ndarray:
        axs = np.array([axs])
    if type(axs[0]) != np.ndarray:
        axs = np.array([axs])
    fig.set_size_inches(*fig_size)
    fig.tight_layout(**layout_args)

    dims = [range(s) for s in tensor[0].shape[2:]]
    frames = []
    N = np.prod(tensor[0].shape[2:])

    def sub_idx(k):
        if subplot_shape[0] == 1:
            return 0, k
        elif subplot_shape[1] == 1:
            return k, 0
        return k // subplot_shape[1], k % subplot_shape[1]

    start_time = time()
    for indices in product(*dims):
        for i in range(len(tensor)):
            print(f'{i}, {indices}', flush=True)
            slices = [slice(None), slice(None)] + list(indices)
            slice_ = tensor[i][slices]

            # Generate title based on labels and indices
            curr_title = (
                labels[i][0]
                + "\n("
                + ", ".join(
                    [f"{e}={indices[i]}" for i, e in enumerate(labels[i][3:])]
                )
                + ")"
            )

            row_idx, col_idx = sub_idx(i)
            ax = axs[row_idx, col_idx]
            im = ax.imshow(slice_, aspect="auto", **kw)
            ax.set_xlabel(labels[i][1])
            ax.set_ylabel(labels[i][2])
            config(fig, ax, im, curr_title)

        # buf = io.BytesIO()
        # plt.savefig(buf, format=frame_format)
        # buf.seek(0)
        # frames.append(Image.open(buf))
        fig.canvas.draw()
        frame = Image.frombytes(
            'RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
        frames.append(frame)

        plt.clf()
        plt.close(fig)
        fig, axs = plt.subplots(*subplot_shape)
        fig.set_size_inches(*fig_size)
        fig.tight_layout(**layout_args)
        if type(axs) != np.ndarray:
            axs = np.array([axs])
        if type(axs[0]) != np.ndarray:
            axs = np.array([axs])
            # fig.set_size_inches(18.5, 10.5)
            # fig.tight_layout(pad=3.0)

        if len(frames) % print_freq == 0:
            full_time = time() - start_time
            start_time = time()
            printv(f"Processed {len(frames)} out of {N} slices...{full_time} s")

    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    name = name.replace(f".{movie_format}", "")
    plot_name = os.path.join(abs_path, name) + f".{movie_format}"
    printv(f"Creating GIF at {plot_name} ...", end="")
    frames[0].save(
        plot_name,
        format=movie_format.upper(),
        append_images=frames[1:],
        save_all=True,
        duration=duration,
        loop=0,
    )


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
    fig: Figure,
    axes: List[Axes],
    plotter: PlotTypes.PlotHandler,
    framer: PlotTypes.PlotHandler = None,
    **kw,
):
    frames = []

    from time import time

    if framer is None:

        def default_frame_handler(*, data, idx, fig, axes, **kw2):
            fig.canvas.draw()
            frame = Image.frombytes(
                'RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
            )
            return frame

        frame_handler = default_frame_handler

    curr_kw = kw
    for idx, plot_frame in iter:
        iter_time = time()
        curr_kw = plotter(data=data, idx=idx, fig=fig, axes=axes, **curr_kw)
        if plot_frame:
            frames.append(
                frame_handler(data=data, idx=idx, fig=fig, axes=axes, **kw)
            )
        iter_time = time() - iter_time
        print(
            f'idx={idx} took {iter_time:.2f} seconds:'
            f' len(frames)=={len(frames)}',
            flush=True,
        )

    return frames


def save_frames(
    frames, *, path, movie_format='gif', duration=100, verbose=False, loop=0
):
    dir, name = os.path.split(os.path.abspath(path))
    os.makedirs(dir, exist_ok=True)
    name = name.replace(f".{movie_format}", "")
    plot_name = f'{os.path.join(dir, name)}.{movie_format}'
    if verbose:
        print(f"Creating GIF at {plot_name} ...", end="")
    frames[0].save(
        plot_name,
        format=movie_format.upper(),
        append_images=frames[1:],
        save_all=True,
        duration=duration,
        loop=loop,
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
            plot_tensor2d_fast(
                **v, config=config(labels=v['labels']), name=k, **kw
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
    *data,
    sub,
    name,
    layer,
    idx=None,
    xlabel=("", ""),
    ylabel=("", ""),
    title=("", ""),
    ind_var=None,
    xlim=None,
    ylim=None,
):
    specs = sub.plts[name]
    opts = specs.opts[layer]

    order = sub.get('order', list(sub.plts.keys()))
    specs_idx = order.index(name)

    plt.subplot(*sub.shape, specs_idx)
    plot_type = opts.get('type', 'plot')
    opts = {k: v for k, v in opts.items() if k != 'type'}
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

    idx = slice(None) if idx in [None, 'all', ':'] else idx
    data_slice = data[idx]
    data_slice = (
        data_slice
        if type(data_slice) != torch.Tensor
        else data_slice.detach().cpu()
    )
    if ind_var is not None:
        callback(ind_var, data, **opts)
    else:
        callback(data, **opts)

    if ylim is None:
        ylim = specs.get('ylim', 'static')
        if ylim == 'static':
            ylim = (data.min(), data.max())

    if xlabel is not None:
        plt.xlabel(f"{xlabel[0]}{xlabel_hat}{xlabel[1]}")
    if ylabel is not None:
        plt.ylabel(f"{ylabel[0]}{ylabel_hat}{ylabel[1]}")
    if ylim not in [None, 'dynamic']:
        plt.ylim(ylim_hat)
    if xlim:
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
