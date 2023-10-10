import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from itertools import product
import os
from PIL import Image
import io
from tabulate import tabulate as tab

# global pre_colors
pre_colors = list(mcolors.CSS4_COLORS.keys())
pre_colors_dict = mcolors.CSS4_COLORS


def colors_str(
    *,
    colors,
    normalize=True,
    ncols=5,
    tablefmt="plain",
    **kw,
):
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
    if config is None:
        config = lambda x: plt.title(x)

    def printv(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    # Check tensor shape
    if len(tensor.shape) == 2:
        # Directly plot the tensor and save
        plt.imshow(tensor, aspect="auto", **kw)
        config(labels)
        plt.savefig("tensor_plot.jpg")
        plt.clf()
        return  # exit function after handling this case

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
        plt.imshow(tensor, aspect="auto", **kw)
        config(labels)
        plt.savefig(f"tensor_plot.{frame_format}")
        plt.clf()
        return  # exit function after handling this case

    # Ensure tensor has more than 2 dimensions for the following
    assert len(tensor.shape) > 2

    # Create a list of indices for each dimension after the first two
    dims = [range(s) for s in tensor.shape[2:]]

    frames = []  # list to store each frame in memory
    N = np.prod(tensor.shape[2:])
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
