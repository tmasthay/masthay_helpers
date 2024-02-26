from itertools import product
import os
import matplotlib.pyplot as plt
import io
from PIL import Image


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
    N = 1
    N_prods = [e for e in tensor.shape[2:]]
    for e in N_prods:
        N *= e
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
