from masthay_helpers.typlotlib import *
import torch


def main():
    # Takes ~30 seconds on my machine

    # Constants
    PI = torch.tensor(3.14159265358979323846)
    T_MIN = -2 * PI
    T_MAX = 2 * PI

    # Spatial domain parameters
    # y will come first because we want the y-axis to be vertical
    #     and that's just how imshow rolls.
    y0, y1 = -2, 2  # Define the range for y
    x0, x1 = -2, 2  # Define the range for x
    s0, s1 = 0.01, 1.0

    # Tensor size parameters
    y_size, x_size, s_size, t_size = 200, 200, 5, 50

    print("Generating data...", end="")
    # Generate the spatial and temporal grids
    y = torch.linspace(y0, y1, y_size)
    x = torch.linspace(x0, x1, x_size)
    t = torch.linspace(T_MIN, T_MAX, t_size)
    s = torch.linspace(s0, s1, s_size)

    # Create meshgrids
    Y, X, S, T = torch.meshgrid(y, x, s, t, indexing="ij")

    # Compute the function values
    r = torch.sqrt(X**2 + Y**2)
    wave_number = T
    smoother = S
    approx_helmholtz_kernel_real = torch.cos(wave_number * r) / (r + smoother)
    Z = approx_helmholtz_kernel_real

    print("DONE")

    # Now, Z is your 3D tensor populated with the function values.
    # You can then use the `plot_tensor2d` function from `typlotlib` to visualize slices of this tensor.

    setup_gg_plot(clr_out="black", clr_in="lightsteelblue", figsize=(10, 10))

    config = set_color_plot_static(
        axis_color="limegreen",
        title_color="gold",
        tick_color="white",
        xlabel="X",
        ylabel="Y",
        use_colorbar=True,
        colorbar_kw={
            "label": r"G(x,y,t)",
            "color": "white",
            "labelcolor": "white",
        },
    )

    plot_tensor2d_fast(
        tensor=Z,
        labels=["s", "t"],
        config=config,
        cmap="seismic",
        verbose=True,
        print_freq=3,
        frame_format="jpg",
        movie_format="gif",
        duration=100,
    )


if __name__ == "__main__":
    main()
