import pandas as pd
import panel as pn
import holoviews as hv
import numpy as np
import panel as pn
import torch


def get_axes(slices):
    return [i for i in range(len(slices)) if slices[i] == slice(None)]


def iplot_workhorse(data_frame, **kw):
    # (1) Check if DataFrame has at least 3 columns
    if len(data_frame.columns) < 3:
        raise ValueError(
            "The DataFrame should have at least 3 columns. Use a different"
            " function for this data."
        )

    # Protect the first dimension by reshaping -- reason is because pandas treats
    #     the first dimension in a dataframe in a special way and its name is harder to extract.
    if len(data_frame) != 1:
        raise ValueError(
            "Data frame must have only one row. Please reshape your data"
            " accordingly."
        )

    # Reshape DataFrame to multi-dimensional array for easier indexing.
    #     Refer to pandas documentation if this is confusing; persist and it'll make sense.
    df_shape = [len(data_frame.index)] + [
        len(level) for level in data_frame.columns.levels
    ]
    df_shape = df_shape[1:]
    data = data_frame.values.reshape(df_shape)

    # Grab index names for auto-axis updates
    index_names = data_frame.columns.names

    # Define plotting functions
    def plot_1D(*indices):
        kw["xlabel"] = index_names[get_axes(indices)[0]]
        return hv.Curve(data[tuple(indices)]).opts(**kw)

    def plot_2D(*indices, kdims):
        kw["ylabel"] = index_names[get_axes(indices)[1]]
        kw["xlabel"] = index_names[get_axes(indices)[0]]
        return hv.Image(data[tuple(indices)], kdims=kdims).opts(**kw)

    # Create widgets
    dim_selector = pn.widgets.RadioBoxGroup(
        name="Dimension", options=["1D", "2D"], inline=True
    )

    sliders = [
        pn.widgets.IntSlider(name=index_names[i], start=0, end=df_shape[i] - 1)
        for i in range(len(index_names))
    ]

    special_dim_0 = pn.widgets.IntInput(
        name="Special Dimension 0", value=0, step=1
    )
    special_dim_1 = pn.widgets.IntInput(
        name="Special Dimension 1", value=1, step=1
    )

    # Bind slider names to updated info
    @pn.depends(special_dim_0.param.value, special_dim_1.param.value)
    def update_slider_names(special_dim_0_value, special_dim_1_value):
        for dim_idx, slider in enumerate(sliders):
            if dim_idx == special_dim_0_value:
                slider.name = (
                    f"{index_names[dim_idx]} (IGNORE SLIDER -- Plot Axis 0)"
                )
            elif dim_idx == special_dim_1_value:
                slider.name = (
                    f"{index_names[dim_idx]} (IGNORE SLIDER in 2D MODE -- Plot"
                    " Axis 1)"
                )
            else:
                slider.name = f"{index_names[dim_idx]}"

    # Bind plot updates to the widgets
    @pn.depends(
        dim_selector.param.value,
        *sliders,
        special_dim_0.param.value,
        special_dim_1.param.value,
    )
    def reactive_plot(dim, *indices):
        dim = 1 if dim == "1D" else 2
        special_dims = indices[-2:]
        indices = indices[:-2]
        plot_method = (
            plot_1D
            if dim == 1
            else lambda *x: plot_2D(
                *x,
                kdims=[
                    index_names[special_dims[0]],
                    index_names[special_dims[1]],
                ],
            )
        )
        if special_dims[0] == special_dims[1] and dim == 2:
            print(
                (
                    "Plotted dimensions are the same, dimension ="
                    f" {special_dims[0]}, no update to plot taken"
                ),
                end="\r",
            )
            return
        idx = [
            slice(None) if i in special_dims[:dim] else indices[i]
            for i in range(len(indices))
        ]
        return plot_method(*idx)

    # Create layout and return
    layout = pn.Row(
        pn.Column(
            dim_selector,
            *sliders,
            special_dim_0,
            special_dim_1,
            update_slider_names,
        ),
        reactive_plot,
    )
    return layout


def iplot(data, column_names, **kw):
    # Check data dimensions
    if len(data.shape) < 2:
        raise ValueError("Data should have at least 2 dimensions.")

    # Reshape data to have a first dimension of length 1
    reshaped_data = data.reshape((1,) + data.shape)

    # Create a multi-level column index using the provided column names
    columns = pd.MultiIndex.from_product(
        [range(dim_size) for dim_size in reshaped_data.shape[1:]],
        names=column_names,
    )

    # Create a DataFrame
    data_frame = pd.DataFrame(
        reshaped_data.reshape(reshaped_data.shape[0], -1), columns=columns
    )

    # Generate and return the interactive plot
    return iplot_workhorse(data_frame, **kw)


def iplot_disk(file_path, column_names, **kw):
    # Load data from file
    data = torch.load(file_path)
    return iplot(data, column_names, **kw)
