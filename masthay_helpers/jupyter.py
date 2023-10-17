import pandas as pd
import panel as pn
import holoviews as hv
import numpy as np
import torch
from .global_helpers import (
    filter_kw,
    pandify,
    depandify,
    get_full_slices,
    call_vars,
)
import copy
import importlib
import matplotlib.pyplot as plt


def get_axes(slices):
    return [i for i in range(len(slices)) if slices[i] == slice(None)]


def iplot_workhorse(*, data_frame, cols=1, one, two):
    # (1) Check if DataFrame has at least 3 columns
    if len(data_frame.columns) < 3:
        raise ValueError(
            "The DataFrame should have at least 3 columns. Use a different"
            " function for this data."
        )

    data, index_names = depandify(data_frame)
    df_shape = data.shape

    # Define plotting functions
    def plot_1D(special_dim_0, *indices):
        lcl_one = copy.deepcopy(one)
        lcl_one['xlabel'] = lcl_one['loop']['xlabel'][special_dim_0]
        loop = copy.deepcopy(lcl_one["loop"])
        del lcl_one["loop"]

        overlay = None
        for i in range(data.shape[0]):
            idx = tuple([i] + list(indices))
            curve = hv.Curve(data[idx], label=loop['labels'][i])
            curve = curve.opts(**lcl_one)
            overlay = curve if overlay is None else overlay * curve
        return overlay

    def plot_2D(*indices, kdims, transpose, invert_xaxis, invert_yaxis, cmap):
        plots = []
        lcl_two = copy.deepcopy(two)
        loop = copy.deepcopy(two['loop'])
        del lcl_two['loop']
        lcl_two['invert_axes'] = transpose
        lcl_two['invert_xaxis'] = invert_xaxis
        lcl_two['invert_yaxis'] = invert_yaxis
        lcl_two['cmap'] = cmap

        full = get_full_slices(indices)
        for i in range(data.shape[0]):
            idx = tuple([i] + list(indices))
            curr = hv.Image(data[idx], kdims=kdims).redim.range(
                x=(0, df_shape[full[1]]), y=(0, df_shape[full[0]])
            )
            lcl_two["title"] = loop['labels'][i]
            curr = curr.opts(**lcl_two)
            plots.append(curr)
        layout = hv.Layout(plots)
        assert type(cols) == int
        layout = layout.cols(cols)
        return layout

    # Create widgets
    dim_selector = pn.widgets.RadioBoxGroup(
        name="Dimension", options=["1D", "2D"], inline=True
    )

    transpose_checkbox = pn.widgets.Checkbox(name="Transpose?", value=False)
    invert_x_checkbox = pn.widgets.Checkbox(
        name="Invert Horizontal Axis?", value=False
    )
    invert_y_checkbox = pn.widgets.Checkbox(
        name="Invert Vertical Axis?", value=False
    )

    sliders = [
        pn.widgets.IntSlider(
            name=index_names[i], start=0, end=max(1, df_shape[i + 1] - 1)
        )
        for i in range(len(index_names))
    ]

    special_dim_0 = pn.widgets.IntInput(
        name="Special Dimension 0",
        value=0,
        step=1,
        start=0,
        end=len(sliders) - 1,
    )
    special_dim_1 = pn.widgets.IntInput(
        name="Special Dimension 1",
        value=1,
        step=1,
        start=0,
        end=len(sliders) - 1,
    )

    colormap_selector = pn.widgets.Select(
        name='Colormap', options=plt.colormaps(), value='jet'
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
        transpose_checkbox.param.value,
        invert_x_checkbox.param.value,
        invert_y_checkbox.param.value,
        colormap_selector.param.value,
        *sliders,
        special_dim_0.param.value,
        special_dim_1.param.value,
    )
    def reactive_plot(
        dim, transpose, invert_xaxis, invert_yaxis, cmap, *indices
    ):
        dim = 1 if dim == "1D" else 2
        special_dims = indices[-2:]
        indices = indices[:-2]
        plot_method = (
            lambda *x: plot_1D(special_dim_0.value, *x)
            if dim == 1
            else lambda *x: plot_2D(
                *x,
                transpose=transpose,
                invert_xaxis=invert_xaxis,
                invert_yaxis=invert_yaxis,
                cmap=cmap,
                kdims=[
                    index_names[special_dims[0]],
                    index_names[special_dims[1]],
                ],
            )
        )
        if special_dims[0] == special_dims[1] and dim == 2:
            return reactive_plot.last
        idx = [
            slice(None) if i in special_dims[:dim] else indices[i]
            for i in range(len(indices))
        ]
        res = plot_method(*idx)
        reactive_plot.last = res
        return plot_method(*idx)

    # Create layout and return
    layout = pn.Row(
        pn.Column(
            dim_selector,
            transpose_checkbox,
            invert_x_checkbox,
            invert_y_checkbox,
            colormap_selector,
            *sliders,
            special_dim_0,
            special_dim_1,
            update_slider_names,
        ),
        reactive_plot,
    )
    return layout


def iplot(*, data, column_names, cols, one, two):
    data_frame = pandify(data, column_names)
    return iplot_workhorse(data_frame=data_frame, cols=cols, one=one, two=two)
