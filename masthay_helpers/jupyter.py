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
from returns.curry import curry


def get_axes(slices):
    """EMPTY"""
    return [i for i in range(len(slices)) if slices[i] == slice(None)]


@curry
def rules_one(*, opts_info, loop_info, data, column_names, idx, active_dim):
    loop = {"label": loop_info["labels"][idx[0]]}

    def hook(plot, element):
        plot.handles["axis"].set_yscale(
            *opts_info["yscale"]["args"], **opts_info["yscale"]["kwargs"]
        )

    opts_info.setdefault("yscale", {"args": [], "kwargs": {}})
    opts_info["hooks"] = opts_info.get("hooks", []) + [hook]
    opts_info["ylim"] = opts_info.get("ylim", (data.min(), data.max()))
    opts = {
        "ylim": opts_info["ylim"],
        "hooks": opts_info["hooks"],
        "xlabel": column_names[active_dim],
    }

    return {"opts": opts, "loop": loop, "plot_type": hv.Curve}


@curry
def rules_two(
    *,
    opts_info,
    loop_info,
    data,
    column_names,
    idx,
    active_dims,
    transpose,
    kdims,
    invert_xaxis,
    invert_yaxis,
    cmap,
):
    active_dims = active_dims[::-1] if transpose else active_dims
    loop = {
        "label": loop_info["labels"][idx[0]],
        "kdims": kdims,
        "xlabel": column_names[active_dims[0]],
        "ylabel": column_names[active_dims[1]],
    }
    opts = {
        "cmap": cmap,
        "clim": opts_info.get("clim", (data.min(), data.max())),
        "invert_xaxis": invert_xaxis,
        "invert_yaxis": invert_yaxis,
        "invert_axes": transpose,
        "colorbar": opts_info.get("colorbar", True),
    }
    return {"opts": opts, "loop": loop, "plot_type": hv.Image}


@curry
def plot_series(*, data, rules, merge, idx, kw):
    runner = []
    for i in range(data.shape[0]):
        idx_lcl = tuple([i] + list(idx))
        r = rules(idx=idx_lcl, **kw)
        curr = r["plot_type"](data[idx_lcl], **r["loop"]).opts(**r["opts"])
        print(f'opts = {r["opts"]}', flush=True)
        runner.append(curr)
    return merge(runner)


def iplot_workhorse(*, data_frame, cols=1, rules):
    # (1) Check if DataFrame has at least 3 columns
    if len(data_frame.columns) < 3:
        raise ValueError(
            "The DataFrame should have at least 3 columns. Use a different"
            " function for this data."
        )

    data, index_names = depandify(data_frame)
    df_shape = data.shape

    plot_1D = plot_series(
        data=data,
        rules=rules["one"](data=data, column_names=index_names),
        merge=hv.Overlay,
    )
    plot_2D = plot_series(
        data=data,
        rules=rules["two"](data=data, column_names=index_names),
        merge=(lambda runner: hv.Layout(runner).cols(cols)),
    )

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
        name='Colormap', options=plt.colormaps(), value='nipy_spectral'
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
        special_dim_0.param.value,
        special_dim_1.param.value,
        *sliders,
    )
    def reactive_plot(
        dim,
        transpose,
        invert_xaxis,
        invert_yaxis,
        cmap,
        special_dim_0,
        special_dim_1,
        *indices,
    ):
        dim = 1 if dim == "1D" else 2
        if special_dim_0 == special_dim_1 and dim == 2:
            print("NO CHANGE", flush=True)
            return reactive_plot.last
        special_dims = [special_dim_0, special_dim_1]
        idx = [
            slice(None) if i in special_dims[:dim] else indices[i]
            for i in range(len(indices))
        ]
        if dim == 1:
            res = plot_1D(idx=idx, kw={"active_dim": special_dims[0]})
        else:
            res = plot_2D(
                idx=idx,
                kw={
                    "transpose": transpose,
                    "invert_xaxis": invert_xaxis,
                    "invert_yaxis": invert_yaxis,
                    "cmap": cmap,
                    "kdims": [
                        index_names[special_dim_0],
                        index_names[special_dim_1],
                    ],
                    "active_dims": special_dims,
                },
            )

        reactive_plot.last = res
        return res

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


def iplot(*, data, column_names, cols, rules):
    data_frame = pandify(data, column_names)
    return iplot_workhorse(data_frame=data_frame, cols=cols, rules=rules)
