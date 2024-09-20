from typing import Union

import holoviews as hv
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy
import panel as pn
import torch
from omegaconf import OmegaConf
from returns.curry import curry

from .core import depandify, pandify


def get_labels_str(
    *, ndims, loop_info, idx, column_names, active_dims, base_title
):
    loop_info["labels"] = 'inactive'
    if loop_info["labels"] == 'inactive':
        labels_str = ''
        inactive_dims = [i for i in range(ndims - 1) if i not in active_dims]

        for i in inactive_dims:
            labels_str += f'{column_names[i]}: {idx[i+1]}\n'
        # for i in inactive_dims:
        #     labels_str += f'{column_names[i]} {idx[i]}\n'
    else:
        labels_str = loop_info["labels"][idx[0]]
    labels_str = base_title + labels_str
    return labels_str


def extremes(t: Union[torch.Tensor, numpy.array, list]):
    if isinstance(t, torch.Tensor):
        return t.min().item(), t.max().item()
    elif isinstance(t, numpy.ndarray):
        return t.min(), t.max()
    return min(t), max(t)


def get_axes(slices):
    """EMPTY"""
    return [i for i in range(len(slices)) if slices[i] == slice(None)]


@curry
def rules_one(*, opts_info, loop_info, data, column_names, idx, active_dim):
    # loop = {"label": loop_info["labels"][idx[0]]}
    # loop = {"label": 'nope'}
    base_title = loop_info.get('base_title', '')
    if not base_title.endswith('\n'):
        base_title += '\n'
    labels_str = get_labels_str(
        ndims=data.ndim,
        loop_info=loop_info,
        idx=idx,
        column_names=column_names,
        active_dims=[active_dim],
        base_title=base_title,
    )

    loop = {**loop_info, 'label': labels_str}

    def hook(plot, element):
        opts_info.setdefault("yscale", {"args": [], "kwargs": {}})
        # plot.handles["axis"].set_yscale(
        #     *opts_info["yscale"]["args"], **opts_info["yscale"]["kwargs"]
        # )

    opts_info.setdefault("yscale", {"args": [], "kwargs": {}})
    opts_info["hooks"] = opts_info.get("hooks", []) + [hook]

    fixed_indices = [
        slice(None) if dim == active_dim else idx[dim]
        for dim in range(data.ndim)
    ]
    opts_info["ylim"] = opts_info.get("ylim", extremes(data))
    if opts_info["ylim"] == "active":
        active_data_slice = data[tuple(fixed_indices)]
        opts_info["ylim"] = opts_info.get("ylim", extremes(active_data_slice))
    # opts = {
    #     "ylim": opts_info["ylim"],
    #     "hooks": opts_info["hooks"],
    #     "xlabel": column_names[active_dim],
    # }
    opts = {"hooks": opts_info["hooks"], "ylim": opts_info["ylim"]}

    exclude = ['labels', 'base_title']
    loop_final = {k: v for k, v in loop.items() if k not in exclude}
    return {"opts": opts, "loop": loop_final, "plot_type": hv.Curve}


@curry
def rules_two(
    *,
    opts_info,
    loop_info,
    data: torch.Tensor,
    column_names,
    idx,
    active_dims,
    transpose,
    kdims,
    invert_xaxis,
    invert_yaxis,
    cmap,
):
    active_dims_lcl = active_dims[::-1] if transpose else active_dims

    base_title = loop_info.get('base_title', '')
    if not base_title.endswith('\n'):
        base_title += '\n'
    labels_str = get_labels_str(
        ndims=data.ndim,
        loop_info=loop_info,
        idx=idx,
        column_names=column_names,
        active_dims=active_dims_lcl,
        base_title=base_title,
    )

    dummy_bounds = [0.0, 1.0, 0.0, 1.0]
    dummy_bounds = [dummy_bounds for _ in range(data.ndim)]
    bounds = loop_info.get("bounds", dummy_bounds)
    # bounds = get_warn(loop_info, 'bounds', dummy_bounds)
    try:
        bnd1 = bounds[active_dims_lcl[0]]
        bnd2 = bounds[active_dims_lcl[1]]
    except IndexError as e:
        print(f'{bounds=}, {active_dims_lcl=}, {e=}')
        raise e
    bounds = (bnd1[0], bnd2[0], bnd1[1], bnd2[1])
    if transpose:
        bounds = (bnd2[0], bnd1[0], bnd2[1], bnd1[1])

    # print(f'{bnd1=}, {bnd2=}, {bounds=}')
    loop = {
        "label": labels_str,
        "kdims": kdims,
        "bounds": bounds,
        # "xlabel"': column_names[active_dims_lcl[0]],
        # "ylabel": column_names[active_dims_lcl[1]],
    }
    opts = {
        "cmap": cmap,
        "clim": opts_info.get("clim", extremes(data)),
        "invert_xaxis": invert_xaxis,
        "invert_yaxis": invert_yaxis,
        "invert_axes": transpose,
        "colorbar": opts_info.get("colorbar", True),
        "xaxis": opts_info.get("xaxis", "top"),
        "yaxis": opts_info.get("yaxis", "left"),
    }
    return {"opts": opts, "loop": loop, "plot_type": hv.Image}


@curry
def plot_series(*, data, rules, merge, idx, kw):
    runner = []

    def styles(i):
        idx_lcl = tuple([i] + list(idx))
        r = rules(idx=idx_lcl, **kw)
        if r['plot_type'] == hv.Curve:
            # linestyles = ['solid', 'dashed', 'dashdot']
            # markers = ['*', 'o', '^', 's', 'p', 'h', 'H', 'D', 'd', 'P', 'X']

            # extras = {
            #     'linestyle': linestyles[i] if i < 2 else 'dashed',
            #     'marker': markers[i % len(markers)],
            # }
            extras = {}
            r['opts'].update(extras)
        return r, idx_lcl

    for i in range(data.shape[0]):
        r, idx_lcl = styles(i)
        s = str(r["plot_type"])
        # print(f'{r=}, {idx_lcl=}, {s=}')
        curr = r["plot_type"](data[idx_lcl], **r['loop']).opts(**r["opts"])
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
        name="Invert Horizontal Axis Data?", value=False
    )
    invert_y_checkbox = pn.widgets.Checkbox(
        name="Invert Vertical Axis Data?", value=False
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

    colormaps = sorted([e for e in plt.colormaps() if not e.endswith("_r")])
    colormap_selector = pn.widgets.Select(
        name='Colormap', options=colormaps, value='seismic'
    )

    # Bind slider names to updated info
    @pn.depends(special_dim_0.param.value, special_dim_1.param.value)
    def update_slider_names(special_dim_0_value, special_dim_1_value):
        for dim_idx, slider in enumerate(sliders):
            if dim_idx == special_dim_0_value:
                slider.name = f"{index_names[dim_idx]} (INACT 1+2D)"
            elif dim_idx == special_dim_1_value:
                slider.name = f"{index_names[dim_idx]} (INACT 2D)"
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


def iplot(*, data, column_names=None, cols, rules):
    if column_names is not None:
        if len(column_names) == len(data.shape):
            column_names = column_names[1:]
        data_frame = pandify(data, column_names)
    else:
        data_frame = data
    return iplot_workhorse(data_frame=data_frame, cols=cols, rules=rules)


def get_warn(d, key, default):
    if key in d:
        return d[key]
    print(f"WARNING: {key} not found in {d}. Using default value {default}")
    return default


def get_servable(d):
    import torch
    from dotmap import DotMap

    if 'data' in d:
        data = d.data
    else:
        data = torch.load(d.path)
    column_names = get_warn(
        d, 'column_names', [f'VAR {i+1}' for i in range(len(data.shape[1:]))]
    )

    d.unsqueeze = get_warn(
        d, 'unsqueeze', DotMap({'perform': False, 'column_name': 'DUMMY_VAR'})
    )
    dummy_name = d.unsqueeze.get('column_name', 'DUMMY_VAR')
    num_unsqueezes = max(
        max(0, 3 - len(data.shape)), int(d.unsqueeze.get('perform', True))
    )
    for _ in range(num_unsqueezes):
        data = data.unsqueeze(0)
        column_names = [dummy_name] + column_names
    labels = [f"{column_names[0]} {i}" for i in range(data.shape[0])]
    one = rules_one(
        opts_info=d.one.get('opts_info', {}),
        loop_info={**{'labels': labels}, **d.one.get('loop_info', {})},
    )
    two = rules_two(
        opts_info=d.two.get('opts_info', {'colorbar': True}),
        loop_info={**{'labels': labels}, **d.two.get('loop_info', {})},
    )
    try:
        layout = iplot(
            data=data,
            column_names=column_names,
            cols=get_warn(d, 'cols', 2),
            rules={'one': one, 'two': two},
        )
    except:
        layout = None
        print(f'WARNING: failed to create valid layout for interactive plot')
    return {'plot': layout, 'data': data}


def get_servable_from_data(
    *,
    data,
    column_names=None,
    unsqueeze=True,
    unsqueeze_name='DUMMY_VAR',
    one_opts=None,
    one_loop=None,
    two_opts=None,
    two_loop=None,
):
    column_names = column_names or [
        f'VAR {i+1}' for i in range(len(data.shape[1:]))
    ]
    num_unsqueezes = max(max(0, 3 - len(data.shape)), int(unsqueeze))
    for _ in range(num_unsqueezes):
        data = data.unsqueeze(0)
        column_names = [unsqueeze_name] + column_names
    labels = [f"{column_names[0]} {i}" for i in range(data.shape[0])]
    one_opts = one_opts or {}
    one_loop = {**{'labels': labels}, **(one_loop or {})}
    # one = {'opts_info': one_opts, 'loop_info': one_loop}

    two_opts = {**{'colorbar': True}, **(two_opts or {})}


def get_cfg_servables(path, exclude=None):
    from dotmap import DotMap

    exclude = exclude or []
    cfg = OmegaConf.load(path)
    cfg = DotMap(OmegaConf.to_container(cfg, resolve=True))
    d = {}
    for k, v in cfg.items():
        if k not in exclude:
            try:
                d[k] = get_servable(v)
            except:
                print(
                    'WARNING: Failed to create interactive plot for'
                    f' {k}...setting to None'
                )
                d[k] = None
    return DotMap(d)


def display_nested_dict(d, parent_key=''):
    items = []
    subdicts = []
    for k, v in d.items():
        key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            # items.append(widgets.Accordion([display_nested_dict(v, key)], titles=(k,)))
            subdicts.append((k, v))
        else:
            items.append(
                widgets.HBox(
                    [
                        widgets.Label(value=f"{key}: "),
                        widgets.Label(value=str(v)),
                    ]
                )
            )
    for k, v in subdicts:
        items.append(
            widgets.Accordion([display_nested_dict(v, k)], titles=(k,))
        )
    return widgets.VBox(items)
