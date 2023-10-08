import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# global pre_colors
pre_colors = list(mcolors.CSS4_COLORS.keys())


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
    colorbar_kw={}
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
        commands=[]
    ):
        kwargs = {k: kw[k] for k in kw if k not in exclude_keys}
        kwargs = {**kwargs, **appendage}
        set_color_plot(title=the_title, title_color=title_color, **kwargs)
        for cmd in commands:
            cmd()

    return helper
