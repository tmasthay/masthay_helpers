import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

#global pre_colors
pre_colors = list(mcolors.CSS4_COLORS.keys())

def rand_color():
    return pre_colors[int(np.round(len(pre_colors) * np.random.random()))]

def setup_gg_plot(fig_color, face_color):
    plt.style.use('ggplot')
    plt.rcParams['axes.facecolor'] = fig_color
    plt.rcParams['figure.facecolor'] = face_color
    plt.rcParams['text.usetex'] = True 

def set_color_plot(**kw):
    axis_color = kw.get('axis_color', 'white')
    leg_edge_color = kw.get('leg_edge_color', 'white')
    leg_label_color = kw.get('leg_label_color', 'white')
    tick_color = kw.get('tick_color', 'white')
    title_color = kw.get('title_color', 'white')
    xlabel = kw.get('xlabel', '')
    ylabel = kw.get('ylabel', '')
    title = kw.get('title', '')
    use_legend = kw.get('use_legend', False)
    use_grid = kw.get('use_grid', False)
    plt.xlabel(xlabel, color=axis_color)
    plt.ylabel(ylabel, color=axis_color)
    plt.xticks(color=tick_color)
    plt.yticks(color=tick_color)
    if( use_legend ):
        plt.legend(
            edgecolor=leg_edge_color, 
            labelcolor=leg_label_color
        )
    plt.grid(use_grid)
    plt.title(title, color=title_color)

def set_color_plot_global(**kw):
    def helper(the_title,the_title_color):
        set_color_plot(title=the_title, title_color=the_title_color, **kw)
    return helper
