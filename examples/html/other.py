import holoviews as hv
import numpy as np
import panel as pn
import rich.traceback
import torch

rich.traceback.install(show_locals=True)

pn.extension()
hv.extension("bokeh")

# Generate random data
data_shape = (3, 50, 50, 30, 20)
variance = 100.0
random_data = variance * torch.randn(*data_shape)

labels = [f"Random Data {i}" for i in range(data_shape[0])]
column_names = ["shot_no", "receiver_idx", "nonphysical_dummy", "time sample"]


# Function to generate slices dynamically based on active dimensions
def get_slices(index, active_dims):
    slices = [
        slice(None) if i in active_dims else index[i] for i in range(len(index))
    ]
    return tuple(slices)


# Function to get a 1D plot (trace) based on the current index
def get_1d_plot(index):
    active_dims = [2]  # Only the third dimension is active for 1D plot
    slices = get_slices(index, active_dims)
    data_slice = random_data[slices].numpy().flatten()
    curve = hv.Curve(
        (np.arange(data_slice.shape[0]), data_slice), 'Sample', 'Amplitude'
    )
    return curve.opts(title=f"1D Trace View | Indices: {index}")


# Function to get a 2D plot (heatmap) based on the current index
def get_2d_plot(index):
    active_dims = [1, 2]  # Second and third dimensions are active for 2D plot
    slices = get_slices(index, active_dims)
    data_slice = random_data[slices].numpy()
    heatmap = hv.Image(
        data_slice, bounds=(0, 0, data_slice.shape[1], data_slice.shape[0])
    ).opts(
        colorbar=True,
        cmap='Viridis',
        title=f"2D Heatmap View | Indices: {index}",
    )
    return heatmap


# Interactive widgets
index_sliders = [
    pn.widgets.IntSlider(
        name=column_names[i], start=0, end=data_shape[i], value=0
    )
    for i in range(len(column_names))
]

# Radio button to switch between 1D and 2D view
view_selector = pn.widgets.RadioButtonGroup(
    name='View Mode', options=['1D', '2D'], button_type='success'
)


# Callback function to update the plot
@pn.depends(view_selector, *index_sliders)
def update_plot(view_mode, *index):
    if view_mode == '1D':
        return get_1d_plot(index)
    elif view_mode == '2D':
        return get_2d_plot(index)


# Layout the widgets and plot
layout = pn.Column(
    pn.Row(*index_sliders),
    pn.Row(view_selector),
    update_plot,
    sizing_mode='fixed',
    width=800,
    height=600,
)

# Save the layout to a standalone HTML file
pn.panel(layout).save('mine.html', embed=True)
