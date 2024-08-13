import holoviews as hv
import numpy as np
import torch
from bokeh.resources import INLINE
from holoviews import opts

hv.extension('bokeh')

# Example PyTorch tensor (some data that changes with a parameter)
# tensor = torch.tensor([[1, 2], [2, 3], [3, 5]])
domain = torch.linspace(0, 10, 100)


# Function to update data based on a slider value
def update_data(param):
    updated_tensor = torch.sin(domain * param)
    data = updated_tensor.numpy()
    return hv.Curve(data, label=f'Sample Curve (param={param})')


# Create a HoloViews DynamicMap with a slider
dmap = hv.DynamicMap(update_data, kdims=['param']).redim.range(param=(1, 10))

# Optional: Customize the appearance
dmap.opts(
    opts.Curve(
        width=600,
        height=400,
        tools=['hover'],
        title="Interactive Plot with Slider",
    )
)

# Export the plot to an HTML file
hv.save(dmap, 'mine.html', backend='bokeh', resources=INLINE)
