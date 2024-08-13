import inspect
import sys

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import rich.traceback
import torch

from mh.jupyter import iplot, plot_series, rules_one, rules_two

rich.traceback.install(show_locals=True)

pn.extension()
np.random.seed(0)
hv.extension('bokeh')

data_shape = (3, 50, 50, 30, 20)
steps = (1, 10, 10, 10, 10)
variance = 100.0
random_data = variance * torch.randn(*data_shape)

labels = [f'Random Data {i}' for i in range(data_shape[0])]
column_names = ['shot_no', 'receiver_idx', 'nonphysical_dummy', 'time sample']
rules = {
    'one': rules_one(
        opts_info={'yscale': {'args': ('linear',), 'kwargs': {}}},
        loop_info={'labels': labels},
    ),
    'two': rules_two(
        opts_info={'colorbar': True}, loop_info={'labels': labels}
    ),
}
u = plot_series(data=random_data, rules=rules, idx=[0, 0, 0])
# print(inspect.signature(u))

layout = iplot(
    data=random_data,
    column_names=column_names,
    cols=2,
    rules=rules,
    steps=steps,
)
layout.servable()
