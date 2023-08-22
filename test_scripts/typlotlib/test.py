from helpers.typlotlib import *
import matplotlib.pyplot as plt
import os

# Create a figure
setup_gg_plot(clr_out='black', clr_in='black', figsize=(10,10))
config_plot = set_color_plot_global(
    axis_color='white',
    leg_edge_color='white',
    leg_label_color='white',
    tick_color='white',
    xlabel='',
    ylabel='',
    use_legend=False,
    use_grid=False,
    use_colorbar=True,
    colorbar_kw={'label': 'Some unit of measurement', 'color': 'white'},
)


# Create a plot
plt.imshow(np.random.random((100,100)))
config_plot('Random Image')
plt.savefig('random.jpg')
plt.clf()

plt.plot(range(10), range(10))
config_plot(
    'Identity',
    title_color='white',
    exclude_keys=['use_colorbar', 'colorbar_kw'],
    appendage={'tick_color': rand_color()}
)
plt.savefig('identity.jpg')
plt.clf()

os.system('code random.jpg')
os.system('code identity.jpg')