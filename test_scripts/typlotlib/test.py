from helpers.typlotlib import \
    pre_colors, \
    rand_color, \
    setup_gg_plot, \
    set_color_plot, \
    set_color_plot_global
import matplotlib.pyplot as plt

# Create a figure
setup_gg_plot(clr_out='black', clr_in='black', figsize=(10,10))

# Create a plot
plt.plot([1,2,3,4,5], [1,2,3,4,5], label='line 1')
plt.savefig('1.jpg')
plt.clf()

plt.plot([1,2,3,4,5], [1,4,9,16,25], label='line 2')
plt.savefig('2.jpg')
plt.clf()

plt.plot([1,2,3,4,5], [1,8,27,64,125], label='line 3')
plt.savefig('3.jpg')
plt.clf()