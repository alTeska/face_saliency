import gc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def save_plot_without_frames(img, directory):
    fig, ax = plt.subplots(1,1)

    ax.imshow(img)
    ax.set_axis_off()
    ax.axis('off')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    fig.savefig(directory, bbox_inches='tight', pad_inches=0, transparent=True);

    plt.close(fig)
    gc.collect()
    pass
