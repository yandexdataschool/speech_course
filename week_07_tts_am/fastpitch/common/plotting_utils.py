import matplotlib.pylab as plt
import numpy as np


def save_figure_to_numpy(fig):
    fig.canvas.draw()
    image_flat = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
    image = image_flat.reshape(*reversed(fig.canvas.get_width_height()), 3)  # (H, W, 3)
    image = np.moveaxis(image, 2, 0)
    plt.close(fig)

    return image

def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    return save_figure_to_numpy(fig)