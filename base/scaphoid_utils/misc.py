import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def get_ptcloud_img(ptcloud, elevation=240, azimuth=140):
    fig = plt.figure(figsize=(8, 8))

    x, z, y = ptcloud[:,0], ptcloud[:,1], ptcloud[:,2]
    try:
        ax = fig.gca(projection=Axes3D.name, adjustable='box')
    except:
        ax = fig.add_subplot(projection=Axes3D.name, adjustable='box')
    ax.axis('off')
    # ax.axis('scaled')
    ax.view_init(elevation, azimuth)
    max, min = np.max(ptcloud), np.min(ptcloud)
    # max_x, min_x = np.max(x), np.min(x)
    # max_y, min_y = np.max(y), np.min(y)
    # max_z, min_z = np.max(z), np.min(z)
    # ax.set_xbound(min_x, max_x)
    # ax.set_ybound(min_y, max_y)
    # ax.set_zbound(min_z, max_z)
    ax.set_xbound(min, max)
    ax.set_ybound(min, max)
    ax.set_zbound(min, max)
    ax.scatter(x, y, z, zdir='z', c=y, cmap='jet')

    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    plt.close(fig)
    return img