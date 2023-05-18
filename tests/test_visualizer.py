import utils.ext_import

from sphdet.visualizers import SphVisualizer
import numpy as np
import torch
import matplotlib.pyplot as plt

def test_sph_visualizer():
    vis = SphVisualizer(canvas='light', canvas_size=(512, 1024), with_lonlat=True)
    vis.quick_test([170, 60, 45, 90], [210, 40, 140, 35], mode='auto')

    vis.show('vis/test/visualizer/sph_visualizer.jpg')

def test_mat_visualizer():
    mat = np.arange(16).reshape(((4, 4)))
    from sphdet.visualizers.plot_visualizer import plot_matrix
    plot_matrix(mat, 'vis/test/visualizer/mat.png')


if __name__ == '__main__':
    test_sph_visualizer()
    #test_mat_visualizer()