'''
Created on Apr 23, 2015

    Spectral Clustering based segmentation of the image

@author: Francois Belletti
'''

import numpy as np
from sklearn.feature_extraction import image as skimage
from sklearn.cluster import spectral_clustering

def spectral_seg(pict, N_REGIONS, 
                 beta = 5, eps = 1e-6, assign_labels = 'discretize'):
    graph = skimage.img_to_graph(pict)
    # Take a decreasing function of the gradient: an exponential
    # The smaller beta is, the more independent the segmentation is of the
    # actual image. For beta=1, the segmentation is close to a voronoi
    graph.data = np.exp(-beta * graph.data / pict.std()) + eps
    #
    #    Produce specral clusters
    #
    labels = spectral_clustering(graph,
                                 n_clusters = N_REGIONS,
                                 assign_labels = assign_labels,
                                 random_state = 1)
    #
    labels = labels.reshape(pict.shape)
    #
    return labels

