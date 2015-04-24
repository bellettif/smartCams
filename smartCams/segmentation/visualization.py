'''
Created on Apr 23, 2015

    Visualization tools

@author: Francois
'''

import matplotlib.pyplot as plt

def plot_contours(pict, labels, N_REGIONS):
    plt.figure(figsize=(5, 5))
    plt.imshow(pict, cmap=plt.cm.gray)
    for l in range(N_REGIONS):
        plt.contour(labels == l, contours = 1,
                    colors=[plt.cm.spectral(l / float(N_REGIONS)), ])
    plt.xticks(())
    plt.yticks(())
    plt.title('Segmented image')
    plt.show()