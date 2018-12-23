#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
from PIL import Image
import numpy as np
import os
import MyKMeans
class ColorQuantizer:
    """Quantizer for color reduction in images. Use MyKMeans class that is implemented.
    
    Parameters
    ----------
    n_colors : int, optional, default: 64
        The number of colors that wanted to exist at the end.
    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Read more from:
    http://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html
    """
    
    def __init__(self, n_colors=64, random_state=None):
        self.image = []
        self.random_state=random_state
        self.n_colors = n_colors
        self.clusters = []
        self.labels = []
        self.recreated_image = []
        pass
    
    def read_image(self, path):
        """Reads jpeg image from given path as numpy array. Stores it inside the
        class in the image variable.
        
        Parameters
        ----------
        path : string, path of the jpeg file
        """

        image = cv2.imread(os.getcwd()+'/'+path)
        self.image = image
        pass
    
    def recreate_image(self, path_to_save):
        """Reacreates image from the trained MyKMeans model and saves it to the
        given path.
        
        Parameters
        ----------
        path_to_save : string, path of the png image to save
        """
        for i in range(0,len(self.labels)):
            self.recreated_image.append(self.clusters[self.labels[i]])
        self.recreated_image = np.asarray(self.recreated_image)
        self.recreated_image = self.recreated_image.reshape(self.shape0,self.shape1,self.shape2)
        self.recreated_image = self.recreated_image.astype('uint8')
        Image.fromarray(self.recreated_image).save(os.getcwd()+'/'+path_to_save)
        pass

    def export_cluster_centers(self, path):
        """Exports cluster centers of the MyKMeans to given path.

        Parameters
        ----------
        path : string, path of the txt file
        """
        np.savetxt(os.getcwd()+'/'+path,self.clusters, fmt='%i')
        pass
        
    def quantize_image(self, path, weigths_path, path_to_save):
        """Quantizes the given image to the number of colors given in the constructor.
        
        Parameters
        ----------
        path : string, path of the jpeg file
        weigths_path : string, path of txt file to export weights
        path_to_save : string, path of the output image file
        """
        ColorQuantizer.read_image(self, path=path)
        copyOfImage = self.image.copy()
        self.shape0 = copyOfImage.shape[0]
        self.shape1 = copyOfImage.shape[1]
        self.shape2 = copyOfImage.shape[2]
        copyOfImage=copyOfImage.reshape(copyOfImage.shape[0]*copyOfImage.shape[1],copyOfImage.shape[2])
        kmeans = MyKMeans.MyKMeans(n_clusters=quantizer.n_colors, init_method='random',max_iter=20, random_state=self.random_state)

        kmeans.initialize(copyOfImage)

        kmeans.fit(copyOfImage[np.random.choice(image.shape[0]*image.shape[1],1000)])
        self.clusters = kmeans.cluster_centers

        self.labels = kmeans.predict(copyOfImage)
        ColorQuantizer.export_cluster_centers(self, weigths_path)
        ColorQuantizer.recreate_image(self,path_to_save)

        pass

        
quantizer = ColorQuantizer(n_colors=64)
quantizer.quantize_image("metu.jpg",'cluster_centers.txt','my_quantized_metu.png')


