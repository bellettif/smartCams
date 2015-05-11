'''
Created on 7 mai 2015

    Attempt to fuse segmentation and classification in the lower layers of the net

@author: Francois Belletti
'''

import caffe
import numpy as np

class Segmenting_net(caffe.Net):
    """
    Segmenting net extends caffe net by conducting segmentation
    thanks to the lower layers of the alexnet
    """
    def __init__(self, model_file, pretrained_file, mean=None,
                 input_scale=None, raw_scale=None, channel_swap=None,
                 context_pad=None):
        """
        Take
            mean, input_scale, raw_scale, channel_swap: params for
            preprocessing options.
        context_pad: amount of surrounding context to take s.t. a `context_pad`
            sized border of pixels in the network input image is context, as in
            R-CNN feature extraction.
        """
        caffe.Net.__init__(self, model_file, pretrained_file, caffe.TEST)
        #
        # configure pre-processing
        in_ = self.inputs[0]
        self.transformer = caffe.io.Transformer(
            {in_: self.blobs[in_].data.shape})
        self.transformer.set_transpose(in_, (2,0,1))
        if mean is not None:
            self.transformer.set_mean(in_, mean)
        if input_scale is not None:
            self.transformer.set_input_scale(in_, input_scale)
        if raw_scale is not None:
            self.transformer.set_raw_scale(in_, raw_scale)
        if channel_swap is not None:
            self.transformer.set_channel_swap(in_, channel_swap)

        self.configure_crop(context_pad)
    
    def configure_crop(self, context_pad):
        """
        Configure crop dimensions and amount of context for cropping.
        If context is included, make the special input mean for context padding.

        Take
        context_pad: amount of context for cropping.
        """
        # crop dimensions
        in_ = self.inputs[0]
        tpose = self.transformer.transpose[in_]
        inv_tpose = [tpose[t] for t in tpose]
        self.crop_dims = np.array(self.blobs[in_].data.shape[1:])[inv_tpose]
        #.transpose(inv_tpose)
        # context padding
        self.context_pad = context_pad
        if self.context_pad:
            in_ = self.inputs[0]
            transpose = self.transformer.transpose.get(in_)
            channel_order = self.transformer.channel_swap.get(in_)
            raw_scale = self.transformer.raw_scale.get(in_)
            # Padding context crops needs the mean in unprocessed input space.
            mean = self.transformer.mean.get(in_)
            if mean is not None:
                inv_transpose = [transpose[t] for t in transpose]
                crop_mean = mean.copy().transpose(inv_transpose)
                if channel_order is not None:
                    channel_order_inverse = [channel_order.index(i)
                                            for i in range(crop_mean.shape[2])]
                    crop_mean = crop_mean[:,:, channel_order_inverse]
                if raw_scale is not None:
                    crop_mean /= raw_scale
                self.crop_mean = crop_mean
            else:
                self.crop_mean = np.zeros(self.crop_dims, dtype=np.float32)

    