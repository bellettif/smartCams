'''
Created on May 1, 2015

    Hybrid NN for featurization / SVM or other for classification
    The topmost classifier is called the ``head''.

@author: Francois Belletti
'''

import numpy as np
import sklearn as sk
import caffe
import cPickle as pickle

caffe.set_mode_gpu()
caffe.set_device(0)

class Hybrid_classifier:
    
    #
    #    Instantiates caffe model and head classifier
    #    @param model_filepath     String path to caffe model prototxt
    #    @param weight_filepath    String path to model's weights
    #    @param head_filepath      String path to shallow featurizer
    #    @param label_lookup       String path to label lookup table
    #    @param mean_path          String path to mean image
    def __init__(self, model_filepath, 
                 weight_filepath,
                 head_filepath = None, 
                 label_lookup = None,
                 mean_path = 'ilsvrc_2012_mean.npy',
                 context_pad = 16):
        #    Setup neural net
        self.net            =   caffe.Net(model_filepath, weight_filepath, caffe.TEST)
        #    Setup ``head'' if needed
        if head_filepath is not None:
            self.head           =   pickle.load(open(head_filepath, 'rb'))
        else:
            self.head           =   None
        #    Setup label lookup table if needed
        if label_lookup is not None:
            self.label_to_num   =   pickle.load(open(label_lookup, 'rb'))
            self.num_to_label   =   dict(zip(self.label_to_num.values(), self.label_to_num.keys()))
        else:
            self.label_to_num   =   None
            self.num_to_label   =   None
        #    Setup image transformations
        self.mean_image     =   np.load(mean_path)
        self.transformer    =   caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_mean('data', np.load(mean_path).mean(1).mean(1))
        self.transformer.set_transpose('data', (2,0,1))
        self.transformer.set_channel_swap('data', (2,1,0))
        self.transformer.set_raw_scale('data', 255.0)
        self.context_pad = context_pad
        self.configure_crop(context_pad)
    
    #
    #    Featurize a given image, works with a file path or an image
    #
    def featurize(self, input_image, target_layers = ['fc7']):
        if type(input_image) is str:
            im = caffe.io.load_image(input_image)
            out = self.net.forward_all(target_layers, 
                                       data = np.asarray([self.transformer.preprocess('data', im)]))
        else:
            out = self.net.forward_all(target_layers, 
                                       data = np.asarray([self.transformer.preprocess('data', input_image)]))
        return [out[x] for x in target_layers]
    
    #
    #    Classify a given image, works with a file path or an image
    #
    def classify_with_head(self, input_image, target_feature, log_probas = False):
        target_layers = [target_feature]
        if type(input_image) is str:
            im  = caffe.io.load_image(input_image)
            out = self.featurize(im, target_layers)
        else:
            out = self.featurize(input_image, target_layers)
        feature_vect = out[0]
        if not log_probas:
            return self.head.predict(feature_vect)
        else:
            return self.head.predict_proba(feature_vect)
        
    def classify_pure_NN(self, input_image, log_probas = False):
        out = self.net.forward_all(data = np.asarray([self.transformer.preprocess('data', input_image)]))
        probas = out.values()[-1]
        if log_probas:
            return np.log(probas)
        else:
            return np.argmax(probas)
        
    def classify(self, input_image, log_probas = False, target_feature = None):
        if self.head is not None:
            return self.classify_with_head(input_image, target_feature, log_probas)
        else:
            return self.classify_pure_NN(input_image, log_probas)
        
    def classify_windows(self, images_windows, feature_layer = 'fc7'):
        """
        Do windowed detection over given images and windows. Windows are
        extracted then warped to the input dimensions of the net.

        Take
        images_windows: (image filename, window list) iterable.
        context_crop: size of context border to crop in pixels.

        Give
        detections: list of {filename: image filename, window: crop coordinates,
            predictions: prediction vector} dicts.
        """
        # Extract windows.
        window_inputs = []
        for image_fname, windows in images_windows:
            image = caffe.io.load_image(image_fname).astype(np.float32)
            for window in windows:
                window_inputs.append(self.crop(image, window))

        # Run through the net (warping windows to input dimensions).
        in_ = self.net.inputs[0]
        caffe_in = np.zeros((len(window_inputs), window_inputs[0].shape[2])
                            + self.net.blobs[in_].data.shape[2:],
                            dtype=np.float32)
        for ix, window_in in enumerate(window_inputs):
            caffe_in[ix] = self.transformer.preprocess(in_, window_in)
        if self.head is None:
            out = self.net.forward_all(**{in_: caffe_in})
            #        predictions = out[self.outputs[0]].squeeze(axis=(2,3))
            predictions = out[self.net.outputs[0]].squeeze() # https://github.com/BVLC/caffe/issues/2041
        else:
            out = self.net.forward_all([feature_layer], **{in_: caffe_in})
            #
            feature_vects = out[feature_layer].squeeze()
            #
            predictions = self.head.predict_proba(feature_vects)
            
        # Package predictions with images and windows.
        detections = []
        ix = 0
        for image_fname, windows in images_windows:
            for window in windows:
                detections.append({
                    'window': window,
                    'prediction': predictions[ix],
                    'filename': image_fname
                })
                ix += 1
        return detections

        
    def lookup(self, label): 
        if type(label) is str:
            return self.label_to_num[label]
        else:
            return self.num_to_label[int(label)]
        
    def crop(self, im, window):
        """
        Crop a window from the image for detection. Include surrounding context
        according to the `context_pad` configuration.

        Take
        im: H x W x K image ndarray to crop.
        window: bounding box coordinates as ymin, xmin, ymax, xmax.

        Give
        crop: cropped window.
        """
        # Crop window from the image.
        crop = im[window[0]:window[2], window[1]:window[3]]

        if self.context_pad:
            box = window.copy()
            crop_size = self.net.blobs[self.net.inputs[0]].width  # assumes square
            scale = crop_size / (1. * crop_size - self.context_pad * 2)
            # Crop a box + surrounding context.
            half_h = (box[2] - box[0] + 1) / 2.
            half_w = (box[3] - box[1] + 1) / 2.
            center = (box[0] + half_h, box[1] + half_w)
            scaled_dims = scale * np.array((-half_h, -half_w, half_h, half_w))
            box = np.round(np.tile(center, 2) + scaled_dims)
            full_h = box[2] - box[0] + 1
            full_w = box[3] - box[1] + 1
            scale_h = crop_size / full_h
            scale_w = crop_size / full_w
            pad_y = round(max(0, -box[0]) * scale_h)  # amount out-of-bounds
            pad_x = round(max(0, -box[1]) * scale_w)

            # Clip box to image dimensions.
            im_h, im_w = im.shape[:2]
            box = np.clip(box, 0., [im_h, im_w, im_h, im_w])
            clip_h = box[2] - box[0] + 1
            clip_w = box[3] - box[1] + 1
            assert(clip_h > 0 and clip_w > 0)
            crop_h = round(clip_h * scale_h)
            crop_w = round(clip_w * scale_w)
            if pad_y + crop_h > crop_size:
                crop_h = crop_size - pad_y
            if pad_x + crop_w > crop_size:
                crop_w = crop_size - pad_x

            # collect with context padding and place in input
            # with mean padding
            context_crop = im[box[0]:box[2], box[1]:box[3]]
            context_crop = caffe.io.resize_image(context_crop, (crop_h, crop_w))
            crop = np.ones(self.crop_dims, dtype=np.float32) * self.crop_mean
            crop[pad_y:(pad_y + crop_h), pad_x:(pad_x + crop_w)] = context_crop
        #
        return crop
    
    
    def configure_crop(self, context_pad):
        """
        Configure crop dimensions and amount of context for cropping.
        If context is included, make the special input mean for context padding.

        Take
        context_pad: amount of context for cropping.
        """
        # crop dimensions
        in_ = self.net.inputs[0]
        tpose = self.transformer.transpose[in_]
        inv_tpose = [tpose[t] for t in tpose]
        self.crop_dims = np.array(self.net.blobs[in_].data.shape[1:])[inv_tpose]
        #.transpose(inv_tpose)
        # context padding
        self.context_pad = context_pad
        if self.context_pad:
            in_ = self.net.inputs[0]
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