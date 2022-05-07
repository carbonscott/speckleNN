#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import numpy as np
from skimage.transform import rotate, resize


class RandomPatch:
    """ Randomly place num_patch patch with the size of size_y * size_x onto an image.
    """

    def __init__(self, num_patch, size_patch_y,    size_patch_x, 
                                  var_patch_y = 0, var_patch_x = 0, 
                                  is_random_flip = False,
                                  is_return_mask = False):
        self.num_patch      = num_patch                   # ...Number of patches
        self.size_patch_y   = size_patch_y                # ...Size of the patch in y dimension
        self.size_patch_x   = size_patch_x                # ...Size of the patch in x dimension
        self.var_patch_y    = max(0, min(var_patch_y, 1)) # ...Percent variation with respect to the patch size in x dimension
        self.var_patch_x    = max(0, min(var_patch_x, 1)) # ...Percent variation with respect to the patch size in y dimension
        self.is_random_flip = is_random_flip              # ...Is it allowed to have random flip between x and y dimensions
        self.is_return_mask = is_return_mask              # ...Is it allowed to return a mask

        return None


    def __call__(self, img):
        # Get the size of the image...
        size_img_y, size_img_x = img.shape

        # Construct a mask of ones with the same size of the image...
        mask = np.ones_like(img)

        # Generate a number of random position...
        pos_y = np.random.randint(low = 0, high = size_img_y, size = self.num_patch)
        pos_x = np.random.randint(low = 0, high = size_img_x, size = self.num_patch)

        # Stack two column vectors to form an array of (x, y) indices...
        pos_y = pos_y.reshape(-1,1)
        pos_x = pos_x.reshape(-1,1)
        pos   = np.hstack((pos_y, pos_x))

        # Place patch of zeros at all pos as top-left corner...
        for (y, x) in pos:
            size_patch_y = self.size_patch_y
            size_patch_x = self.size_patch_x

            # Apply random variance...
            # Find the absolute max pixel to vary
            varsize_patch_y = int(size_patch_y * self.var_patch_y)
            varsize_patch_x = int(size_patch_x * self.var_patch_x)

            # Sample an integer from the min-max pixel to vary
            # Also allow user to set var_patch = 0 when variance is not desired
            delta_patch_y = np.random.randint(low = -varsize_patch_y, high = varsize_patch_y if varsize_patch_y else 1)
            delta_patch_x = np.random.randint(low = -varsize_patch_x, high = varsize_patch_x if varsize_patch_x else 1)

            # Apply the change
            size_patch_y += delta_patch_y
            size_patch_x += delta_patch_x

            # Apply random flip???
            is_flip = random.choice((True, False)) if self.is_random_flip else self.is_random_flip
            if is_flip: size_patch_y, size_patch_x = size_patch_x, size_patch_y 

            # Find the limit of the bottom/right-end of the patch...
            y_end = min(y + size_patch_y, size_img_y)
            x_end = min(x + size_patch_x, size_img_x)

            # Patch the area with zeros...
            mask[y : y_end, x : x_end] = 0

        # Appy the mask...
        img *= mask

        # Construct the return value...
        # Parentheses are necessary
        output = img if not self.is_return_mask else (img, mask)

        return output




class Crop:
    """ Crop an image given area defined by its top-left corner (crop_orig)
        and bottom-right corner (crop_end).
    """
    def __init__(self, crop_orig, crop_end):
        self.crop_orig = crop_orig
        self.crop_end  = crop_end


    def __call__(self, img):
        y_orig, x_orig = self.crop_orig
        y_end , x_end  = self.crop_end

        # Prevent end point goes out of bound...
        size_img_y, size_img_x = img.shape
        y_end = min(y_end, size_img_y)
        x_end = min(x_end, size_img_x)

        img_crop = img[y_orig:y_end, x_orig:x_end]

        return img_crop




class RandomRotate:
    """ Apply random rotation to an image around the beam center.
    """
    def __init__(self, angle = None, center = (0, 0)):
        self.angle  = angle
        self.center = center

        return None


    def __call__(self, img): 
        if self.angle is None:
            # Sample an angle from 0 to 360 uniformly...
            angle = np.random.uniform(low = 0, high = 360)

        return rotate(img, angle = angle, center = self.center)




class PanelZoom:
    """ Zoom in an area of the same aspect ratio as the original panel.  
        It returns an zoom-in image with the same size.

        Side effect: distortion occurs when the cropping aspect ratio is
        markedly different from the one of the input image.
    """
    def __init__(self, crop_orig, crop_end):
        self.crop = Crop(crop_orig, crop_end)

        return None


    def __call__(self, img):
        size_img_y, size_img_x = img.shape

        img_crop = self.crop(img)

        img_zoom = resize(img_crop, (size_img_y, size_img_x), anti_aliasing = True)

        return img_zoom




class RandomPanelZoom:
    """ WARNING!!! It's a hard-coded zoom, bottom right corner is fixated.  
        Zoom in an area of the same aspect ratio as the original panel.  
        It returns an zoom-in image with the same size.

        Side effect: distortion occurs when the cropping aspect ratio is
        markedly different from the one of the input image.
    """
    def __init__(self, low, high):
        self.low  = low
        self.high = high

        return None


    def __call__(self, img):
        size_img_y, size_img_x = img.shape

        pos_y = np.random.randint(low = self.low, high = self.high)
        pos_x = int(pos_y * size_img_x / size_img_y)
        crop_orig = (pos_y, pos_x)
        crop_end  = (size_img_y, size_img_x)

        crop     = Crop(crop_orig, crop_end)
        img_crop = crop(img)

        img_zoom = resize(img_crop, (size_img_y, size_img_x), anti_aliasing = True)

        return img_zoom




def vflip(img): 
    return np.flip(img, axis = 0)




def hflip(img): 
    return np.flip(img, axis = 1)




def noise_poisson(img):
    # np.floa32 is a work around
    return np.float32(img + np.random.poisson(img))




def noise_gaussian(img, sigma):
    # np.floa32 is a work around
    return np.float32(img + sigma * np.random.randn(*img.shape))
