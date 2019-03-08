'''
---- Disclaimer ----

This is the official code release for the paper titled -

"Reshaping Inputs for Convolutional Neural Networks - Some common and uncommon 
methods"

Please cite our work if you find this useful.

Copyright 2019, Swarnendu Ghosh, Nibaran Das, Mita Nasipuri, All rights reserved.

---- Prerequisite Packages/ API ----
Numpy 1.14.2, Scipy 1.0.0, Matplotlib 2.1.2

---- Description ----

The purpose of this code is to  provide a set of reshaping functions that can 
be used to reshape a single image or a batch of images organized in a defined 
way to specific shape. 

Each set of reshaping technique comes with its own set of pros and cons when it 
comes to training convolutional neural networks. This has been discussed in the 
above mentioned paper in details.

---- Features ----
Three functions has been provided for the purpose of reshaping images.

1 ) reshape_pil_image : For reshaping a PIL Image object. This function can be 
                        used as a lambda transform in Pytorch.
    Parameters :    img    = A PIL Image Object
                    mode   = The mode of reshaping
                    params = Necessary parameters for the selected mode
                    size   = The required size to which the image must be 
                             resized

2 ) reshape_single_image : For reshaping an image file.
    Parameters :    input_path  = Path to the image file
                    mode        = The mode of reshaping
                    params      = Necessary parameters for the selected mode
                    size        = The required size to which the image must be 
                                  resized
                    output_path = path for saving output image (Conditional)
                    save        = Boolean for saving output image (output_path 
                                  is necessary if True, default = False)
                    display     = Boolean for displaying output image 
                                  (default = False)
                    ret         = Boolean for returning output image 
                                  (default = False)
        
3 ) batch_reshape : For reshaping a batch of images distributed among class 
                    specific directories
                    e.g.    input_dir/class_1/image_1.png
                            input_dir/class_1/image_2.png
                            input_dir/class_1/image_3.png
                            ...
                            input_dir/class_2/image_1.png
                            input_dir/class_2/image_2.png
                            input_dir/class_2/image_3.png
                            ...
                    This structure can be easily used with the 
                    torchvision.datasets.ImageFolder class in Pytorch
                    
    Parameters :    input_dir = Directory with input images arranged into 
                                folders specific to classes 
                                (input_dir/class_name/)
                    output_dir = Directory where output images will be saved  
                    mode = The mode of reshaping
                    params = Necessary parameters for the selected mode
                    size = The required size to which the image must be resized  

PARAMETER DESCRIPTIONS

1 ) RESHAPING MODES: This is controlled by the 'mode' attribute in the above 
functions. The 'mode' attribute accepts a string as an input that can be one of 
the following

a) 'interp' -   For reshaping by interpolation method. Additionally it needs a 
                parameter for choosing the type of interpolation
b) 'tile' -     For reshaping by tiling method. Additionally it needs a 
                position parameter for choosing the location of the original 
                image in the reshaped image
c) 'mirror' -   For reshaping by mirroring method. Additionally it needs a 
                position parameter for choosing the location of the original 
                image in the reshaped image
d) 'crop' -     For reshaping by cropping method. Additionally it needs a 
                position parameter for choosing the location of the cropping 
                window with respect to the original image.
e) 'contain' -  For reshaping by containing method. Additionally it needs a 
                position parameter for choosing the location of the original 
                image in the reshaped image as well as the type of padding to 
                fill up the the extra space.

2 ) RESHAPING PARAMETERS: The reshaping parameters 'params' should be provided 
                          in the form of a list. The choice of parameters is 
                          dependent on the mode
   For interpolation        : params = [interpolation]
   For containing           : params = [position,padding]
   For tile, mirror or crop : params = [position]
   
  'interpolation' parameters : 'nearest','lanczos','bilinear','bicubic','cubic'
  'position' parameters : 'topleft', 'center', 'random'
  'padding' paramters : 'black', 'white', 'random', 'clone'

3 ) SIZE PARAMETER : The size parameter accepts a list in the form of 
                     [height,width].
'''

# IMPORTS
from scipy.misc import imread, imresize, imsave
import matplotlib.pyplot as plt
import numpy as np
import os

# PUBLIC FUNCTIONS

# RESHAPE SINGLE IMAGE

# FOR RESHAPING A PIL IMAGE OBJECT. THIS CAN BE USED AS PYTORCH IMAGE TRANSFORM
def reshape_pil_image(img, mode, params, size):
    img=img.convert('RGBA')
    img=img.convert('RGB')
    img = np.array(img)
    reshaped_image = __reshape_image(img, mode, [params, size])
    return reshaped_image

# FOR RESHAPING AN IMAGE FILE WHICH IS WHOSE PATH IS PROVIDED
# CHECK DESCRIPTION ABOVE FOR MORE DETAILS REGARDING MODE AND PARAMS
    
def reshape_single_image(input_path,        # PATH OF THE INPUT IMAGE
                         mode,              # MODE OF RESHAPING
                         params,            # RESHAPE PARAMETERS:MODE DEPENDANT
                         size,              # RESHAPED SIZE-[X,Y]
                         output_path=None,  # OUTPUT PATH FOR SAVING IMAGE
                         save=False,        # SAVE IMAGE FLAG-NEEDS OUTPUT PATH
                         display=False,     # DISPLAY IMAGE FLAG
                         ret=False):        # RETURN IMAGE FLAG
    input_image = imread(input_path)
    reshaped_image = __reshape_image(input_image,
                                     mode,
                                     [params, [size[0], size[1]]]
                                     )
    if save:
        imsave(output_path, reshaped_image)
    if display:
        fig = plt.figure()
        a = fig.add_subplot(1, 2, 1)
        a.set_title('Original')
        plt.imshow(input_image)
        a = fig.add_subplot(1, 2, 2)
        a.set_title(mode+'_'+'_'.join(params))
        plt.imshow(reshaped_image)
        plt.show()
    if ret:
        return(reshaped_image)

# RESHAPE BATCH OF IMAGES (DISTRIBUTED AMONG CLASS SPECIFIC DIRECTORIES)
# CHECK DESCRIPTION ABOVE FOR MORE DETAILS REGARDING MODE AND PARAMS
        
def batch_reshape(input_dir,    # INPUT DIR CONTAINING:CLASS_DIR/IMAGE_FILE
                  output_dir,   # OUTPUT DIR FOR RESHAPED IMAGES(SAME STRUCTURE)
                  mode,         # MODE OF RESHAPE
                  params,       # RESHAPE PARAMETERS (MODE DEPENDENT)
                  size):        # RESHAPED SIZE-[X,Y]
    classes = os.listdir(input_dir)
    for cls in classes:
        # print cls
        if not os.path.exists(output_dir+cls):
            os.makedirs(output_dir+cls)
        files = os.listdir(input_dir+cls)
        for f in files:
            # print file
            try:
                reshape_single_image(input_dir+cls+'/'+f,
                                     mode,
                                     params,
                                     size,
                                     output_path=output_dir+cls+'/'+f,
                                     save=True)
            except (Exception, err):
                print (cls,f)
                print (Exception, err)
                
# %% PRIVATE FUNCTIONS

# %% GENERIC RESHAPER (CALLS SPECIFIC MODES OF RESHAPING AS NEEDED)


def __reshape_image(image,      # IMAGE ARRAY
                    mode,       # RESHAPE MODE
                    params):    # RESHAPE PARAMETERS

    # RESHAPE BY INTERPOLATION,
    # MODE = 'interp',  PARAMS = [params_interp, [X, Y]]
    # e.g.: ['nearest'|'lanczos'|'bilinear'|'bicubic'|'cubic', [224, 224]]

    if (mode == 'interp'):
        reshaped_image = imresize(image, params[1], interp=params[0][0])

    # RESHAPE BY TILING,
    # MODE='tile' PARAMS=[params_position, [X, Y]]
    # e.g.: ['topleft'|'center'|'random', [224, 224]]

    if (mode == 'tile'):
        a = image.shape
        if params[0][0] == 'topleft':
            start_pos = 0
        if params[0][0] == 'center':
            start_pos = int(abs((a[1]-a[0])/2))
        if params[0][0] == 'random':
            a = image.shape
            if (a[1]==a[0]):
                start_pos=0
            else:
                start_pos = np.random.randint(0, abs(a[1]-a[0]))
        reshaped_image = imresize(__tile_image(image, start_pos), params[1])

    # RESHAPE BY MIRRORING,
    # MODE = 'mirror',  PARAMS = [params_position, [X, Y]]
    # e.g.: ['topleft'|'center'|'random', [224, 224]]

    if (mode == 'mirror'):
        a = image.shape
        if params[0][0] == 'topleft':
            start_pos = 0
        if params[0][0] == 'center':
            start_pos = int(abs((a[1]-a[0])/2))
        if params[0][0] == 'random':
            a = image.shape
            if (a[1]==a[0]):
                start_pos=0
            else:
                start_pos = np.random.randint(0, abs(a[1]-a[0]))
        reshaped_image = imresize(__mirror_image(image, start_pos), params[1])

    # RESHAPE BY CROPPING,
    # MODE='crop',  PARAMS=[params_position, [X, Y]]
    # e.g.: ['topleft'|'center'|'random', [224, 224]

    if (mode == 'crop'):
        a = image.shape
        if params[0][0] == 'topleft':
            start_pos = 0
        if params[0][0] == 'center':
            start_pos = int(abs((a[1]-a[0])/2))
        if params[0][0] == 'random':
            a = image.shape
            if (a[1]==a[0]):
                start_pos=0
            else:
                start_pos = np.random.randint(0, abs(a[1]-a[0]))
        reshaped_image = imresize(__crop_image(image, start_pos), params[1])

    # RESHAPE BY CONTAINING,
    # MODE='pad',  PARAMS=[[params_position, params_padding], [X, Y]]
    # e.g.: [[topleft|center|random, zero|random|clone], [224, 224]]
    if (mode == 'contain'):
        a = image.shape
        if params[0][0] == 'topleft':
            start_pos = 0
        if params[0][0] == 'center':
            start_pos = int(abs((a[1]-a[0])/2))
        if params[0][0] == 'random':
            a = image.shape
            if (a[1]==a[0]):
                start_pos=0
            else:
                start_pos = np.random.randint(0, abs(a[1]-a[0]))
        reshaped_image = imresize(__contain_image(image, start_pos, params[0][1]), params[1])
    return reshaped_image

# %% MODES

# TILE MODE


def __tile_image(image,         # IMAGE ARRAY
                 start_pos):    # WHERE TILING STARTS FROM
    a = image.shape
    if a[0] < a[1]:  # Landscape mode
        reshaped_image = np.zeros((a[1], a[1], 3))
        i = 0
        j = start_pos
        while(j != a[1]):  # Going forward from start_pos
            reshaped_image[j, :, :] = image[i % a[0], :, :]
            i = i+1
            j = j+1
        i = 0
        j = start_pos-1
        while(j != -1):  # Going backward from start_pos
            reshaped_image[j, :, :] = image[a[0]-i % a[0]-1, :, :]
            i = i+1
            j = j-1
    else:  # Portrait mode
        reshaped_image = np.zeros((a[0], a[0], 3))
        i = 0
        j = start_pos
        while(j != a[0]):  # Going forward
            reshaped_image[:, j, :] = image[:, i % a[1], :]
            i = i+1
            j = j+1
        i = 0
        j = start_pos-1
        while(j != -1):  # Going backward
            reshaped_image[:, j, :] = image[:, a[1]-i % a[1]-1, :]
            i = i+1
            j = j-1
    return reshaped_image

# MIRROR MODE


def __mirror_image(image,       # IMAGE ARRAY
                   start_pos):  # ROW/COL WHERE MIRRORING STARTS FROM
    start_pos=int(start_pos)
    a = image.shape
    if a[0] < a[1]:  # Landscape mode
        reshaped_image = np.zeros((a[1], a[1], 3))
        i = 0
        j = start_pos
        sign = 1
        dir = 'f'
        while(j != a[1]):  # Going forward from start_pos
            reshaped_image[j, :, :] = image[i, :, :]
            i = i+sign
            j = j+1
            if (i == a[0]-1 and dir == 'f'):
                sign = sign*(-1)
                dir = 'b'
            elif (i == 0 and dir == 'b'):
                sign = sign*(-1)
                dir = 'f'
        i = 0
        j = start_pos-1
        sign = 1
        dir = 'f'
        while(j != -1):  # Going backward from start_pos
            reshaped_image[j, :, :] = image[i, :, :]
            i = i+sign
            j = j-1
            if (i == a[0]-1 and dir == 'f'):
                sign = sign*(-1)
                dir = 'b'
            elif (i == 0 and dir == 'b'):
                sign = sign*(-1)
                dir = 'f'

    else:  # Portrait mode
        reshaped_image = np.zeros((a[0], a[0], 3))
        i = 0
        j = start_pos
        sign = 1
        dir = 'f'
        while(j != a[0]):  # Going forward from start_pos
            reshaped_image[:, j, :] = image[:, i, :]
            i = i+sign
            j = j+1
            if (i == a[1]-1 and dir == 'f'):
                sign = sign*(-1)
                dir = 'b'
            elif (i == 0 and dir == 'b'):
                sign = sign*(-1)
                dir = 'f'
        i = 0
        j = start_pos-1
        sign = 1
        dir = 'f'
        while(j != -1):  # Going backward from start_pos
            reshaped_image[:, j, :] = image[:, i, :]
            i = i+sign
            j = j-1
            if (i == a[1]-1 and dir == 'f'):
                sign = sign*(-1)
                dir = 'b'
            elif (i == 0 and dir == 'b'):
                sign = sign*(-1)
                dir = 'f'
    return reshaped_image

# CROP MODE

def __crop_image(image,         # IMAGE ARRAY
                 start_pos):    # ROW/COL WHERE CROPPING STARTS FROM
    a = image.shape
    start_pos=int(start_pos)
    if a[0] < a[1]:  # Landscape mode
        reshaped_image = np.zeros((a[0], a[0], 3))
        i = start_pos
        j = 0
        while(j != a[0]):
            reshaped_image[:, j, :] = image[:, i, :]
            i = i+1
            j = j+1
    else:  # Portrait mode
        reshaped_image = np.zeros((a[1], a[1], 3))
        i = start_pos
        j = 0
        while(j != a[1]):
            reshaped_image[j, :, :] = image[i, :, :]
            i = i+1
            j = j+1
    return reshaped_image

# CONTAIN MODE

def __contain_image(image,             # IMAGE ARRAY
                    start_pos,         # ROW/COL WHERE CONTAINED IMAGE BEGINS
                    padding_method):   # PADDING METHOD
    start_pos = int(start_pos)
    a = image.shape
    if padding_method == 'black':
        reshaped_image = np.zeros((max(a[0], a[1]), max(a[0], a[1]), 3))
    if padding_method == 'white':
        reshaped_image = np.full((max(a[0], a[1]), max(a[0], a[1]), 3), 255)
    if padding_method == 'random':
        reshaped_image = np.random.randint(256, size=(max(a[0], a[1]),
                                                      max(a[0], a[1]),
                                                      3))

    if a[0] < a[1]:  # Landscape mode
        if padding_method == 'clone':
            reshaped_image = np.zeros((a[1], a[1], 3))
            for i in range(start_pos):
                for j in range(max(a[0], a[1])):
                    reshaped_image[i, j, :] = image[np.random.randint(a[0]),
                                                    np.random.randint(a[1]),
                                                    :]
            for i in range(start_pos+a[0], a[1]):
                for j in range(max(a[0], a[1])):
                    reshaped_image[i, j, :] = image[np.random.randint(a[0]),
                                                    np.random.randint(a[1]),
                                                    :]
        reshaped_image[start_pos:start_pos+a[0], :, :] = image[:, :, :]
    else:            # Portrait mode
        if padding_method == 'clone':
            reshaped_image = np.zeros((a[0], a[0], 3))
            for i in range(max(a[0], a[1])):
                for j in range(start_pos):
                    reshaped_image[i, j, :] = image[np.random.randint(a[0]),
                                                    np.random.randint(a[1]),
                                                    :]
            for i in range(max(a[0], a[1])):
                for j in range(start_pos+a[1], a[0]):
                    reshaped_image[i, j, :] = image[np.random.randint(a[0]),
                                                    np.random.randint(a[1]),
                                                    :]
        reshaped_image[:, start_pos:start_pos+a[1], :] = image[:, :, :]
    return reshaped_image

