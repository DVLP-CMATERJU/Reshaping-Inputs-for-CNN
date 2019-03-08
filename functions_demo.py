# -*- coding: utf-8 -*-

'''
---- Disclaimer ----

This is the official code release for the paper titled -

"Reshaping Inputs for Convolutional Neural Networks - Some common and uncommon 
methods"

Please cite our work if you find this useful.

Copyright 2019, Swarnendu Ghosh, Nibaran Das, Mita Nasipuri, All rights reserved.

---- Description ----

The purpose of this code is to  provide a demonstration of using the functions
provided in image_reshaper.py. 

Each set of reshaping technique comes with its own set of pros and cons when it 
comes to training convolutional neural networks. This has been discussed in the 
above mentioned paper in details.

---- Prerequisite Packages/API ----
Numpy 1.14.2, PIL 5.0.0, Scipy 1.0.0, Matplotlib 2.1.2

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

import os
import image_reshaper
from PIL import Image

# AVAILABLE MODES AND PARAMETERS
modes = ['interp', 'tile', 'mirror', 'crop', 'contain']
params_interp = ['nearest', 'lanczos', 'bilinear',  'bicubic',  'cubic']
params_position = ['topleft', 'center', 'random']
params_padding = ['black', 'white', 'random', 'clone']

mode = modes[0]
params = [params_interp[1]]                            # If mode = 'interp'
# params = [params_position[1]]                        # If mode = 'tile', 'mirror', or 'crop'
# params = [params_position[1],params_padding[0]]      # If mode = 'contain'
size = [256,256]

# RESHAPING A SINGLE IMAGE FILE FROM A PATH
if not os.path.exists('outputs/'):
    os.makedirs('outputs/')
image_path = 'samples/cats/cat2.jpg'
output_path = 'outputs/'+image_path.split('/')[-1]
returned_image = image_reshaper.reshape_single_image(image_path,                    # Path to the image file
                                                     mode,                          # Mode of reshaping
                                                     params,                        # Parameters corresponding reshaping technique
                                                     size,                          # Size of reshaping
                                                     output_path = output_path,     # Path to save reshaped image. Required if 'save' == True
                                                     save = True,                   # For saving image in the output path
                                                     display = True,                # For displaying image as pyplot
                                                     ret = True)                    # For returning the image as a numpy array
print(returned_image.shape)


# RESHAPING A BATCH OF IMAGES DISTRIBUTED IN CLASS SPECIFIC DIRECTORIES
if not os.path.exists('outputs/'):
    os.makedirs('outputs/')
input_dir = 'samples/'
output_dir = 'outputs/'
image_reshaper.batch_reshape(input_dir,                                             # Directory of input images organized in class specific directories
                             output_dir,                                            # Directory for saving output images in the same directory structure
                             mode,                                                  # Mode of reshaping       
                             params,                                                # Parameters corresponding reshaping technique
                             size)                                                  # Size of reshaping


# RESHAPING A PIL IMAGE OBJECT
image_path = 'samples/cats/cat2.jpg'
pil_object = Image.open(image_path)
output_image = image_reshaper.reshape_pil_image(pil_object,                         # PIL Image Object for reshaping
                                                mode,                               # Mode of reshaping  
                                                params,                             # Parameters corresponding reshaping technique
                                                size)                               # Size of reshaping
print(type(output_image), output_image.shape)