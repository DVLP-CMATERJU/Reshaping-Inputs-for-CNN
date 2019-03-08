# -*- coding: utf-8 -*-
'''
---- Disclaimer ----

This is the official code release for the paper titled -

"Reshaping Inputs for Convolutional Neural Networks - Some common and uncommon 
methods"

Please cite our work if you find this useful.

Copyright 2019, Swarnendu Ghosh, Nibaran Das, Mita Nasipuri, All rights reserved.

---- Description ----

The purpose of this code is to  provide a demonstration as how to use the 
reshape_pil_image function in image_reshaper.py as a lambda function in 
torchvision.transforms. Additionally, a function to create a torchvision.transform 
object has also been provided that can be used to convert PIL Image objects to 
Pytorch Tensors. Moreover it can be used with torchvision.datasets.ImageFolder 
class to create a dataset object for reshaping images on the fly and feeding into
CNNs through a DataLoader object. Demonstration regarding all the above cases
has been shown below.

Each set of reshaping technique comes with its own set of pros and cons when it 
comes to training convolutional neural networks. This has been discussed in the 
above mentioned paper in details.

---- Prerequisite Packages/API ----
Numpy 1.14.2, PIL 5.0.0, Scipy 1.0.0, Matplotlib 2.1.2, Pytorch 0.4.0

---- Features ----
The reshape_pil_image can be used as a lambda function.

1 ) reshape_pil_image : For reshaping a PIL Image object. This function can be 
                        used as a lambda transform in Pytorch.
    Parameters :    img    = A PIL Image Object
                    mode   = The mode of reshaping
                    params = Necessary parameters for the selected mode
                    size   = The required size to which the image must be 
                             resized

2) get_transform : This function has been provided to create a 
                   torchvision.transforms object for using in Pytorch
                   environment.
   Parameters :    mode   = The mode of reshaping
                   params = Necessary parameters for the selected mode
                   size   = The required size to which the image must be 
                            resized
                  
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
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.models as models
import image_reshaper
from PIL import Image

# AVAILABLE MODES AND PARAMETERS. CHECK DESCRIPTION FOR MORE DETAILS
modes = ['interp', 'tile', 'mirror', 'crop', 'contain']
params_interp = ['nearest', 'lanczos', 'bilinear',  'bicubic',  'cubic']
params_position = ['topleft', 'center', 'random']
params_padding = ['black', 'white', 'random', 'clone']

mode = modes[0]
params = [params_interp[1]]                            # If mode = 'interp'
# params = [params_position[1]]                        # If mode = 'tile', 'mirror', or 'crop'
# params = [params_position[1],params_padding[0]]      # If mode = 'contain'
size = [224,224]



# A FUNCTION TO CREATE A torchvision.transforms OBJECT USING LAMBDA TRANSFORMS
def get_transform(mode, params, size):
    transform = transforms.Compose([
        transforms.Lambda(lambda img: image_reshaper.reshape_pil_image(img, mode, params, size)),
        transforms.ToTensor()
        ]) 
    return transform

# CREATING A TRANSFORM OBJECT
transform = get_transform(mode,params,size)

# USING THE TRANSFORM OBJECT TO GET A PYTORCH TENSOR FROM A PIL OBJECT
image_path = 'samples/cats/cat2.jpg'
pil_object = Image.open(image_path)
image_tensor = transform(pil_object)
print('TYPE =', type(image_tensor),',SIZE =',image_tensor.size())


# USING THE TRANSFORM OBJECT WITH torchvision.datasets.ImageFolder CLASS
# TO PROVIDE INPUT TO A CNN MODEL THROUGH A DATALOADER

dataset = datasets.ImageFolder('samples/',transform = transform)                # Creating Dataset Object
dataloader = data.DataLoader(dataset,batch_size=4)                              # Creating DataLoader Object
model = models.resnet18()                                                       # A ResNet Model with 18 weight layers
for batch in dataloader:
    image,label = batch
    out = model(image)                                                          # Forward Pass
    print('INPUT SIZE =', image.size())
    print('OUTPUT SIZE (ResNet18) =', out.size())
