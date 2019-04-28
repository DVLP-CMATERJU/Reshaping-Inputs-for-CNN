# Reshaping Inputs for Convolutional Neural Networks - Some common and uncommon methods

This is the official code release for the paper titled -

**_"Reshaping Inputs for Convolutional Neural Networks - Some common and uncommon methods"_**

Please cite our work if you find this useful.

```html
# BIBTEX ENTRY
@article{ghosh2019reshaping,
  title={Reshaping Inputs for Convolutional Neural Networks-Some common and uncommon methods},
  author={Ghosh, Swarnendu and Das, Nibaran and Nasipuri, Mita},
  journal={Pattern Recognition},
  year={2019},
  publisher={Elsevier}
}
```
Link to paper : https://www.sciencedirect.com/science/article/abs/pii/S0031320319301505

Alternative Link : https://www.researchgate.net/publication/332320967_Reshaping_Inputs_for_Convolutional_Neural_Networks_-Some_common_and_uncommon_methods

*Copyright 2019, Swarnendu Ghosh, Nibaran Das, Mita Nasipuri, All rights reserved.*

## Abstract of the Paper
Convolutional Neural Network has become very common in the field of computer vision in recent years. But it comes with a severe restriction regarding the size of the input image. Most convolutional neural networks are designed in a way so that they can only handle images of a consistent size. This creates several challenges during data acquisition and model deployment. The common practice to overcome this limitation is to reshape the input images so that they can be fed into the networks. Many standard pre-trained networks and datasets come with a provision of working with square images. Hence in this work we analyze 25 different reshaping methods across 6 datasets corresponding to different domains trained on three famous architectures namely Inception-V3, which is an extension of GoogLeNet, the Residual Networks (ResNet-18) and the 121-Layer deep DenseNet. While some of the methods have been commonly used with convolutional neural networks, some uncommon techniques have also been suggested. In total, 450 neural networks were trained from scratch to provide various analyses regarding the convergence of the validation loss and the accuracy obtained on the test data. Statistical measures have been provided to demonstrate the dependence between parameter choices and datasets. The paper intends to guide the reader to choose a proper technique of reshaping inputs for their convolutional neural networks.

## Repository Descriptions
1. **'samples/'** - It is a directory with 4 sample images of dogs and cats sorted in respective class specific directories  
2. **image_reshaper.py** - The purpose of this code is to provide a set of reshaping functions that can be used to 
   reshape a single image or a batch of images organized in a defined way to specific shape.  
3. **functions_demo.py** - The purpose of this code is to  provide a demonstration of using the functions provided 
   in *image_reshaper.py*.  
   Three functions has been provided for the purpose of reshaping images, namely,    
   * reshape_pil_image : For reshaping a PIL Image object. This function can be used as a lambda transform in Pytorch.
   * reshape_single_image : For reshaping an image file from a path.
   * batch_reshape : For reshaping a batch of images distributed among class specific directories
4. **pytorch_demo.py** - The purpose of this code is to  provide a demonstration as how to use the *reshape_pil_image* function in *image_reshaper.py* as a lambda function in torchvision.transforms. Additionally, a function to create a  torchvision.transform object has also been provided that can be used to convert PIL Image objects to Pytorch Tensors. Moreover, it can be used with torchvision.datasets.ImageFolder class to create a dataset object for reshaping images on the fly and feeding into CNNs through a DataLoader object.

### External Packages Required
* Numpy 1.14.2
* PIL 5.0.0
* Scipy 1.0.0
* Matplotlib 2.1.2
* Pytorch 0.4.0

## Reshaping Methods Provided
This is controlled by the 'mode' attribute in the provided functions. The 'mode' attribute accepts a string as an input that can be one of the following
1. **'interp'**  - For reshaping by interpolation method.               
                   Additionally it needs a parameter for choosing the type of interpolation
2. **'tile'**    - For reshaping by tiling method.  
                   Additionally it needs a position parameter for choosing the location of the original image in the reshaped image
3. **'mirror'**  - For reshaping by mirroring method.
                   Additionally it needs a position parameter for choosing the location of the original image in the reshaped image
4. **'crop'**    - For reshaping by cropping method.  
                   Additionally it needs a position parameter for choosing the location of the cropping window with respect to the original image.
5. **'contain'** - For reshaping by containing method.  
                   Additionally it needs a position parameter for choosing the location of the original image in the reshaped image as well as the type of padding to fill up the the extra space.
## Parameters corresponding to various Reshaping Methods
The reshaping parameters 'params' should be provided in the form of a list.  
The choice of parameters is dependent on the mode

 * *For interpolation*        : params = \['interpolation'\]
 * *For containing*           : params = \['position','padding'\]
 * *For tile, mirror or crop* : params = \['position'\]
 
 Here 'interpolation','position' and 'padding' can take the following values  
 
 * *'interpolation' parameters* : 'nearest','lanczos','bilinear','bicubic','cubic'
 * *'position' parameters*      : 'topleft', 'center', 'random'
 * *'padding' paramters*        : 'black', 'white', 'random', 'clone'

The 'size' parameter accepts a list in the form of \[height,width\].

