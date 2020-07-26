### Pytorch Image Classification Models

This directory contains code cloned from the [Pytorch Image Classification](https://github.com/hysts/pytorch_image_classification/) repository on GitHub.  It provided us with implementations of the baseline models we used during our research.  We selected it for this purpose based on its use in the CIFAR 10.1 research.  

A complete record of how we executed the models during our research can be found by reviewing the Jupyter notebooks in our [aws-notebooks](https://github.com/danzisar/w210-capstone/tree/master/aws-notebooks) directory of this repository.  

Slight modifications to the original code were made to enable us to train and test with our augmented datasets.  These have all been flagged in the source code with the identifer **W210** and are summarized below:

  - **./train.py**    
    Modifications were made to support CutMix's use of two labels for a single image 

  - **./pytorch_image_classification/utils/metrics.py**      
    Bespoke accuracy calculations for CutMix were added to handle CutMix's use of two labels for a single image

  - **./pytorch_image_classification/losses/__init__.py**   
    Bespoke loss calculations for CutMix were added to handle CutMix's use of two labels for a single image
       
  - **./pytorch_image_classification/datasets/datasets.py** 
    New class definitions that extend the torch Dataset class were created to allow us to import our training and test datasets as torchvision datasets.
    
  - **./pytorch_image_classification/collators/cutmix.py**
    Modifications were made to support CutMix 
    
  - **pytorch_image_classification/collators/__init__.py**   
    Modifications were made to support CutMix  
    
  - **pytorch_image_classification/transforms/__init__.py**
    Normalization values for each of the new datasets we created were added to this file