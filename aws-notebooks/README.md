### AWS Notebooks

This directory contains the Jupyter notebooks developed to AWS to train and test models.

Four models were the focus of our exploration:
1. [densenet_BC_100_12](https://arxiv.org/abs/1608.06993)	
2. [resnet_basic_32](https://arxiv.org/abs/1512.03385)
3. [resnext_29_4x64d](https://arxiv.org/abs/1611.05431)
4. [wrn_28_10](https://arxiv.org/abs/1605.07146)

The naming convention used for the notebooks is significant:
  - Model Name
  - Optional:  Augmentation Indicator (e.g., ra_2_5, ra_3_20);  When *ra* appears, it indicates that the training dataset was augmented using the RandAugment Algorithm.  The following integers indicates the N and M hyperparameters for RandAugment:  the number of transformation that were applied sequentially and the magnitude of each transformation applied respectively.  When *cm* appears, it indicates that the training dataset was augmented using the CutMix Algorithm.  The following integers indicates the beta and alpha parameters for CutMix.
  - Optional:  *c10val* indicates that unaugmented data was used for the validation phase of epochs trained on augmented data.

As an example, the file **resnet_basic_32_ra_2_20.ipynb** trained the model **resnet_basic_32** using a dataset that had been augmented using the RandAugment Algorithm with an N parameter of 2 and an M parameter of 20. 
