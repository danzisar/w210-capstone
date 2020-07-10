### AWS Notebooks

This directory contains the Jupyter notebooks developed to AWS to train and test models.

Four models were the focus of our exploration:
1. [densenet_BC_100_12](https://arxiv.org/abs/1608.06993)	
2. [resnet_basic_32](https://arxiv.org/abs/1512.03385)
3. [resnext_29_4x64d](https://arxiv.org/abs/1611.05431)
4. [wrn_28_10](https://arxiv.org/abs/1605.07146)

The naming convention used for the notebooks is significant:
  - Model Name
  - Optional:  Augmentation Indicator (e.g., ra_2_5, ra_3_20)
  - Optional:  'c10val' indicates that unaugmented data was used for the validation phase of epochs trained on augmented data.
