### Model Results

This directory contains exports of the results recorded during our trials.  

The naming convention used for the directories is significant.  It consists of up to three identifying elements:
  - Model Name
  - Optional:  Augmentation Indicator (e.g., ra_2_5, ra_3_20);  When *ra* appears, it indicates that the training dataset was augmented using the RandAugment Algorithm.  The following integers indicates the N and M hyperparameters for RandAugment:  the number of transformation that were applied sequentially and the magnitude of each transformation applied respectively.  When *cm* appears, it indicates that the training dataset was augmented using the CutMix Algorithm.  The following integers indicates the beta and alpha parameters for CutMix.
  - Optional:  *c10val* indicates that unaugmented data was used for the validation phase of epochs trained on augmented data.

As an example, the directory **resnet_basic_32_ra_2_20** contains results and outputs for the **resnet_basic_32** model, training on a dataset that had been augmented using the RandAugment Algorithm with an N parameter of 2 and an M parameter of 20. 

Within each directory, two types of files exist:  
 1. **predictions_XXX_refinedYY_DATASET.npz** - These files contain 5 keys that allow us to understand the model's performance:  preds, probs, labels, loss, and acc.  
    - The *XXX* in the name indicates the number of training epochs performed on the model making the predictions. 
    - The *refinedYY* in the name is an optional indicator (e.g., refined50).  This identifer is used to indicate that the model predictions were performed on a model that had been refined with unaugmented data for YY epochs.
    - The *DATASET* in the name is an optional indicator (e.g., CIFAR101 for CIFAR 10.1).  If no dataset name is supplied, the model predictions are for the CIFAR 10 testset.
    
 2. **results.csv**  - These files contain aggregate statistics on all of the predictions files in the directory (e.g., Accuracy, Loss)
