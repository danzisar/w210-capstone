### Dataset Generation 

This directory contains the utilities the team develoepd to generate augmented datasets.  Two approaches were used:
 
 1. [RandAugment](https://arxiv.org/abs/1909.13719)
 2. [CutMix](https://arxiv.org/abs/1905.04899)

In both cases, slight modifications to the original code had to be made to develop static datasets.  The methods were originally developed for dynamic augmentation of datasets --- allowing the model to generate augmented batches as it trains.  However, in the interest of running an experiment to compare results across models and model runs, it was important to the team that the same augmented training dataset be supplied to all models to ensure apples-to-apples comparisons.
