# Targeting the distribution gap by data augmentation

Summer 2020 Capstone
 
In 2019, researchers from UC Berkeley created the CIFAR 10.1 test dataset to assess how well current CIFAR classification models could generalize to a new test dataset.  Expecting to find diminishing accuracy drops on the new test set due to adaptive overfitting, they instead observed an accuracy drop that fit almost linearly between the original test set and the new test set;  an accuracy drop they attribute to a distribution shift in sampling during the test set creation process.  In our work, we explore using data augmentation as a regularization mechanism to improve model generalization and lessen this distribution gap.  Using the RandAugment and CutMix augmentation algorithms, we create five new datasets based on the CIFAR 10 training dataset, varying the severity of the augmentation, and use these to train a subset of the models that the original research benchmarked.  We then train these models at a reduced learning rate for 50 epochs with the original CIFAR 10 training dataset.  The accuracy of each model is then assessed against both the CIFAR 10 and CIFAR 10.1 test sets.   Running over 80 trials and expanding our experimental setup to test the impact of inserting augmentation at each phase of the pipeline (training, validation, test), our research shows a persistent distribution gap across experimental permutations.  These results lead us to believe that augmentation is not a viable solution for minimizing the distribution gap and, in fact, often worsens overall accuracy.
 
Contributors: Sarah Danzi, Jennifer Mahle, John Boudreaux

The repository is organized into the following subdirectories:
  * **analysis** contains artifacts that document the various lines of exploration, investigation, and analysis that the team performed on the datasets and our results.  
  * **aws-notebooks** contains the artifacts developed on AWS to execute the training and testing of models.
  * **data** contains the raw data we used in support of our research.
  * **dataset-generation** contains the utilities the team develoepd to generate augmented datasets.
  * **model_results** contains exports of the results recorded during our experimental trials.
  * **models** contains implementations of the models we used during our research.

#### Links
* [Final Report](https://docs.google.com/document/d/1B51_CuTtN47iD6n39JeM0KiFAwyoZMMRby2jW2rx5Ho/edit?usp=sharing)
* [Web Deliverable](https://www.ischool.berkeley.edu/courses/project-gallery/284/2/2020)
