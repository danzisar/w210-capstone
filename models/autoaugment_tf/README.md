# Autoaugment Models

### Quick Links

[Source Code Details](#changes)

[Training Details](#training-procedure)

[How to Run](#how-to-run)

The Wide-ResNet, Shake-Shake and ShakeDrop models with AutoAugment are some of the best-performing neural network architectures with the CIFAR-10 dataset. The original source code for these model implementations can be found here:

https://github.com/tensorflow/models/tree/master/research/autoaugment

As part of the The CIFAR-10.1 research, these models were trained and tested on CIFAR-10 achieving the following results:

**Benchmark Performance on CIFAR-10**

<center>

| Model                       | Train Set | Test Set | Accuracy |
| --------------------------- | --------- | -------- | -------- |
| autoaug_pyramid_net_tf      | CIFAR-10  | CIFAR-10 | 98.4     |
| autoaug_shake_shake_112_tf  | CIFAR-10  | CIFAR-10 | 98.1     |
| autoaug_shake_shake_96_tf   | CIFAR-10  | CIFAR-10 | 98.0     |
| autoaug_wrn_tf              | CIFAR-10  | CIFAR-10 | 97.5     |
| autoaug_shake_shake_32_tf   | CIFAR-10  | CIFAR-10 | 97.3     |

</center>

<h2 id="changes">
Changes to Original Source Code
</h2>

**Note:** minimal changes to the original code were required, merely to enable the optional training/testing on new datasets

The original source code, referenced in the link above, is designed to accept command line arguments and run training jobs for each model type with either CIFAR-10 or CIFAR-100 datasets. Specifically, the `data_utils.py` script has a `DataSet` class that is used to load CIFAR-10 training and test data, from the same directory, in Python batch format.

### data_utils.py

* In order to load _new_ training and test sets, the models trained in this research use a modified version of the original `data_utils.py`. The modified code is designed to recognize new dataset names in order to load the intended input data according to their batch names and the corresponding number of examples.

* **New dataset names:**

    * `cifar10_10k`: 10,000 subsample of original CIFAR-10; 8,000 train/2,000 test

    * `cifar10_30k`: 30,000 subsample of original CIFAR-10; 24,000 train/6,000 test

    * `cifar102`: 10,000 example dataset with mixture of new and original CIFAR-10 training images; 8,000 train/2,000 test

    * `cifar102_30k`: 30,000 example dataset with mixture of new and original CIFAR-10 training images; 24,000 train/6,000 test
    
**Note 1:** these dataset names occur in `data_utils.py`, `tf_models_ps.py`, and `train_cifar_ps.py`. A change in one location will require a change in all.

**Note 2:** while `data_utils.py` defines the physical shape of the input data, the `_setup_images_and_labels` method of the `CifarModel` class in `train_cifar_ps.py` defines the number of classes. This is because the original source code was designed for either CIFAR-10 or CIFAR-100. In our case all models are for classification of 10-classes, and the code has been modified to select such an architecture for all new dataset names.

* **Data Preprocessing**

   * Dataset preprocessing in the original code involved subtracting training dataset channel means and dividing by training dataset channel standard deviations. These values, for the originally intended datasets, are loaded from hard-coded variables provided from `augmentation_transforms.py`. In our approach, training dataset channel means and channel standard deviations for a specified training set are calculated when the training code instantiates an instance of the `DataSet` class.

### helper_utils.py

* `steps_per_epoch`: modified to be a value that is divided by the number of cluster workers; if more than one worker is assigned to    the training job, this modification speeds up training at the expense of potentially requiring additional epochs to converge to the same result as a single GPU training job.

<h2 id="training-procedure">
Training Environment
</h2>

* All models were trained via AWS SageMaker training jobs and ml.p3.2xlarge (1xTesla V100 GPU) instance types. Source data was saved in S3 buckets as pickle files in protocol 2 (Python 2). Since the code expects all data batches in the same location (folder), both training and testing batches for the same dataset were saved to the same S3 bucket path.

* Training jobs were launched from a lightweight ml.t2.medium instance

## Training Procedure

For training tasks in our CIFAR-10.2 research the main training script `tf_models_ps.py` was created to launch SageMaker model training jobs for the TensorFlow AutoAugment models.

This script receives command line arguments and in turn provides arguments that are needed for `train_cifar_ps.py`. The `ps` in the filenames refers to the optional distributed appraoch: parameter servers.

To launch a training job, this script accepts a `workers` argument that will optionally run distributed training in AWS SageMaker. If `workers` is set to `1`, the model trains on a single GPU and the gradient updates are synchronous on that single GPU. If `workers` is set to a value greater than `1`, SageMaker will spin up a cluster of workers and utilize parameter servers to perform asynchronous gradient updates.

**Note:** while distributed approaches sped up training time, some models did not converge to their previously published accuracy given default hyperparameter values. Specifically, without increasing the number of epochs, Wide-ResNet and Pyramid Net models underperformed published accuracy by ~0.5% each. All other models were within 0.2% of published accuracy using distributed learning and default hyperparameters. 

<h2 id="how-to-run">
Steps to Use This Code
</h2>

1. Store data in an S3 bucket
2. Create a lightweight SageMaker instance with access to the bucket
3. Ensure source code in `tf_models_ps.py`:

   a. Uses your bucket name

   b. Points to correct directories in S3 (e.g., both training and testing batches should reside within `s3://{bucket_name}/data/{dataset_name}/training`)

4. Ensure source code for `datafiles` variable in `data_utils.py` matches filename titles for your train and test batches 

5. Start a training job from the SageMaker notebook instance command line prompt:

   - **ex:** `python tf_models_ps.py --model_name shake_shake_112 --train_data cifar102_30k --workers 1`

6. When training completes trained models will be uploaded to an S3 bucket directory that is created according to the `base_job_name` parameter.

   - **ex:** `ss112-cifar10-1-2020-04-15-15-38-14-319`
      - **model:** shake_shake_112
      - **train_data:** cifar10
      - **workers:** 1




