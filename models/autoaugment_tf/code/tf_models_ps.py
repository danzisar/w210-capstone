# Launch SageMaker Training Jobs for Autoaugment Models
#
# This script accepts command line argument to launch SageMaker training jobs
# to train models from the TensorFlow research Autoaugment model collection.
#
# ref: https://github.com/tensorflow/models/tree/master/research/autoaugment
# 
# While, the source code in the repo can be used to run training jobs,
# this script was created to launch those same model training tasks but give
# a user the option to run distributed training in AWS SageMaker.
# 
# If workers is set to 1, this script trains a model with a single V100
# instance, and the parameter server gradient updates behave as they would
# using a single machine.
# 
# If workers is set to a value > 1, a number of V100 instances are launched
# to train a specified model using parameter servers to perform gradient
# updates.
#
# ==============================================================================

import argparse
import numpy as np
import os
import sagemaker
from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlow

# Set up command line argument parsing
parser = argparse.ArgumentParser()

parser.add_argument('--model_name', dest='model_name', type=str, help='autoaugment model to train')
parser.add_argument('--train_data', dest='train_data', type=str, help='specify dataset for model training') # e.g., 'cifar10_10k'
parser.add_argument('--workers', dest='workers', type=int, help='number of V100 worker instances; 1 indicates non-distributed training')

args = parser.parse_args()

# Check quality of arguments
valid_args = {'datasets': ['cifar10', 'cifar10_10k', 'cifar10_30k', 'cifar102', 'cifar102_30k'],
              'model_names': ['wrn', 'shake_shake_32', 'shake_shake_96', 'shake_shake_112', 'pyramid_net']}

if args.train_data not in valid_args['datasets']:
    parser.error('Invalid train_data parameter')

if args.model_name not in valid_args['model_names']:
    parser.error('Invalid model_name parameter')
    
if args.workers < 1:
    parser.error('Invalid number of workers')

if not args.model_name:
    parser.error('--model_name parameter is required')
elif not args.train_data:
    parser.error('--train_data parameter is required')
elif not args.workers:
    parser.error('--workers parameter is required')


# Set SageMaker session & execution role
bucket='sagemaker-may29'
sagemaker_session = sagemaker.Session(default_bucket=bucket)
role = get_execution_role()

# Set S3 path for data batches
inputs = 's3://' + bucket + '/sagemaker/{}'.format(args.train_data)

# Convert model_name and train_data to create S3 job name
job_name_params = {'model_name': {'wrn': 'wrn',
                                  'shake_shake_32': 'ss32',
                                  'shake_shake_96': 'ss96',
                                  'shake_shake_112': 'ss112',
                                  'pyramid_net': 'pyramid'},

                   'train_data': {'cifar10': 'cifar10',
                                  'cifar10_10k': 'cifar10-10k',
                                  'cifar10_30k': 'cifar10-30k',
                                  'cifar102': 'cifar102',
                                  'cifar102_30k': 'cifar102-30k'}}


# Set training job parameters
ps_instance_type = 'ml.p3.2xlarge' # 1xV100
ps_instance_count = args.workers

# SageMaker will upload model checkpoints to S3
# from this directory on a cluster instance
model_dir = "/opt/ml/model" 

distributions = {'parameter_server': {
                    'enabled': True}
                }

# Note: num_epochs is assigned in a conditional statement within train_cifar_ps.py
#       main block; it must be manually edited in that file, cannot be adjusted here

# these hyperparameter names correspond to the command line arguments
# in train_cifar_ps.py script; they are parsed in the main function
hyperparameters = {'model_name': args.model_name,
                   'checkpoint_dir': '/opt/ml/model',
                   'dataset': args.train_data}

estimator_ps = TensorFlow( base_job_name='{}-{}-{}'.format(job_name_params['model_name'][args.model_name],
                                                           job_name_params['train_data'][args.train_data],
                                                           str(args.workers)),
                           train_max_run=48 * 60 * 60,
                           source_dir='/home/ec2-user/SageMaker/w210-capstone/models/autoaugment_tf/code',
                           entry_point='train_cifar_ps.py',
                           role=role,
                           framework_version='1.13',
                           py_version='py2',
                           script_mode=True,
                           hyperparameters=hyperparameters,
                           train_instance_count=ps_instance_count,
                           train_instance_type=ps_instance_type, 
                           model_dir=model_dir,
                           distributions=distributions )
                           
# key: command line argument
# value: s3 location of file
remote_inputs = {'batches' : inputs+'/training'}

estimator_ps.fit(remote_inputs, wait=True)
