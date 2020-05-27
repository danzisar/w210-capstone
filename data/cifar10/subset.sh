#!/bin/bash

files=(/Users/sarahdanzi/Desktop/Berkeley/W210-Capstone/gitrepos/w210-capstone/data/cifar10/train/*.png)
n=${#files[@]}

for i in {1..300}
do
  file_to_retrieve="${files[RANDOM % n]}"
  cp $file_to_retrieve /Users/sarahdanzi/Desktop/Berkeley/W210-Capstone/gitrepos/w210-capstone/data/cifar10/subset300/.
done
