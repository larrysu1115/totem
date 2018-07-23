# cnn_mnist

https://www.tensorflow.org/tutorials/layers

## run on local machine

```
$ python -m cifar10.cifar10_train --job-dir=/tmp/cifar10_a

$ tensorboard --logdir=/tmp/cifar10_a
```

## run on ml-engine

```
# config
BUCKET_NAME=stagem
JOB_NAME="cifar10_$(date +%y%m%d_%H%M%S)_gpu"
OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME
REGION=us-central1

# run a training job in ml-engine
gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $OUTPUT_PATH \
    --runtime-version 1.5 \
    --python-version 3.5 \
    --module-name cifar10.cifar10_train \
    --package-path cifar10/ \
    --scale-tier=BASIC_GPU \
    --region $REGION

gcloud ml-engine jobs describe $JOB_NAME

gcloud ml-engine jobs stream-logs $JOB_NAME

tensorboard --logdir=$OUTPUT_PATH
```

testing date : `2018-03-30`


| scale tier | batch size | sec/batch | examples/sec | steps   | loss
|           -|           -|          -|             -|        -|
| local      | 128        | 0.3       |              |         |
| basic_gpu  | 128        | 0.031     | 4100         | 100,000 | 

