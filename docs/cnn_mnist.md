# cnn_mnist

https://www.tensorflow.org/tutorials/layers

## run on local machine

```
$ python -m tutorial.cnn_mnist --job-dir=/tmp/minst_2

$ tensorboard --logdir=/tmp/minst_2
```

## run on ml-engine

```
# config
BUCKET_NAME=stagem
JOB_NAME="cnn_mnist_$(date +%y%m%d_%H%M%S)_gpu"
OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME
REGION=us-central1

# run a training job in ml-engine
gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $OUTPUT_PATH \
    --runtime-version 1.5 \
    --python-version 3.5 \
    --module-name tutorial.cnn_mnist \
    --package-path tutorial/ \
    --scale-tier=BASIC_GPU \
    --region $REGION

gcloud ml-engine jobs describe $JOB_NAME

gcloud ml-engine jobs stream-logs $JOB_NAME

tensorboard --logdir=$OUTPUT_PATH
```

testing date : `2018-03-29`

| steps | loss (train) | loss (validation)  | accuracy   | scale tier | time 100 steps |
|      -|             -|                   -|           -|           -|               -|
| 2000  | 0.965334     | 0.87567109         | 0.81489998 | basic      | ~ 25.0 sec     |
| 2000  | 0.836681     | 0.5847652          | 0.8642     | basic_gpu  | ~  1.2 sec     |
| 30000 | 0.0514625    | 0.077157892        | 0.9752     | basic_gpu  | ~  1.2 sec     |

