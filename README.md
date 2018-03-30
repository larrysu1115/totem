# totem

## run cnn_minst.py

```
python -m tutorial.cnn_mnist --job-dir=/tmp/minst_2

MODEL_DIR=/tmp/mnist_convnet_model

gcloud ml-engine local train \
    --module-name model.tuto.cnn_minst \
    --package-path model/ \
    --job-dir $MODEL_DIR

gsutil rm -r gs://bk_tina/mnist_1

BUCKET_NAME=stagem    
JOB_NAME=mnist_6
OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME
REGION=us-central1

gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $OUTPUT_PATH \
    --runtime-version 1.5 \
    --python-version 3.5 \
    --module-name tutorial.cnn_mnist \
    --package-path tutorial/ \
    --region $REGION

 tensorboard --logdir=$OUTPUT_PATH

gcloud ml-engine jobs describe mnist_2
```