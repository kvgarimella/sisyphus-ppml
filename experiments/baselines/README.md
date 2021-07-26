## Baselines
This directory contains code for training and storing baseline models.

Run any python file with `-h` to see the necessary arguments.
```
usage: train_cifar10.py [-h] [--train-batch-size TRAIN_BATCH_SIZE]
                        [--test-batch-size TEST_BATCH_SIZE] [--epochs EPOCHS]
                        [--lr LR] [--momentum MOMENTUM]
                        [--weight-decay WEIGHT_DECAY] [--nesterov NESTEROV]
                        [--seed SEED] [--log-interval LOG_INTERVAL]
                        [--activation ACTIVATION] [--project PROJECT]
                        [--name NAME] [--model MODEL]

PyTorch CIFAR10 Baseline Training

optional arguments:
  -h, --help            show this help message and exit
  --train-batch-size TRAIN_BATCH_SIZE
                        training batch size
  --test-batch-size TEST_BATCH_SIZE
                        testing batch size
  --epochs EPOCHS       number of epochs
  --lr LR               learning rate
  --momentum MOMENTUM   SGD momentum value
  --weight-decay WEIGHT_DECAY
                        L2 weight decay
  --nesterov NESTEROV   nesterov
  --seed SEED           random seed
  --log-interval LOG_INTERVAL
                        logging interval
  --activation ACTIVATION
                        activation function
  --project PROJECT     wandb project
  --name NAME           wandb name
  --model MODEL         nn model name
```
