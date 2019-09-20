# ImageNet Classification with Declarative Projection Nodes

Modified PyTorch ImageNet example code for testing declarative projection nodes.

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the ImageNet dataset and move validation images to labeled subfolders
    - To do this, you can use the following script: https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

## Training

To train a model, run `main.py` with the desired model architecture and the path to the ImageNet dataset:

```bash
python main.py --arch resnet18 --projection-type [PROJECTION_TYPE] --radius [RADIUS] --log-dir [LOG_DIR] [imagenet-folder]
```

This code can only be used on ResNet architectures; some minor modifications would be required to run on different architectures.

The strings available for PROJECTION_TYPE are {'L1S', 'L1B', 'L2S', 'L2B', 'LInfS', 'LInfB', ''} and correspond to the following Euclidean projections:
- L1S: L1-sphere
- L1B: L1-ball
- L2S: L2-sphere
- L2B: L2-ball
- LInfS: LInf-sphere
- LInfB: LInf-ball
- None: default, no projection

The default learning rate schedule starts at 0.1 and decays by a factor of 10 every 30 epochs.

For example, the following command can be used to train ResNet-18 from scratch with a radius-250 L1Sphere projection layer (before the final fully-connected layer and after a batch normalization layer without learnable affine parameters):

```bash
python3 main.py --arch resnet18 --gpu 0 --projection-type 'L1S' --radius 250.0 --log-dir [log directory] [imagenet-folder]
```

Further details (from the PyTorch example) are available at [this permalink](https://github.com/pytorch/examples/tree/ee964a2eeb41e1712fe719b83645c79bcbd0ba1a/imagenet).

## Usage

```
usage: main.py [-h] [--arch ARCH] [-j N] [--epochs N] [--start-epoch N] [-b N]
               [--lr LR] [--momentum M] [--weight-decay W] [--print-freq N]
               [--resume PATH] [-e] [--pretrained] [--world-size WORLD_SIZE]
               [--rank RANK] [--dist-url DIST_URL]
               [--dist-backend DIST_BACKEND] [--seed SEED] [--gpu GPU]
               [--multiprocessing-distributed]
               [--log-dir LOG_DIR] [--projection-type PROJECTION_TYPE]
               [--radius RADIUS]
               DIR

PyTorch ImageNet Training

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  --arch ARCH, -a ARCH  model architecture: alexnet | densenet121 |
                        densenet161 | densenet169 | densenet201 |
                        resnet101 | resnet152 | resnet18 | resnet34 |
                        resnet50 | squeezenet1_0 | squeezenet1_1 | vgg11 |
                        vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn | vgg19
                        | vgg19_bn (default: resnet18)
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 256), this is the total
                        batch size of all GPUs on the current node when using
                        Data Parallel or Distributed Data Parallel
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --weight-decay W, --wd W
                        weight decay (default: 1e-4)
  --print-freq N, -p N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --pretrained          use pre-trained model
  --world-size WORLD_SIZE
                        number of nodes for distributed training
  --rank RANK           node rank for distributed training
  --dist-url DIST_URL   url used to set up distributed training
  --dist-backend DIST_BACKEND
                        distributed backend
  --seed SEED           seed for initializing training.
  --gpu GPU             GPU id to use.
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N
                        processes per node, which has N GPUs. This is the
                        fastest way to use PyTorch for either single node or
                        multi node data parallel training
  --log-dir LOG_DIR     directory for logging loss and accuracy
  --projection-type PROJECTION_TYPE
                        Euclidean projection type {L1S, L1B, L2S, L2B, LInfS, LInfB, ''}
  --radius RADIUS       Lp-sphere or Lp-ball radius
```

## Links
- [PyTorch imagenet example repository](https://github.com/pytorch/examples/tree/ee964a2eeb41e1712fe719b83645c79bcbd0ba1a/imagenet)
- [ResNet paper](https://arxiv.org/pdf/1512.03385)
