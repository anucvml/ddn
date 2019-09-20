# ModelNet40 Classification with Declarative Robust Pooling Nodes

Modified PyTorch PointNet code for testing declarative robust pooling nodes.

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`

## Training

To train a model, run `main.py`:

```bash
python3 main.py --outlier_fraction [OUTLIER_FRACTION] --robust_type [ROBUST_TYPE] --alpha [ALPHA]
```

The strings available for ROBUST_TYPE are {'Q', 'PH', 'H', 'W', 'TQ', ''} and correspond to the following penalty functions:
- Q: quadratic
- PH: pseudo-Huber
- H: Huber
- W: Welsch
- TQ: truncated quadratic
- None: default, max-pooling

The default number of epochs is 60 and the learning rate starts at 0.01 and decays by a factor of 2 every 20 epochs.

For example, to train PointNet from scratch on GPU 0 with 60% outliers and Huber pooling replacing max pooling, use the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --outlier_fraction 0.6 --robust_type 'H' --alpha 1.0
```

Point clouds of ModelNet40 models in HDF5 files will be automatically downloaded (416MB) to the data folder. Each point cloud contains 2048 points uniformly sampled from a shape surface. Each cloud is zero-mean and normalized into an unit sphere. There are also text files in data/modelnet40_ply_hdf5_2048 specifying the ids of shapes in h5 files.

## Usage

```
usage: PointNet [-h] [--batchsize BATCHSIZE] [--epoch EPOCH]
                [--learning_rate LEARNING_RATE] [--train_metric]
                [--optimizer OPTIMIZER] [--pretrain PRETRAIN]
                [--decay_rate DECAY_RATE] [--rotation ROTATION]
                [--model_name MODEL_NAME] [--input_transform INPUT_TRANSFORM]
                [--feature_transform FEATURE_TRANSFORM] [-e]
                [--outlier_fraction OUTLIER_FRACTION]
                [--robust_type ROBUST_TYPE] [--alpha ALPHA]

optional arguments:
  -h, --help            show this help message and exit
  --batchsize BATCHSIZE
                        batch size in training
  --epoch EPOCH         number of epoch in training
  --learning_rate LEARNING_RATE
                        learning rate in training
  --train_metric        whether to evaluate on training dataset
  --optimizer OPTIMIZER
                        optimizer for training
  --pretrain PRETRAIN   whether to use pretrained model
  --decay_rate DECAY_RATE
                        decay rate of learning rate
  --rotation ROTATION   range of training rotation
  --model_name MODEL_NAME
                        model to use
  --input_transform INPUT_TRANSFORM
                        use input transform in pointnet
  --feature_transform FEATURE_TRANSFORM
                        use feature transform in pointnet
  -e, --evaluate        evaluation on test set only
  --outlier_fraction OUTLIER_FRACTION
                        fraction of data that is outliers
  --robust_type ROBUST_TYPE
                        use robust pooling {Q, PH, H, W, TQ, ''}
  --alpha ALPHA         robustness parameter
```

Further details (from the yanx27/Pointnet_Pointnet2_pytorch repository) are available at [this permalink](https://github.com/yanx27/Pointnet_Pointnet2_pytorch/tree/31deedb10b85ec30178df57a6389b2f326f7c970) for the PyTorch repository and 
[this permalink](https://github.com/charlesq34/pointnet/tree/539db60eb63335ae00fe0da0c8e38c791c764d2b) for the original TensorFlow repository.

## Links
- [Official PointNet repository](https://github.com/charlesq34/pointnet)
- [PointNet paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf)
