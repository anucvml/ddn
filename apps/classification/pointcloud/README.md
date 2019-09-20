# ModelNet40 Classification with Declarative Robust Pooling Nodes

Modified PyTorch PointNet code for testing declarative robust pooling nodes.

For example, to train PointNet from scratch with 60% outliers and Huber pooling replacing max pooling, use the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --epoch 60 --outlier_fraction 0.6 --robust_type 'H' --alpha 1.0
```

Point clouds of ModelNet40 models in HDF5 files will be automatically downloaded (416MB) to the data folder. Each point cloud contains 2048 points uniformly sampled from a shape surface. Each cloud is zero-mean and normalized into an unit sphere. There are also text files in data/modelnet40_ply_hdf5_2048 specifying the ids of shapes in h5 files.

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`

## Training

To train a model, run `main.py`:

```bash
python main.py TODO
```

The default learning rate schedule starts at 0.1 and decays by a factor of 10 every 30 epochs.

## Usage

```
```

Further details (from the yanx27/Pointnet_Pointnet2_pytorch repository) are copied below. See [this permalink](https://github.com/yanx27/Pointnet_Pointnet2_pytorch/tree/31deedb10b85ec30178df57a6389b2f326f7c970) for the PyTorch repository and 
[this permalink](https://github.com/charlesq34/pointnet/tree/539db60eb63335ae00fe0da0c8e38c791c764d2b) for the original TensorFlow repository.

# Pytorch Implementation of PointNet and PointNet++ 

This repo is implementation for [PointNet](http://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf) and [PointNet++](http://papers.nips.cc/paper/7095-pointnet-deep-hierarchical-feature-learning-on-point-sets-in-a-metric-space.pdf) in pytorch.

## Links
[Official PointNet](https://github.com/charlesq34/pointnet)
