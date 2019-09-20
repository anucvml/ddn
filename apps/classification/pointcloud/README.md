# ModelNet40 Classification with Declarative Robust Pooling Nodes

Modified PyTorch PointNet code for testing declarative robust pooling nodes.

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`

## Training

To train a model, run `main.py`:

```bash
python3 main.py --outlier_fraction [OUTLIER_FRACTION] --robust_type [PHI] --alpha [ALPHA]
```

The default number of epochs is 60 and the learning rate starts at 0.01 and decays by a factor of 2 every 20 epochs.

For example, to train PointNet from scratch on GPU 0 with 60% outliers and Huber pooling replacing max pooling, use the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --outlier_fraction 0.6 --robust_type 'H' --alpha 1.0
```

Point clouds of ModelNet40 models in HDF5 files will be automatically downloaded (416MB) to the data folder. Each point cloud contains 2048 points uniformly sampled from a shape surface. Each cloud is zero-mean and normalized into an unit sphere. There are also text files in data/modelnet40_ply_hdf5_2048 specifying the ids of shapes in h5 files.

## Usage

ToDo
```
    parser.add_argument('--batchsize', type=int, default=24, help='batch size in training')
    parser.add_argument('--epoch',  default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training, ignored for SGD')
    parser.add_argument('--train_metric', action='store_true', help='whether evaluate on training dataset')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer for training')
    parser.add_argument('--pretrain', type=str, default=None,help='whether use pretrain model')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate of learning rate')
    parser.add_argument('--rotation',  default=None, help='range of training rotation')
    parser.add_argument('--model_name', default='pointnet2', help='range of training rotation')
    parser.add_argument('--input_transform', default=False, help="use input transform in pointnet")
    parser.add_argument('--feature_transform', default=False, help="use feature transform in pointnet")
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help="evaluation on test set only")
    parser.add_argument('--outlier_fraction', type=float, default=0, help='fraction of data that is outliers')
    parser.add_argument('--robust_type', dest='robust_type', type=str, default='', help="use robust pooling {Q, PH, H, W, TQ, ''}")
    parser.add_argument('--alpha', dest='alpha', type=float, default=1.0, help="robustness parameter")
```

Further details (from the yanx27/Pointnet_Pointnet2_pytorch repository) are available at [this permalink](https://github.com/yanx27/Pointnet_Pointnet2_pytorch/tree/31deedb10b85ec30178df57a6389b2f326f7c970) for the PyTorch repository and 
[this permalink](https://github.com/charlesq34/pointnet/tree/539db60eb63335ae00fe0da0c8e38c791c764d2b) for the original TensorFlow repository.

## Links
- [Official PointNet repository](https://github.com/charlesq34/pointnet)
- [PointNet paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf)
