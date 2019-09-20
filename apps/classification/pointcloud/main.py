# MODELNET40 CLASSIFICATION WITH DECLARATIVE ROBUST POOLING NODES
# Dylan Campbell <dylan.campbell@anu.edu.au>
# Stephen Gould <stephen.gould@anu.edu.au>
#
# Modified from PyTorch PointNet code:
# https://github.com/yanx27/Pointnet_Pointnet2_pytorch/tree/31deedb10b85ec30178df57a6389b2f326f7c970
# with dataset download code from the charlesq34/pointnet repository:
# https://github.com/charlesq34/pointnet/blob/539db60eb63335ae00fe0da0c8e38c791c764d2b/provider.py
# and with mean average precision code adapted from:
# https://github.com/rbgirshick/py-faster-rcnn/blob/781a917b378dbfdedb45b6a56189a31982da1b43/lib/datasets/voc_eval.py

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from ModelNetDataLoader import ModelNetDataLoader, load_data
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
from utils import test, save_checkpoint
from pointnet import PointNetCls, feature_transform_regularizer

np.random.seed(2809)

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batchsize', type=int, default=24, help='batch size in training')
    parser.add_argument('--epoch',  default=60, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--train_metric', action='store_true', help='whether evaluate on training dataset')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer for training')
    parser.add_argument('--pretrain', type=str, default=None,help='whether use pretrain model')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate of learning rate')
    parser.add_argument('--rotation',  default=None, help='range of training rotation')
    parser.add_argument('--model_name', default='pointnet', help='model to use')
    parser.add_argument('--input_transform', default=False, help="use input transform in pointnet")
    parser.add_argument('--feature_transform', default=False, help="use feature transform in pointnet")
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help="evaluation on test set only")
    parser.add_argument('--outlier_fraction', type=float, default=0, help='fraction of data that is outliers')
    parser.add_argument('--robust_type', dest='robust_type', type=str, default='', help="use robust pooling {Q, PH, H, W, TQ, ''}")
    parser.add_argument('--alpha', dest='alpha', type=float, default=1.0, help="robustness parameter")
    return parser.parse_args()

def main():
    # Download dataset for point cloud classification
    modelnet_dir = 'modelnet40_ply_hdf5_2048'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, modelnet_dir)):
        www = 'https://shapenet.cs.stanford.edu/media/' + modelnet_dir + '.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))

    datapath = './data/' + modelnet_dir + '/'

    args = parse_args()

    if args.robust_type == 'Q':
        type_string = 'quadratic'
        outlier_string = 'outliers_' + str(args.outlier_fraction)
    elif args.robust_type == 'PH':
        type_string = 'pseudohuber'
        outlier_string = 'outliers_' + str(args.outlier_fraction)
    elif args.robust_type == 'H':
        type_string = 'huber'
        outlier_string = 'outliers_' + str(args.outlier_fraction)
    elif args.robust_type == 'W':
        type_string = 'welsch'
        outlier_string = 'outliers_' + str(args.outlier_fraction)
    elif args.robust_type == 'TQ':
        type_string = 'truncatedquadratic'
        outlier_string = 'outliers_' + str(args.outlier_fraction)
    else:
        type_string = 'max'
        outlier_string = 'outliers_' + str(args.outlier_fraction)

    if args.rotation is not None:
        ROTATION = (int(args.rotation[0:2]),int(args.rotation[3:5]))
    else:
        ROTATION = None

    '''CREATE DIRS'''
    experiment_dir = Path('./tests/')
    if not experiment_dir.exists():
        experiment_dir.mkdir()
    type_dir = Path(str(experiment_dir) + '/' + type_string + '/')
    if not type_dir.exists():
        type_dir.mkdir()
    outlier_dir = Path(str(type_dir) + '/' + outlier_string + '/')
    if not outlier_dir.exists():
        outlier_dir.mkdir()
    checkpoints_dir = outlier_dir

    '''LOG'''
    logger = logging.getLogger("PointNet")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(checkpoints_dir) + '/' + 'train_%s_'%args.model_name+ str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))+'.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('---------------------------------------------------TRAINING---------------------------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    '''DATA LOADING'''
    logger.info('Load dataset ...')
    train_data, train_label, test_data, test_label = load_data(datapath, classification=True)
    logger.info("The number of training data is: %d",train_data.shape[0])
    logger.info("The number of test data is: %d", test_data.shape[0])

    ## Replace a fraction of the points with outliers drawn uniformly from the unit sphere
    if args.outlier_fraction > 0.0:
        # Training set
        num_outliers = int(args.outlier_fraction * train_data.shape[1])
        print('Number of training set outliers per point cloud: {}'.format(num_outliers))
        for i in range(train_data.shape[0]): # For each point cloud in the batch
            random_indices = np.random.choice(train_data.shape[1], num_outliers, replace=False)
            for j in range(num_outliers): # For each point in outlier subset
                random_point = 2.0 * np.random.rand(3) - 1.0
                # Ensure outliers are within unit sphere:
                while np.linalg.norm(random_point) > 1.0:
                    random_point = 2.0 * np.random.rand(3) - 1.0
                train_data[i, random_indices[j], :] = random_point # Make an outlier, uniform distribution in [-1,1]^3
        # Testing set
        num_outliers = int(args.outlier_fraction * test_data.shape[1])
        print('Number of test set outliers per point cloud: {}'.format(num_outliers))
        for i in range(test_data.shape[0]): # For each point cloud in the batch
            random_indices = np.random.choice(test_data.shape[1], num_outliers, replace=False)
            for j in range(num_outliers): # For each point in outlier subset
                random_point = 2.0 * np.random.rand(3) - 1.0
                # Ensure outliers are within unit sphere:
                while np.linalg.norm(random_point) > 1.0:
                    random_point = 2.0 * np.random.rand(3) - 1.0
                test_data[i, random_indices[j], :] = random_point # Make an outlier, uniform distribution in [-1,1]^3

    trainDataset = ModelNetDataLoader(train_data, train_label, rotation=ROTATION)
    if ROTATION is not None:
        print('The range of training rotation is',ROTATION)
    testDataset = ModelNetDataLoader(test_data, test_label, rotation=ROTATION)
    trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=args.batchsize, shuffle=True)
    testDataLoader = torch.utils.data.DataLoader(testDataset, batch_size=args.batchsize, shuffle=False)

    '''MODEL LOADING'''
    num_class = 40
    classifier = PointNetCls(num_class, args.input_transform, args.feature_transform, args.robust_type, args.alpha).cuda()
    if args.pretrain is not None:
        print('Use pretrain model...')
        logger.info('Use pretrain model')
        checkpoint = torch.load(args.pretrain)
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
    else:
        print('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.evaluate:
        acc, map, _ = test(classifier, testDataLoader, do_map=True)
        logger.info('Test Accuracy: %f', acc)
        logger.info('mAP: %f', map)
        logger.info('%f,%f'%(acc, map))
        print('Test Accuracy:\n%f'%acc)
        print('mAP:\n%f'%map)
        # print('%f,%f'%(acc, map))
        return

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    global_epoch = 0
    global_step = 0
    best_tst_accuracy = 0.0
    blue = lambda x: '\033[94m' + x + '\033[0m'

    '''TRAINING'''
    logger.info('Start training...')
    for epoch in range(start_epoch,args.epoch):
        print('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        logger.info('Epoch %d (%d/%s):' ,global_epoch + 1, epoch + 1, args.epoch)

        scheduler.step()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()
            classifier = classifier.train()
            pred, trans_feat = classifier(points)
            loss = F.nll_loss(pred, target.long())
            if args.feature_transform and args.model_name == 'pointnet':
                loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()
            global_step += 1

        train_acc = test(classifier.eval(), trainDataLoader) if args.train_metric else None
        acc, map, _ = test(classifier, testDataLoader, do_map=True)

        print('\r Loss: %f' % loss.data)
        logger.info('Loss: %f', loss.data)
        if args.train_metric:
            print('Train Accuracy: %f' % train_acc)
            logger.info('Train Accuracy: %f', (train_acc))
        logger.info('Test Accuracy: %f', acc)
        logger.info('Test mAP: %f', map)
        print('\r Test %s: %f' % (blue('Accuracy'),acc))
        print('\r Test %s: %f' % (blue('mAP'), map))
        if args.train_metric:
            logger.info('%f,%f,%f' % (train_acc, acc, map))
            print('\r%f,%f,%f' % (train_acc, acc, map))
        else:
            logger.info('%f,%f' % (acc, map))
            print('\r%f,%f' % (acc, map))

        if (acc >= best_tst_accuracy):
            best_tst_accuracy = acc
        # Save every 10
        if (epoch + 1) % 10 == 0:
            logger.info('Save model...')
            save_checkpoint(
                global_epoch + 1,
                train_acc if args.train_metric else 0.0,
                acc,
                map,
                classifier,
                optimizer,
                str(checkpoints_dir),
                args.model_name)
            print('Saving model....')
        global_epoch += 1
    print('Best Accuracy: %f'%best_tst_accuracy)

    logger.info('Save final model...')
    save_checkpoint(
        global_epoch,
        train_acc if args.train_metric else 0.0,
        acc,
        map,
        classifier,
        optimizer,
        str(checkpoints_dir),
        args.model_name)
    print('Saving final model....')

    logger.info('End of training...')

if __name__ == '__main__':
    main()
