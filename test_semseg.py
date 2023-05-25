from S3DISDataLoader import S3DISDataLoader, recognize_all_data,class2label
import logging
import numpy as np
import torch
from utils import test_seg, test
from model import GACNet
import argparse

seg_classes = class2label
seg_label_to_cat = {}
for i,cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def parse_args():
    parser = argparse.ArgumentParser('GACNet')
    parser.add_argument('--batchSize', type=int, default=24, help='input batch size [default: 24]')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers [default: 4]')
    parser.add_argument('--epoch', type=int, default=200, help='number of epochs for training [default: 200]')
    parser.add_argument('--log_dir', type=str, default='logs/',help='decay rate of learning rate')
    parser.add_argument('--pretrain', type=str, default=None,help='whether use pretrain model')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training [default: 0.001 for Adam, 0.01 for SGD]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay for Adam')
    parser.add_argument('--optimizer', type=str, default='SGD', help='type of optimizer')
    parser.add_argument('--multi_gpu', type=str, default=None, help='whether use multi gpu training')
    parser.add_argument('--dropout', type=float, default=0, help='dropout [default: 0]')
    parser.add_argument('--alpha', type=float, default=0.2, help='alpha for leakyRelu [default: 0.2]')
    parser.add_argument('--test_class_num', type=float, default=4, help='class value for creating test dataloader [default: 4]')
    return parser.parse_args()

def main(args):
    '''LOG'''
    print("Segmentation of Test data")
    args = parse_args()
    logger = logging.getLogger('GACNet')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str('logs') + '/train_GACNet.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('PARAMETER ...')
    print('Load data...')
    train_data, train_label, test_data, test_label = recognize_all_data(test_area = args.test_class_num)
    dataset = S3DISDataLoader(train_data,train_label)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
                                             shuffle=True, num_workers=int(args.workers))
    test_dataset = S3DISDataLoader(test_data,test_label)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8,
                                                 shuffle=True, num_workers=int(args.workers))

    num_classes = 13
    blue = lambda x: '\033[94m' + x + '\033[0m'
    model = GACNet(num_classes,args.dropout,args.alpha)
    model.load_state_dict(torch.load('./checkpoints/GACNet_000_0.3636.pth', map_location=torch.device('cpu')))
    model.eval()
    test_metrics, _, cat_mean_iou = test_seg(model, testdataloader, seg_label_to_cat)
    mean_iou = np.mean(cat_mean_iou)

    print('%s accuracy: %f  meanIOU: %f' % (blue('test'), test_metrics['accuracy'],mean_iou))
    logger.info('%s accuracy: %f  meanIOU: %f' % ('test', test_metrics['accuracy'],mean_iou))


if __name__ == '__main__':
    args = parse_args()
    main(args)