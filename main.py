
from __future__ import print_function

import argparse
import os
import time, platform

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import DATASET_NAMES, TestDataset, dataset_info
from modelB4 import LDC
from utils.img_processing import (save_image_batch_to_disk, count_parameters)
import todos
import pdb

IS_LINUX = True if platform.system()=="Linux" else False


def test(checkpoint_path, dataloader, model, device, output_dir, args):
    # checkpoint_path = 'checkpoints/BRIND/11/11_model.pth'
    # output_dir = 'result/BRIND2CLASSIC'

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint filte note found: {checkpoint_path}")
    print(f"Restoring weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model.eval()

    with torch.no_grad():
        total_duration = []
        for batch_id, sample_batched in enumerate(dataloader):
            images = sample_batched['images'].to(device)
            if not args.test_data == "CLASSIC":
                labels = sample_batched['labels'].to(device)
            file_names = sample_batched['file_names']
            image_shape = sample_batched['image_shape']

            print(f"{file_names}: {images.shape}")

            end = time.perf_counter()
            if device.type == 'cuda':
                torch.cuda.synchronize()

            # tensor [images] size: [1, 3, 512, 512], 
            #   min: -123.68000030517578 , 
            #   max: 151.06100463867188

            preds = model(images)
            todos.debug.output_var("preds", preds)
            # list [preds] len: 5
            # tensor [preds[0]] size: [1, 1, 512, 512] , min: -7.049300670623779 , max: 7.8712992668151855
            # tensor [preds[1]] size: [1, 1, 512, 512] , min: -13.446599006652832 , max: 0.9053399562835693
            # tensor [preds[2]] size: [1, 1, 512, 512] , min: -18.81188201904297 , max: 1.7953037023544312
            # tensor [preds[3]] size: [1, 1, 512, 512] , min: -17.774682998657227 , max: 1.6504275798797607
            # tensor [preds[4]] size: [1, 1, 512, 512] , min: -17.566560745239258 , max: 1.7280923128128052

            if device.type == 'cuda':
                torch.cuda.synchronize()
            tmp_duration = time.perf_counter() - end
            total_duration.append(tmp_duration)

            save_image_batch_to_disk(preds, output_dir, file_names, image_shape)
            torch.cuda.empty_cache()
    total_duration = np.sum(np.array(total_duration))
    print("******** Testing finished in", args.test_data, "dataset. *****")
    print("FPS: %f.4" % (len(dataloader)/total_duration))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='LDC trainer.')
    parser.add_argument('--choose_test_data',
                        type=int,
                        default=-1,
                        help='Choose a dataset for testing: 0 - 8')
    # ----------- test -------0--


    TEST_DATA = DATASET_NAMES[parser.parse_args().choose_test_data] # max 8
    # TEST_DATA -- 'CLASSIC'


    test_inf = dataset_info(TEST_DATA, is_linux=IS_LINUX)
    test_dir = test_inf['data_dir'] # -- data

    # Training settings
    # BIPED-B2=1, BIPDE-B3=2, just for evaluation, using LDC trained with 2 or 3 bloacks
    TRAIN_DATA = DATASET_NAMES[6] # BIPED=0, BRIND=6, MDBD=10 # xxxx8888

    train_inf = dataset_info(TRAIN_DATA, is_linux=IS_LINUX)
    train_dir = train_inf['data_dir'] # -- '/opt/dataset/BRIND'


    # Data parameters
    parser.add_argument('--input_dir',
                        type=str,
                        default=train_dir,
                        help='the path to the directory with the input data.')
    parser.add_argument('--input_val_dir',
                        type=str,
                        default=test_inf['data_dir'],
                        help='the path to the directory with the input data for validation.')
    parser.add_argument('--output_dir',
                        type=str,
                        default='checkpoints',
                        help='the path to output the results.')
    parser.add_argument('--train_data',
                        type=str,
                        choices=DATASET_NAMES,
                        default=TRAIN_DATA,
                        help='Name of the dataset.')# TRAIN_DATA,BIPED-B3
    parser.add_argument('--test_data',
                        type=str,
                        choices=DATASET_NAMES,
                        default=TEST_DATA,
                        help='Name of the dataset.')
    parser.add_argument('--test_list',
                        type=str,
                        default=test_inf['test_list'],
                        help='Dataset sample indices list.')
    parser.add_argument('--checkpoint_data',
                        type=str,
                        default='11/11_model.pth',# 37 for biped 60 MDBD
                        help='Checkpoint path.')
    parser.add_argument('--test_img_width',
                        type=int,
                        default=test_inf['img_width'],
                        help='Image width for testing.')
    parser.add_argument('--test_img_height',
                        type=int,
                        default=test_inf['img_height'],
                        help='Image height for testing.')
    parser.add_argument('--res_dir',
                        type=str,
                        default='result',
                        help='Result directory')
    parser.add_argument('--log_interval_vis',
                        type=int,
                        default=100,
                        help='The NO B to wait before printing test predictions. 200')

    # parser.add_argument('--epochs',
    #                     type=int,
    #                     default=25,
    #                     metavar='N',
    #                     help='Number of training epochs (default: 25).')
    parser.add_argument('--lr', default=5e-5, type=float,
                        help='Initial learning rate. =5e-5')
    parser.add_argument('--lrs', default=[25e-4,5e-4,1e-5], type=float,
                        help='LR for set epochs')
    parser.add_argument('--wd', type=float, default=0., metavar='WD',
                        help='weight decay (Good 5e-6)')
    parser.add_argument('--adjust_lr', default=[6,12,18], type=int,
                        help='Learning rate step size.')  # [6,9,19]
    parser.add_argument('--version_notes',
                        default='LDC-BIPED: B4 Exp 67L3 xavier init normal+ init normal CatsLoss2 Cofusion',
                        type=str,
                        help='version notes')
    parser.add_argument('--batch_size',
                        type=int,
                        default=8,
                        metavar='B',
                        help='the mini-batch size (default: 8)')
    parser.add_argument('--workers',
                        default=8,
                        type=int,
                        help='The number of workers for the dataloaders.')
    parser.add_argument('--tensorboard',type=bool,
                        default=True,
                        help='Use Tensorboard for logging.'),
    parser.add_argument('--img_width',
                        type=int,
                        default=352,
                        help='Image width for training.') # BIPED 352 BSDS 352/320 MDBD 480
    parser.add_argument('--img_height',
                        type=int,
                        default=352,
                        help='Image height for training.') # BIPED 480 BSDS 352/320
    parser.add_argument('--mean_pixel_values',
                        default=[103.939,116.779,123.68,137.86],
                        type=float)  # [103.939,116.779,123.68,137.86] [104.00699, 116.66877, 122.67892]
    # BRIND mean = [104.007, 116.669, 122.679, 137.86]
    # BIPED mean_bgr processed [160.913,160.275,162.239,137.86]
    args = parser.parse_args()
    return args


def main(args):
    """Main function."""

    print(f"Number of GPU's available: {torch.cuda.device_count()}")
    print(f"Pytorch version: {torch.__version__}")

    checkpoint_path = os.path.join(args.output_dir, args.train_data,args.checkpoint_data)


    device = torch.device('cpu' if torch.cuda.device_count() == 0
                          else 'cuda')

    model = LDC().to(device)

    dataset_val = TestDataset(args.input_val_dir,
                              test_data=args.test_data,
                              img_width=args.test_img_width,
                              img_height=args.test_img_height,
                              mean_bgr=args.mean_pixel_values[0:3] if len(
                                  args.mean_pixel_values) == 4 else args.mean_pixel_values,
                              test_list=args.test_list, arg=args
                              )
    dataloader_val = DataLoader(dataset_val,
                                batch_size=1,
                                shuffle=False,
                                num_workers=args.workers)
    # Testing
    output_dir = os.path.join(args.res_dir, args.train_data+"2"+ args.test_data)
    print(f"output_dir: {output_dir}")
    test(checkpoint_path, dataloader_val, model, device, output_dir, args)

    # Count parameters:
    num_param = count_parameters(model)
    print('-------------------------------------------------------')
    print('LDC parameters:')
    print(num_param)
    print('-------------------------------------------------------')


if __name__ == '__main__':
    args = parse_args()
    main(args)