import os
import logging
from tqdm import tqdm
import numpy as np
import argparse
import time
import sys
import cv2

import mxnet as mx
from mxnet import gluon, ndarray as nd
from mxnet.gluon.data.vision import transforms
from mxnet.contrib.quantization import *

import gluoncv
gluoncv.utils.check_version('0.6.0')
from gluoncv.model_zoo.segbase import *
from gluoncv.model_zoo import get_model
from gluoncv.data import get_segmentation_dataset, ms_batchify_fn
from gluoncv.utils.viz import get_color_pallete
from gluoncv.utils.parallel import *

#change to local dataset, start custom train
from gluoncv import data as gdata
gdata.VOCSegmentation.CLASSES = ('background', 'left_lane', 'right_lane')
gdata.VOCSegmentation.BASE_DIR = "seg_pig_lane"
gdata.VOCSegmentation.NUM_CLASS = 3

def parse_args():
    parser = argparse.ArgumentParser(description='Validation on Semantic Segmentation model')
    parser.add_argument('--model-zoo', type=str, default=None,
                        help='evaluating on model zoo model')
    parser.add_argument('--model', type=str, default='fcn',
                        help='model name (default: fcn)')
    parser.add_argument('--backbone', type=str, default='resnet101',
                        help='base network')
    parser.add_argument('--deploy', action='store_true',
                        help='whether load static model for deployment')
    parser.add_argument('--model-prefix', type=str, required=False,
                        help='load static model as hybridblock.')
    parser.add_argument('--image-shape', type=int, default=480,
                        help='image shape')
    parser.add_argument('--base-size', type=int, default=520,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=480,
                        help='crop image size')
    parser.add_argument('--height', type=int, default=None,
                        help='height of original image size')
    parser.add_argument('--width', type=int, default=None,
                        help='width of original image size')
    parser.add_argument('--mode', type=str, default='val',
                        help='val, testval')
    parser.add_argument('--dataset', type=str, default='pascal_voc',
                        help='dataset used for validation [pascal_voc, pascal_aug, coco, ade20k]')
    parser.add_argument('--quantized', action='store_true',
                        help='whether to use int8 pretrained  model')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-iterations', type=int, default=100,
                        help='number of benchmarking iterations.')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of workers for data loading')
    parser.add_argument('--pretrained', action="store_true",
                        help='whether to use pretrained params')
    parser.add_argument('--ngpus', type=int,
                        default=len(mx.test_utils.list_gpus()),
                        help='number of GPUs (default: 4)')
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--syncbn', action='store_true', default=False,
                        help='using Synchronized Cross-GPU BatchNorm')
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluation only')
    # dummy benchmark
    parser.add_argument('--benchmark', action='store_true', default=False,
                        help='whether to use dummy data for benchmark')
    # calibration
    parser.add_argument('--calibration', action='store_true',
                        help='quantize model')
    parser.add_argument('--num-calib-batches', type=int, default=5,
                        help='number of batches for calibration')
    parser.add_argument('--quantized-dtype', type=str, default='auto',
                        choices=['auto', 'int8', 'uint8'],
                        help='quantization destination data type for input data')
    parser.add_argument('--calib-mode', type=str, default='entropy',
                        help='calibration mode used for generating calibration table for the quantized symbol; supports'
                             ' 1. none: no calibration will be used. The thresholds for quantization will be calculated'
                             ' on the fly. This will result in inference speed slowdown and loss of accuracy'
                             ' in general.'
                             ' 2. naive: simply take min and max values of layer outputs as thresholds for'
                             ' quantization. In general, the inference accuracy worsens with more examples used in'
                             ' calibration. It is recommended to use `entropy` mode as it produces more accurate'
                             ' inference results.'
                             ' 3. entropy: calculate KL divergence of the fp32 output and quantized output for optimal'
                             ' thresholds. This mode is expected to produce the best inference accuracy of all three'
                             ' kinds of quantized models if the calibration dataset is representative enough of the'
                             ' inference dataset.')

    args = parser.parse_args()

    args.ctx = [mx.cpu(0)]
    args.ctx = [mx.gpu(i) for i in range(args.ngpus)] if args.ngpus > 0 else args.ctx

    args.norm_layer = mx.gluon.contrib.nn.SyncBatchNorm if args.syncbn \
        else mx.gluon.nn.BatchNorm
    args.norm_kwargs = {'num_devices': args.ngpus} if args.syncbn else {}
    return args

if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig()
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    logging.info(args)


    #load model
    model = get_segmentation_model(model=args.model, dataset=args.dataset, ctx=args.ctx,
                                   backbone=args.backbone, norm_layer=args.norm_layer,
                                   norm_kwargs=args.norm_kwargs, aux=args.aux,
                                   base_size=args.base_size, crop_size=args.crop_size,
                                   height=args.height, width=args.width)
    # load local pretrained weight
    assert args.resume is not None, '=> Please provide the checkpoint using --resume'
    if os.path.isfile(args.resume):
        model.load_parameters(args.resume, ctx=args.ctx, allow_missing=True)
    else:
        raise RuntimeError("=> no checkpoint found at '{}'" \
                           .format(args.resume))

    # image transform
    # image transform
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])

    # DO NOT modify!!! Only support batch_size=ngus
    batch_size = args.ngpus

    # output folder
    outdir = 'outdir'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # get dataset
    if args.eval:
        testset = get_segmentation_dataset(
            args.dataset, root = "/home/liyongjing/Egolee/data/dataset/voc/VOCdevkit", split='val', mode='testval', transform=input_transform)
    else:
        testset = get_segmentation_dataset(
            args.dataset,root = "/home/liyongjing/Egolee/data/dataset/voc/VOCdevkit", split='test', mode='test', transform=input_transform)

    if 'icnet' in args.model:
        test_data = gluon.data.DataLoader(
            testset, batch_size, shuffle=False, last_batch='rollover',
            num_workers=args.workers)
    else:
        test_data = gluon.data.DataLoader(
            testset, batch_size, shuffle=False, last_batch='rollover',
            batchify_fn=ms_batchify_fn, num_workers=args.workers)

    if 'icnet' in args.model:
        evaluator = DataParallelModel(SegEvalModel(model, use_predict=True), ctx_list=args.ctx)
    else:
        evaluator = MultiEvalModel(model, testset.num_class, ctx_list=args.ctx)

    tbar = tqdm(test_data)
    for i, (data, dsts) in enumerate(tbar):
        print("data_shape:", data.shape)
        print("dsts_shape:", dsts.shape)
        if args.eval:
            #predicts = [pred[0] for pred in evaluator.parallel_forward(data)]
            predicts = [pred[0] for pred in evaluator(data)]

            predicts_tmp = predicts[0].asnumpy()
            print("predicts_tmp_shape", predicts_tmp.shape)

            predicts_tmp =np.uint8( np.argmax(predicts_tmp[0], 0))*100
            print("predicts_tmp_shape:", predicts_tmp.shape)
            img = np.uint8(data[0].asnumpy())
            img = np.transpose(img, (1,2,0))
            print(img.shape)
            cv2.imshow("img_predict", predicts_tmp)
            cv2.imshow("img", img)
            cv2.waitKey(0)
            cv2.imwrite("./outdir/{}.jpg".format(i), predicts_tmp)
            cv2.imwrite("./outdir/img_{}.jpg".format(i), img)   
