import mxnet as mx
from mxnet import nd

import numpy as np
import cv2
import logging
import os
from gluoncv.model_zoo.segbase import *
from gluoncv import data as gdata
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def make_palette(num_classes):
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for k in range(0, num_classes):
        label = k
        i = 0
        while label:
            palette[k, 0] |= (((label >> 0) & 1) << (7 - i))
            palette[k, 1] |= (((label >> 1) & 1) << (7 - i))
            palette[k, 2] |= (((label >> 2) & 1) << (7 - i))
            label >>= 3
            i += 1
    return palette


def vis_seg(img, seg, palette, alpha=0.4):
    vis = np.array(img, dtype=np.float32)
    mask = seg > 0
    vis[mask] *= 1. - alpha
    vis[mask] += alpha * palette[seg[mask].flat]
    vis = vis.astype(np.uint8)
    return vis


def export_gluoncv_segmentation_model(resume_path, num_class):
    epoch = 0
    height = 640
    width = 640
    model_name = os.path.split(resume_path)[0] + "/icnet_resnet50_v1b"

    print("Export model start.")
    gdata.VOCSegmentation.NUM_CLASS = num_class
    pretrained_net = get_segmentation_model(model="icnet", dataset='pascal_voc',
                                                backbone='resnet50', norm_layer=mx.gluon.nn.BatchNorm,
                                                norm_kwargs={}, aux=False,
                                                base_size=520, crop_size=480, height=height, width=width)

    pretrained_net.load_parameters(resume_path, allow_missing=True)
    pretrained_net.hybridize()
    item = mx.random.uniform(shape=(1, 3, height, width))
    pretrained_net(item)
    pretrained_net.export(model_name, epoch=epoch)
    print("Export model done.")


class SegmentationMxnet():
    def __init__(self, model_path, num_class):
        logger.info("Load segmentation model start.")
        self.mean_ = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std_ = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        self.height_ = 640
        self.width_ = 640
        self.ctx_ = [mx.gpu(int(0))]
        self.epoch_ = 0
        self.palette_ = make_palette(num_class)

        sym, arg_params, aux_params = mx.model.load_checkpoint(model_path, self.epoch_)
        self.mod_ = mx.mod.Module(symbol=sym, context=self.ctx_, data_names=["data"], )
        self.mod_.bind(for_training=False, data_shapes=[("data", (1, 3, self.height_, self.width_))])
        self.mod_.set_params(arg_params, aux_params)
        logger.info("Load segmentation model done.")

    def img_preprocess(self, img):
        img = img[:, :, ::-1]
        img_p = cv2.resize(img, ( self.width_, self.height_))
        img_p = (img_p.astype(np.float32) / 255.)
        img_p = (img_p - self.mean_) / self.std_
        img_p = mx.nd.array(img_p).transpose((2, 0, 1))
        return img_p

    def net_forward(self, img):
        img_p = self.img_preprocess(img)
        batch_data = nd.zeros((1, 3, self.height_, self.width_))
        batch_data[0][:] = img_p
        batch_feed = mx.io.DataBatch(data=[batch_data])

        logger.info("Start segmentation forward.")
        self.mod_.forward(batch_feed, is_train=False)
        mx.nd.waitall()
        predicts = self.mod_.get_outputs()
        # print(len(predicts))
        # exit(1)
        predict = predicts[0]
        logger.info("Finish segmentation forward.")
        return predict

    def vis_img(self, img):
        predict_mask = self.net_forward(img)
        print("predict_mask", predict_mask.shape)
        predict_mask = predict_mask[0].asnumpy()
        predict_mask = np.uint8(np.argmax(predict_mask, 0))
        predict_mask = cv2.resize(predict_mask, (img.shape[1], img.shape[0]), cv2.INTER_NEAREST)
        vis_img = vis_seg(img, predict_mask, self.palette_)
        return vis_img, predict_mask


if __name__=="__main__":
    num_class = 5
    resume_path = "/home/liyongjing/Egolee/programs/gluon-cv/scripts/segmentation/epoch_0028_mIoU_0.9375.params"
    export_gluoncv_segmentation_model(resume_path, num_class)
    # exit(1)

    model_path = os.path.split(resume_path)[0] + "/icnet_resnet50_v1b"
    print("model initial start.")
    seg_model = SegmentationMxnet(model_path, num_class)
    print("model initial done.")

    imgs_dir = "/home/liyongjing/Egolee/data/dataset/voc/VOCdevkit/seg_oil_meter/JPEGImages"
    img_names = list(filter(lambda x: x[-3:] == "jpg" or x[-3:] == "png", os.listdir(imgs_dir)))

    for i, img_name in enumerate(img_names):
        print("Processing:{}".format(img_name))
        img_path = os.path.join(imgs_dir, img_name)
        if not os.path.isfile(img_path):
            print("File {} is not exist!".format(img_path))
            continue
        img = cv2.imread(img_path)
        vis_img, predict_mask = seg_model.vis_img(img)
        cv2.imshow("img_predict", vis_img)
        wait_key = cv2.waitKey(0)
        if wait_key == 27:
            break
    exit(1)
