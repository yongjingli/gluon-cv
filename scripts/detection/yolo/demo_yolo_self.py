"""YOLO Demo script."""
import os
import argparse
import mxnet as mx
import gluoncv as gcv
gcv.utils.check_version('0.6.0')
from gluoncv.data.transforms import presets
from matplotlib import pyplot as plt

from mxnet import nd
from gluoncv.utils import export_block

import numpy as np
import cv2
import logging
import time
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

colors = [(255,0,0), (0,255,0), (0,0,255),
          (255,255,0), (255,0,255), (0,255,255),
          (128,0,0), (0,128,0), (0,0,128),
          (128,128,0), (128,0,128), (0,128,128)]


def parse_args():
    parser = argparse.ArgumentParser(description='Test with YOLO networks.')
    parser.add_argument('--network', type=str, default='yolo3_darknet53_coco',
                        help="Base network name")
    parser.add_argument('--images', type=str, default='',
                        help='Test images, use comma to split multiple.')
    parser.add_argument('--gpus', type=str, default='',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--pretrained', type=str, default='True',
                        help='Load weights from previously saved parameters.')
    parser.add_argument('--thresh', type=float, default=0.5,
                        help='Threshold of object score when visualize the bboxes.')
    args = parser.parse_args()
    return args


def save_imgs_from_videos(videos_path,save_imgs_path,video_name_=-1, read_format=["mp4","h265","ts","mkv", "avi"]):
    videos_name=filter(lambda x: x.split(".")[-1] in read_format,os.listdir(videos_path))
    print(videos_name)
    jump_frame = 1
    for video_name in videos_name:
        if video_name_ != -1:
            if video_name != video_name_:
                continue
        video_path=os.path.join(videos_path,video_name)
        video_cap=cv2.VideoCapture(video_path)
        rval=video_cap.isOpened()
        count=0

        logger.info("Process:{}".format(video_name))
        while rval:
            rval,frame=video_cap.read()
            count = count+1
            img_name = video_name.split(".")[0] + "_" + str(count)+".jpg"
            if rval:
                if count % jump_frame == 0:
                    cv2.imwrite(os.path.join(save_imgs_path,img_name),frame)
            else:
                break
        video_cap.release()

def save_video_from_imgs(imgs_path,video_name = "save_video.avi", video_size = (4096, 3000)):
    imgs_format = ["jpg", "bmp"]
    imgs_name = list(filter(lambda x: x.split(".")[-1] in imgs_format,os.listdir(imgs_path)))
    #imgs_name.sort(key = lambda x: int(x.split(".")[0]))
    imgs_name.sort(key = lambda x: int(x.split(".")[0].split("_")[-1]))

    videoWrite= cv2.VideoWriter(video_name,cv2.VideoWriter_fourcc(*'XVID'), 30, video_size)
    for img_name in imgs_name:
        print(img_name)
        img_path = os.path.join(imgs_path, img_name)
        img =  cv2.imread(img_path)
        img = cv2.resize(img, video_size)
        videoWrite.write(img)
    videoWrite.release()


class YoloDetctor():
    def __init__(self, model_name, model_params, ctx = [mx.cpu()]):
        #ctx = [mx.gpu(int(0))]
        self._ctx = ctx
        self._net = gcv.model_zoo.get_model(model_name, pretrained=False, pretrained_base=False)
        #self._net.load_parameters(model_params, ctx=ctx, ignore_extra=True, allow_missing=True)
        
        #self train voc
        self._net.load_parameters(model_params)
        self._net.collect_params().reset_ctx(ctx = ctx)
        
        self._net.set_nms(0.2, 200)
        
        #name of net classes
        self._class_names=self._net.classes
        self._thresh = 0.5
    
    def vis_det_boxes(self, img_path):
        logger.info("Processing:{}".format(img_path))         
        x, img = presets.yolo.load_test(img_path, short=416)  
        time_start = time.time()             
        x = x.as_in_context(self._ctx[0])
        
        self._net.hybridize()
        labels, scores, bboxes = [xx[0].asnumpy() for xx in self._net(x)]
                
        #covnvert to numpy
        if isinstance(bboxes, mx.nd.NDArray):
            bboxes = bboxes.asnumpy()
        if isinstance(labels, mx.nd.NDArray):
            labels = labels.asnumpy()
        if isinstance(scores, mx.nd.NDArray):
            scores = scores.asnumpy()
        if isinstance(scores, mx.nd.NDArray):
            img = img.asnumpy()

        infer_time = time.time()-time_start
        logger.info("Infer_time:{}ms".format(infer_time*1000)) 
        
        if 0:
            #self._net.hybridize()
            self._net.export("./save_models/yolov3_darknet53", epoch=0)
            #export_block(model_name, self._net)
            exit(1) 
                
        #parse detection result
        for i, bbox in enumerate(bboxes):
            if scores is not None and scores.flat[i] < self._thresh:
                continue
            if labels is not None and labels.flat[i] < 0:
                continue
            
            #box, id, score    
            cls_id = int(labels.flat[i]) if labels is not None else -1
            xmin, ymin, xmax, ymax = [int(x) for x in bbox]
            score = '{:.3f}'.format(scores.flat[i]) if scores is not None else ''
            color_id = cls_id%len(colors)

            xmin = min(max(0, xmin),img.shape[1])
            xmax = min(max(0, xmax), img.shape[1])
            ymin = min(max(0, ymin), img.shape[0])
            ymax = min(max(0, ymax), img.shape[0])


            # save person box img
            if 1:
                if cls_id == 0:
                    save_img_dir = "/home/liyongjing/Egolee/programs/gluon-cv/scripts/detection/yolo/box_imgs"
                    _, img_name = os.path.split(img_path)
                    img_name = img_name[:-3] + "_" + str(i) + ".jpg"
                    save_path = os.path.join(save_img_dir, img_name)
                    box_img = img[ymin:ymax, xmin:xmax, ::-1]
                    if box_img.shape:
                        cv2.imwrite(save_path, box_img)
            
            #draw detection boxes           
            # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), colors[color_id], 2)
            # cv2.putText(img, self._class_names[cls_id]+":"+str(score), (xmin,max(0,ymin-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[color_id], 1)
            #
        img = img[...,::-1]
        return img
 



if __name__ == '__main__':
    from gluoncv import data as gdata
    CLASSES = ('person', 'car', 'bus', 'motorbike', 'bicycle')
    gdata.VOCDetection.CLASSES = CLASSES

    args = parse_args()      
    #save img from video
    if 0:
      videos_path="/home/liyongjing/Egolee/data/local/person_car"
      save_imgs_path=videos_path+"/imgs"
      if not os.path.exists(save_imgs_path):
          os.mkdir(save_imgs_path)
      save_imgs_from_videos(videos_path,save_imgs_path)    
    
    #start detection
    if 1:
      #initial model
      args = parse_args()
      model_name = "yolo3_darknet53_voc"
      model_params = "./person_car_minup/yolo3_darknet53_voc_best.params"
      yolo_detector = YoloDetctor(model_name, model_params, ctx = [mx.gpu(int(0))])
      logger.info("Yolo_detector initialize done.")
      #
      # #img paths
      imgs_path = "/home/liyongjing/Egolee/data/test_person_car/imgs"
      # imgs_path_det = "/home/liyongjing/Egolee/data/local/person_car/yolo_det_imgs"
      # if not os.path.exists(imgs_path_det):
      #     os.mkdir(imgs_path_det)
      #
      imgs_name = list(filter(lambda x: x.split(".")[-1] in ["jpg", "bmp"], os.listdir(imgs_path)))
      # imgs_name.sort(key = lambda x: int(x.split(".")[0].split("_")[-1]))
      
      #begin process
      for img_name in imgs_name:
          img_path = os.path.join(imgs_path, img_name)
          # img_path_save = os.path.join(imgs_path_det, img_name)
          img_ori = cv2.imread(img_path)
          print(img_ori.shape)
          x, img = presets.yolo.load_test(img_path, short=416)
          print(img.shape)

          img_det_result = yolo_detector.vis_det_boxes(img_path)
          #cv2.imwrite(img_path_save, img_det_result)
          
    #save det_imgs to videos
    if 0:
        imgs_path = "/home/liyongjing/Egolee/data/local/person_car/yolo_det_imgs"
        save_video_from_imgs(imgs_path,video_name = "yolo_video.avi", video_size = (1333, 750))              
