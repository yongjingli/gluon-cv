import os
import logging

import numpy as np
import cv2

import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
import gluoncv as gcv

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

colors = [(255,0,0), (0,255,0), (0,0,255),
          (255,255,0), (255,0,255), (0,255,255),
          (128,0,0), (0,128,0), (0,0,128),
          (128,128,0), (128,0,128), (0,128,128)]

classes = ['person', 'car',  'bus', 'motorbike', 'bicycle']

class YoloDetector():    
    def __init__(self, width, height, model_path, epoch, thresh = 0.4, ctx = [mx.cpu()]):
        self._width = width
        self._height = height
        #self._down_sample_ratio = 4  # 1 / 4
        #self._class_num = 6
        
        self._thresh = thresh

        self._mean = np.array([0.485, 0.456, 0.406],
                              dtype=np.float32).reshape(1, 1, 3)
        self._std = np.array([0.229, 0.224, 0.225],
                             dtype=np.float32).reshape(1, 1, 3)

        self._ctx = [mx.gpu(int(0))]
        # Init the model
        #load model 1
        sym, arg_params, aux_params = mx.model.load_checkpoint(model_path, epoch)
        mod = mx.mod.Module(symbol=sym, context=self._ctx, data_names=["data"],)
        mod.bind(for_training=False, data_shapes=[ ("data", (1,3,self._height, self._width)) ] )        
        mod.set_params(arg_params, aux_params)
        self.mod = mod
        
        #name of net classes
        self._class_names= classes
                
        #load model 2
        #net = gluon.nn.SymbolBlock.imports(json_name, ['data'], params_name, ctx = ctx)

        
    def _detect(self, img_src, index = 0):
        #pre_process
        scale_h = self._height / img_src.shape[0]
        scale_w = self._width / img_src.shape[1]
        img = cv2.resize(img_src, (self._height, self._width))
        img_show = img.copy()
        img = (img.astype(np.float32) / 255.)
        img = (img - self._mean) / self._std
        img = mx.nd.array(img).transpose((2,0,1))

        batch_data = nd.zeros((1, 3, self._height, self._width))
        batch_data[0][:] = img
        train_iter = mx.io.DataBatch(data=[batch_data])
        
        #network infer
        self.mod.forward(train_iter, is_train=False)
        mx.nd.waitall()
        labels, scores, bboxes = self.mod.get_outputs()
        labels, scores, bboxes = labels[0], scores[0], bboxes[0]

        #covnvert to numpy
        if isinstance(bboxes, mx.nd.NDArray):
            bboxes = bboxes.asnumpy()
        if isinstance(labels, mx.nd.NDArray):
            labels = labels.asnumpy()
        if isinstance(scores, mx.nd.NDArray):
            scores = scores.asnumpy()
        if isinstance(scores, mx.nd.NDArray):
            img = img.asnumpy()     
        
        
        #visualize result
        for i, bbox in enumerate(bboxes):
            if scores is not None and scores.flat[i] < self._thresh:
                continue
            if labels is not None and labels.flat[i] < 0:
                continue
            
            #box, id, score    
            cls_id = int(labels.flat[i]) if labels is not None else -1
           # print(i,"_label:",cls_id)
            xmin, ymin, xmax, ymax = [int(x) for x in bbox]
            
            
            score = '{:.3f}'.format(scores.flat[i]) if scores is not None else ''
            color_id = cls_id%len(colors)
            
            #draw detection boxes           
            cv2.rectangle(img_show, (xmin, ymin), (xmax, ymax), colors[color_id], 2)
            cv2.putText(img_show, self._class_names[cls_id]+":"+str(score), (xmin,max(0,ymin-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[color_id], 1)
                                
        return img_show
        
def get_img_list(img_list_txt):
    img_list = []
    with open(img_list_txt) as f:
        lines = f.readlines()
        for line in lines:
            img_list.append(line.strip())
    return img_list
        

if __name__ == "__main__":
    logger.info("Start proc.")
    yolo_detector = YoloDetector(416, 416, "./save_models/yolov3_darknet53", 0, 0.3, [mx.gpu(int(0))])
    img_list_txt = "./data/test_person_car_data.txt"
    img_list = get_img_list(img_list_txt)
    for img_path in img_list:        
        #img = cv2.imread("/home/liyongjing/Egolee/programs/gluon-cv/scripts/detection/center_net/img.jpg")
        img= cv2.imread(img_path)
       # print(img.shape)
       # print(img_path.replace('.','_result.'))
        img_show = yolo_detector._detect(img)
        cv2.imwrite(img_path.replace('.jpg','_result.jpg'), img_show)
    logger.info("End proc.")
    exit(1)
            
            
