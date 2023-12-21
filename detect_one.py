# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

import argparse
import sys,os
from pathlib import Path
import numpy as np
import cv2
import torch,time
import torch.backends.cudnn as cudnn



FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams,letterbox
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

def infer(img0,model,device,conf_thres,iou_thres):
    # pre process
    img = letterbox(img0, new_shape=[416,416], stride=model.stride, auto=model.pt)[0]
    img = img[:,:,::-1].transpose(2,0,1).copy().astype(np.float32)
    img /= 255  # 0 - 255 to 0.0 - 1.0
    img = img[None]  # expand for batch dim
    # inference
    img = torch.tensor(img).to(device)
    pred = model(img, augment=False, visualize=False)
    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False, max_det=1000)
    # Process predictions
    for i, det in enumerate(pred):  # per image
        annotator = Annotator(img0, line_width=3, example=str(model.names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            # Write results
            hide_labels, hide_conf = False, False
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = None if hide_labels else (model.names[c] if hide_conf else f'{model.names[c]} {conf:.2f}')
                annotator.box_label(xyxy, label, color=colors(c, True))
        # Stream results
        im0 = annotator.result()
    return im0

def detect_one(img0):
    weights="yolov5_traffic/runs/train/exp2/weights/best.pt"
    img0=cv2.imread(img0)
    conf_thres, iou_thres = 0.25, 0.45
    # Load model
    device = select_device("0")
    model = DetectMultiBackend(weights, device=device)
    model.model.float()
    im0 = infer(img0, model, device, conf_thres, iou_thres)
    return im0
        # im0 = cv2.resize(im0, (512, 288))
        # cv2.imshow('img', im0)
        # cv2.waitKey(0)  # 1 millisecond

def detect_file(model,source):
    dataset = LoadImages(source, img_size=[416,416], stride=model.stride, auto=model.pt)
    conf_thres, iou_thres = 0.25, 0.45
    device = select_device("0")
    # model.warmup(imgsz=(1 if model.pt else 1, 3, *imgsz), half=False)  # warm
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        pred = model(im, augment=False, visualize=False)
        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False, max_det=1000)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            s += '%gx%g ' % im.shape[2:]  # print string
            annotator = Annotator(im0, line_width=3, example=str(model.names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {model.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                # Write results
                hide_labels,hide_conf=False,False
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = None if hide_labels else (model.names[c] if hide_conf else f'{model.names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
            # Stream results
            im0 = annotator.result()
            im0=cv2.resize(im0,(512,288))
            cv2.imshow(str(p), im0)
            cv2.waitKey(0)  # 1 millisecond


def detect_vedio(video_path,video_save_path=''):
    device = select_device("0")
    capture = cv2.VideoCapture(video_path)
    if video_save_path != "":
        video_fps = 25.0
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)
    weights="F:/python/protect/交通标志检测系统/yolo5_Web/yolov5_traffic/runs/train/exp2/weights/best.pt"
    # image=cv2.imread(pics)
    conf_thres, iou_thres = 0.25, 0.45
    # Load model
    model = DetectMultiBackend(weights, device=device)
    model.model.float()
    ref, frame = capture.read()
    if not ref:
        raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")
    fps = 0.0
    while (True):
        t1 = time.time()
        # 读取某一帧
        ref, frame = capture.read()
        if not ref:
            break
        frame = infer(frame, model, device, conf_thres, iou_thres)
        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %.2f" % (fps))
        frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('img',frame)
        if cv2.waitKey(1)&0xff==27:
            break
        if video_save_path != "":
            out.write(frame)
    print("Video Detection Done!")
    capture.release()
    if video_save_path!="":
        print("Save processed video to the path :" + video_save_path)
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # pics= cv2.imread('/home/lqs/Documents/yolov5/Picture/yolov5_traffic/data/images/39.jpg')
    # imgsz = check_img_size(imgsz, s=model.stride)  # check image size
    # detect_one(pics)
    detect_vedio('data/video/traffic.mp4',video_save_path='data/video/save/save.mp4')
    # detect_file(model,device,pics)
