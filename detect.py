import argparse
import csv
import os
import platform
import sys
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
from pathlib import Path
import telepot
bot=telepot.Bot("7663560551:AAGkiMVYvb22jgAykx4elMpZQgyseW1BAQs")
bot1=telepot.Bot("8052306020:AAFI2pBpdQ_h7CdhefZXGeHap4M1_MGTrsg")
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode
import shutil

import pygame
import time
from gtts import gTTS
from mutagen.mp3 import MP3
import time
import csv
from datetime import datetime

def Play(text1):
    myobj = gTTS(text=text1, lang='en-us', tld='com', slow=False)
    myobj.save("voice.mp3")
    print('\n------------Playing--------------\n')
    song = MP3("voice.mp3")
    pygame.mixer.init()
    pygame.mixer.music.load('voice.mp3')
    pygame.mixer.music.play()
    time.sleep(song.info.length)
    pygame.quit()

#A Gender and Age Detection program by Mahesh Sawant

import cv2
import math
import argparse

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes


parser=argparse.ArgumentParser()
parser.add_argument('--image')

args=parser.parse_args()

faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

video=cv2.VideoCapture(args.image if args.image else 0)
padding=20

import cv2
from myUtils import *
from tracking.tracking import Tracking
from tracking.unit_object import UnitObject

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_csv=False,  # save results in CSV format
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    print(names)
    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    tracker = Tracking()
    prev_frame_time = time.time()
    prev_positions = {}
    ids = {}

    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / 'predictions.csv'

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            data = {'Image Name': image_name, 'Prediction': prediction, 'Confidence': confidence}
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            bboxes = []
            coordinates = []
            names1 = []
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f'{names[c]}'
                    confidence = float(conf)
                    confidence_str = f'{confidence:.2f}'

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        anim = names[c]
                        if anim in ['Bear', 'Cheetah', 'Elephant', 'Hedgehog', 'Leopard', 'Lion','Tiger']:
                            label = f'{anim} {conf:.2f}'
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            # Play(anim)
                            x, y, w, h = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                            c1, c2 = (x, y), (w, h)
                            bboxes.append([c1, c2])
                            names1.append([anim, 'None', 'None'])
                            bot.sendMessage("2079890697",str("The detected wild animal is {}".format(anim)))
                            bot1.sendMessage("6265422479",str("The detected wild animal is {}".format(anim)))
                            cv2.imwrite('frame.png', im0)
                            bot.sendPhoto('2079890697', photo = open('frame.png', 'rb'))
                            bot1.sendPhoto('6265422479', photo = open('frame.png', 'rb'))
                            
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            try:
                resultImg,faceBoxes=highlightFace(faceNet,im0)
                if not faceBoxes:
                    print("No face detected")

                for faceBox in faceBoxes:
                    print(faceBox)
                    x, y, w, h = faceBox
                    c1, c2 = (x, y), (w, h)
                    bboxes.append([c1, c2])
                    
                    face=im0[max(0,faceBox[1]-padding):
                            min(faceBox[3]+padding,im0.shape[0]-1),max(0,faceBox[0]-padding)
                            :min(faceBox[2]+padding, im0.shape[1]-1)]

                    blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
                    genderNet.setInput(blob)
                    genderPreds=genderNet.forward()
                    gender=genderList[genderPreds[0].argmax()]
                    print(f'Gender: {gender}')

                    ageNet.setInput(blob)
                    agePreds=ageNet.forward()
                    age=ageList[agePreds[0].argmax()]

                    names1.append(['Human', gender, age])
                    cv2.putText(im0, f'{gender}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
            except Exception as e:
                print(e)
                pass

            for box in bboxes:
                coordinates.append(UnitObject( [box[0][0], box[0][1], box[1][0], box[1][1] ], 1))

            tracker.update(coordinates)

            current_frame_time = time.time()
            elapsed_time = current_frame_time - prev_frame_time

            # Function to read existing names from the CSV file
            def read_existing_names(file_path):
                existing_names = set()
                try:
                    with open(file_path, mode='r', newline='') as file:
                        reader = csv.reader(file)
                        # Skip header row
                        next(reader, None)
                        # Add all existing object names to the set
                        for row in reader:
                            existing_names.add(row[0])  # Assuming name is the first column
                except FileNotFoundError:
                    # If the file does not exist, we'll simply return an empty set
                    pass
                return existing_names

            # Initialize a set with existing names from the CSV
            file_path = 'tracked_objects.csv'
            stored_names = read_existing_names(file_path)

            # Open the CSV file in append mode
            with open(file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                
                # Write the header if the file is empty (first run)
                if file.tell() == 0:
                    writer.writerow(['id', 'Object Name', 'Date', 'Time', 'Gender', 'Age'])

                for j in range(len(tracker.tracker_list)):
                    try:
                        unit_object = tracker.tracker_list[j].unit_object
                        tracking_id = tracker.tracker_list[j].tracking_id
                        name = names1[j][0]
                        Gender = names1[j][1]
                        Age = names1[j][2]
                        if tracking_id in ids:
                            # Calculate distance traveled
                            prev_x, prev_y = ids[tracking_id]
                            current_x, current_y = int(unit_object.box[0]), int(unit_object.box[1])
                            distance = ((current_x - prev_x)**2 + (current_y - prev_y)**2)**0.5
                            cv2.putText(im0, f"{tracking_id}", (current_x+5, current_y+20), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 255, 0), 3)

                        # Ensure that the name is not already in the stored_names set
                        if tracking_id not in stored_names:
                            # Get the current date and time
                            current_datetime = datetime.now()
                            date_str = current_datetime.strftime('%Y-%m-%d')
                            time_str = current_datetime.strftime('%H:%M:%S')

                            # Write to CSV: object_name, date, time
                            writer.writerow([tracking_id, name, date_str, time_str, Gender, Age])

                            # Add the object name to the stored_names set to prevent repetition
                            stored_names.add(name)

                        # Update the previous position for the next frame
                        prev_positions[name] = (int(unit_object.box[0]), int(unit_object.box[1]))
                        ids[tracking_id] = (int(unit_object.box[0]), int(unit_object.box[1]))
                        
                    except:
                        continue
                # Update the previous frame time for the next iteration
                prev_frame_time = current_frame_time

            
            # Stream results
            im0 = annotator.result()
            cv2.imshow(str(p), im0)
            if cv2.waitKey(1) & 0xFF == ord('q'): # 1 millisecond
                exit()

            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
                shutil.copy(save_path, 'static/result')
            else:  # 'video' or 'stream'
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
                    shutil.copy(save_path, 'static/result')
                    
def parse_opt(File):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'wild.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=File, help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-csv', action='store_true', help='save results in CSV format')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))



def Start(File):
    if os.path.exists('tracked_objects.csv'):
        os.remove('tracked_objects.csv')
        print("\n\n\n removed tracked_objects.csv \n\n\n ")
    
    if os.path.exists('tracked_persons.csv'):
        os.remove('tracked_persons.csv')
        print("\n\n\n removed tracked_persons.csv \n\n\n ")

    opt = parse_opt(File)
    main(opt)

f = open('video.txt', 'r')
File = f.read()
f.close()
Start(File)