# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from ultralytics import YOLO
import imutils
import time
import cv2
import os
import numpy as np

# rect1: motorcycle, rect2: people
def checkRectOverlap(frame, rect1, rect2):
    (r1_x1, r1_y1, r1_x2, r1_y2) = rect1
    (r2_x1, r2_y1, r2_x2, r2_y2) = rect2

    c1_x = int((r1_x1+r1_x2)/2)
    c1_y = int((r1_y1+r1_y2)/2)
    w1   = r1_x2-r1_x1
    h1   = r1_y1-r1_y2

    c2_x = int((r2_x1+r2_x2)/2)
    c2_y = int((r2_y1+r2_y2)/2)
    w2   = r2_x2-r2_x1
    h2   = r2_y1-r2_y2

    cv2.circle(frame, (c1_x, c1_y), 5, (0, 0, 225), thickness=-1)
    cv2.circle(frame, (c2_x, c2_y), 5, (0, 0, 225), thickness=-1)

    # detect rectangle overlap
    if (max(r1_x1, r2_x1) < min(r1_x2, r2_x2)) and (max(r1_y1, r2_y1) < min(r1_y2, r2_y2)):
        #detect people riding on the motorcycle
        if r1_x1 < c2_x and c2_x < r1_x2:
            return True
    return False

# ----- main -----
model = YOLO("yolov8n.pt")
ROIfound = False
id_ROI = 999

# grab a reference to the video file
videofile = "/Users/maple0705/Programming/Git/Tello-Tracking/PY3_ObjTracking_Test/video/1.mp4"
cap = cv2.VideoCapture(videofile)
fps = FPS().start()

# loop over frames from the video stream
while True:
    frame = cap.read()
    # if using a videofile
    frame = frame[1]
    if frame is None:
        print("frame is none")
        break

    # resize the frame and grab the frame dimensions
    #frame = imutils.resize(frame, width=640)
    (H, W) = frame.shape[:2]

    # MOT
    stillExistROI = False
    rect_motorcycle = None
    rect_ROI = None
    results = model.track(
        source  = frame, 
        #conf    = 0.5, 
        #iou     = 0.3, 
        persist = True,             # for tracking
        device  = "mps",            # for M1 mac
        classes = [0, 3],           # person, motorcycle
        tracker = "bytetrack.yaml", 
        stream  = True, 
        verbose = False,            # disable output on terminal
    )
    #result = results[0]    # stream=False
    for result in results:  # stream=True
        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        classes = np.array(result.boxes.cls.cpu(), dtype="int")
        if result.boxes.id is not None:
            ids = np.array(result.boxes.id.cpu(), dtype="int")

        # transfer information to the image
        for cls, bbox, id in zip(classes, bboxes, ids):
            (x1, y1, x2, y2) = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame, 
                "#"+str(id)+" "+model.names[int(cls)], (x1, y1 - 5), 
                cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 225), 2
            )
            # detection people(ROI) who is riding the motorcycle
            if not ROIfound and int(cls) == 3:   # 3: motorcycle
                rect_motorcycle = bbox
                for _cls, _bbox, _id in zip(classes, bboxes, ids):
                    if int(_cls) != 0: continue
                    isOverlap = checkRectOverlap(frame, rect_motorcycle, _bbox)
                    if isOverlap:
                        ROIfound = True
                        stillExistROI = True
                        rect_ROI = _bbox
                        id_ROI = _id
            # refresh ROI infomation
            elif id == id_ROI:
                stillExistROI = True
                rect_ROI = bbox

        if not stillExistROI:
            ROIfound = False

        if rect_ROI is not None:
            (x1, y1, x2, y2) = rect_ROI
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # draw FPS info
    fps.update()
    fps.stop()
    info = [
        ("FPS", "{:.2f}".format(fps.fps())), 
    ]
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # break from the loop
    if key == ord("q"):
        print("pressed key q")
        break

cap.release()
cv2.destroyAllWindows()
print("end")
