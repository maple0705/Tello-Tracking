# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from ultralytics import YOLO
import imutils
import time
import cv2
import os
import numpy as np

model = YOLO("yolov8n.pt")
#model = YOLO("yolov8m.pt")

fps = None      # FPS estimator

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
    frame = imutils.resize(frame, width=640)
    (H, W) = frame.shape[:2]

    # MOT
    #results = model(frame, device="mps")
    results = model(frame, device="mps", classes=[0, 3])    #person, motorcycle
    result = results[0]
    bboxes = result.boxes.xyxy
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")

    # transfer information to the image
    #for cls, bbox, id in zip(classes, bboxes, ids):
    for cls, bbox in zip(classes, bboxes):
        (x, y, x2, y2) = bbox
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
        #cv2.putText(frame, str(cls), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)
        cv2.putText(frame, model.names[int(cls)], (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 225), 2)

    fps.update()
    fps.stop()

    info = [
        ("FPS", "{:.2f}".format(fps.fps())), 
    ]

    # loop over the info tuples and draw them on our frame
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