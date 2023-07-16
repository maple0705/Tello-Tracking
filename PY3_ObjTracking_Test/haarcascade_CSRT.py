# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
#import argparse
import imutils
import time
import cv2
import os

fps = None      # FPS estimator

# initialize for tracker
OPENCV_OBJECT_TRACKERS = {
    "csrt"  : cv2.TrackerCSRT_create, 
    "kcf"   : cv2.TrackerKCF_create, 
    "mosse" : cv2.legacy.TrackerMOSSE_create
}
trackerName = "csrt"
tracker = OPENCV_OBJECT_TRACKERS[trackerName]()
initBB = None   # bounding box coordinates of the tracked object

# initialize for Haar cascade
cascade_datapath = "/Users/maple0705/miniconda3/envs/Tello/lib/python3.9/site-packages/cv2/data"
DETECTOR_PATHS = {
    "face" : cascade_datapath + '/haarcascade_frontalface_alt.xml', 
    "fullbody" : cascade_datapath + '/haarcascade_fullbody.xml', 
    "upperbody" : cascade_datapath + '/haarcascade_upperbody.xml', 
    "lowerbody" : cascade_datapath + '/haarcascade_lowerbody.xml'
}
detectors = {}  # a dictionary to store haar cascade detector
for (name, path) in DETECTOR_PATHS.items():
    print(path)
    detectors[name] = cv2.CascadeClassifier(path)
lastBodyRects = None

# grab the reference to the web cam
#print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
#time.sleep(1.0)

# grab a reference to the video file
videofile = "/Users/maple0705/Programming/Git/Tello-Tracking/PY3_ObjTracking_Test/video/1.mp4"
vs = cv2.VideoCapture(videofile)

# loop over frames from the video stream
while True:
    frame = vs.read()
    # if using a videofile
    frame = frame[1]
    if frame is None:
        print("frame is none")
        break

    # resize the frame and grab the frame dimensions
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    (H, W) = frame.shape[:2]

    # perform face detection using the appropriate haar cascade
    bodyRects = detectors["upperbody"].detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30), 
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (bX, bY, bW, bH) in bodyRects:
        cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 2)
    if len(bodyRects) != 0:
        lastBodyRects = bodyRects

    if initBB is not None:
        # grab the new bounding box coordinates of the object
        (success, box) = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        fps.update()
        fps.stop()

        info = [
            ("Tracker", trackerName), 
            ("Success", "Yes" if success else "No"), 
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

    # select a bounding box to track
    if key == ord("s"):
        if lastBodyRects is not None:
            (bX, bY, bW, bH) = lastBodyRects[0]
            initBB = (bX, bY, bX + bW, bY + bH)
            tracker.init(frame, initBB)
            fps = FPS().start()

    # break from the loop
    elif key == ord("q"):
        print("pressed key q")
        break

vs.release()
cv2.destroyAllWindows()
print("end")