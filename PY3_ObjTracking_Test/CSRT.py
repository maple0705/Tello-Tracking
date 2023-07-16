# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
#import argparse
import imutils
import time
import cv2

OPENCV_OBJECT_TRACKERS = {
    "csrt"  : cv2.TrackerCSRT_create, 
    "kcf"   : cv2.TrackerKCF_create, 
    "mosse" : cv2.legacy.TrackerMOSSE_create
}
trackerName = "csrt"
tracker = OPENCV_OBJECT_TRACKERS[trackerName]()

initBB = None

# grab the reference to the web cam
#print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
#time.sleep(1.0)

# grab a reference to the video file
videofile = "/Users/maple0705/Programming/Git/Tello-Tracking/PY3_ObjTracking_Test/video/1.mp4"
vs = cv2.VideoCapture(videofile)

fps = None

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
    (H, W) = frame.shape[:2]

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
        # make sure to press ENTER or SPACE after selecting the ROI
        # press ESC to reselect the ROI
        initBB = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        tracker.init(frame, initBB)
        fps = FPS().start()

    # break from the loop
    elif key == ord("q"):
        print("pressed key q")
        break

vs.release()
cv2.destroyAllWindows()
print("end")