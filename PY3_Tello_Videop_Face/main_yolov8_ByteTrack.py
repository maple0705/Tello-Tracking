from imutils.video import FPS
from ultralytics import YOLO
import tello
import time
import cv2
import numpy as np


# -----main-----
drone = tello.Tello('', 8889, command_timeout=.01)
battery = -1
temperature = -1
current_time = time.time()
pre_time = current_time     # 5秒ごとの'command'送信のための時刻変数

model = YOLO("yolov8n.pt")
ROIfound = False
id_ROI = 999

time.sleep(3.0)     # 通信が安定するまでちょっと待つ

fps = FPS().start()
drone.send_command("command")
print("capture start")
while True:
    # (A)画像取得
    frame_raw = drone.read()    # 映像を1フレーム取得
    if frame_raw is None or frame_raw.size == 0:    # 中身がおかしかったら無視
        continue 

    # (B)ここから画像処理
    frame = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2BGR)      # OpenCV用のカラー並びに変換する
    #frame = cv2.resize(frame, dsize=(640, 480))
    (H, W) = frame.shape[:2]

    # MOT
    stillExistROI = False
    rect_ROI = None
    results = model.track(
        source  = frame, 
        #conf    = 0.5, 
        #iou     = 0.3, 
        persist = True,             # for tracking
        device  = "mps",            # for M1 mac
        classes = [0],              # person
        tracker = "bytetrack.yaml", 
        stream  = True, 
        verbose = False,            # disable output on terminal
    )
    for result in results:
        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        classes = np.array(result.boxes.cls.cpu(), dtype="int")
        if result.boxes.id is not None:
            ids = np.array(result.boxes.id.cpu(), dtype="int")
        else:
            ids = np.empty(0)

        # transfer information to the image
        for cls, bbox, id in zip(classes, bboxes, ids):
            (x1, y1, x2, y2) = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame, 
                "#"+str(id)+" "+model.names[int(cls)], (x1, y1 - 5), 
                cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 225), 1
            )
            # detection person(ROI)
            if not ROIfound and int(cls) == 0:   # 0: person
                ROIfound = True
                stillExistROI = True
                rect_ROI = bbox
                id_ROI = id
            # refresh ROI infomation
            elif id == id_ROI:
                stillExistROI = True
                rect_ROI = bbox

        if not stillExistROI:
            ROIfound = False

        if rect_ROI is not None:
            (x1, y1, x2, y2) = rect_ROI
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


    # (X)ウィンドウに表示
    # draw FPS info
    #time.sleep(0.01)
    fps.update()
    fps.stop()
    info = [
        ("FPS", "{:.2f}".format(fps.fps())), 
        ("Battery", "{}%".format(battery)), 
        ("Temp", "{}C".format(temperature)), 
    ]
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)), 
                    cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 255, 255), 1
        )
    cv2.imshow('Tello camera', frame)

    # (Y)OpenCVウィンドウでキー入力を1ms待つ
    key = cv2.waitKey(1)
    if key == ord("q"):
        print("pressed key q")
        break
    elif key == ord("b"):
        battery = drone.get_battery()
        print("{}% battery remaining.".format(battery))
    elif key == ord("h"):
        height = drone.get_height()
        print(height)
    elif key == ord("u"):
        rising = drone.move_up(0.2)
        print(rising)
    elif key == ord("d"):
        falling = drone.move_down(0.2)
        print(falling)
    elif key == ord('t'):
        drone.takeoff()
    elif key == ord('l'):
        drone.send_command('rc 0 0 0 0')
        drone.land()
        time.sleep(3)   # 着陸するまで他のコマンドを打たないよう，ウェイトを入れる

    # (Z)5秒おきに'command'を送って、死活チェックを通す
    # print battery remaining
    current_time = time.time()
    if current_time - pre_time > 10.0 :
        pre_time = current_time
        drone.send_command('command')

        #temperature = drone.get_temp()
        #print("{}C".format(temperature))


while True:
    #stopping = drone.send_command('rc 0 0 0 0')
    #print("stopping = " + stopping)
    #time.sleep(1)
    landing = drone.land()
    print("landing = " + landing)
    if landing == "ok":
        break
    time.sleep(1)
del drone
