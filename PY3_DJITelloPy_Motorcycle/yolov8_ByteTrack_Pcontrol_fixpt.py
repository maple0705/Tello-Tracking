from djitellopy import Tello, TelloException
from imutils.video import FPS
from ultralytics import YOLO
import cv2
import pygame
import numpy as np
import time
import threading
import logging
import colorama

colorama.init(autoreset=True)

# Speed of the drone (0~100)
S = 60

model = YOLO("yolov8n.pt")
ROIfound = False
id_ROI = 999

deadband = 200 # [px]

fps = FPS().start()

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

class FrontEnd(object):
    """ Maintains the Tello display and moves it through the keyboard keys.
        Press escape key to quit.
        The controls are:
            - T: Takeoff
            - L: Land
            - Arrow keys: Forward, backward, left and right.
            - A and D: Counter clockwise and clockwise rotations (yaw)
            - W and S: Up and down.
    """

    def __init__(self):
        # Init pygame
        pygame.init()

        # Creat pygame window
        pygame.display.set_caption("Tello video stream")
        self.screen = pygame.display.set_mode([960, 720])

        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()
        #self.tello.LOGGER.setLevel(logging.WARNING)

        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10

        self.send_rc_control = False
        self.auto_pilot = False

        # Frames per second of the pygame window display
        # A low number also results in input lag, as input information is processed once per frame.
        self.fps = 120

        # create update timer
        pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // self.fps)

        # the sending_command will send command to tello every 10 seconds
        self.sending_command_thread = threading.Thread(target = self._sendingCommand)

    def run(self):

        self.tello.connect()
        self.tello.set_speed(self.speed)

        # In case streaming is on. This happens when we quit this program without the escape key.
        self.tello.streamoff()
        self.tello.streamon()

        frame_read = self.tello.get_frame_read()

        self.sending_command_thread.setDaemon(True)
        self.sending_command_thread.start()

        should_stop = False
        while not should_stop:
            if frame_read.stopped:
                break

            self.screen.fill([0, 0, 0])

            frame = frame_read.frame
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # no need for my Tello
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
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(
                        frame, 
                        "#"+str(id)+" "+model.names[int(cls)], (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_PLAIN, 1.3, (255, 0, 0), 1
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
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), thickness=-1)
                    # P(ID) control if flying
                    if self.auto_pilot:
                        gain_yaw = 0.2
                        error = W / 2 - center_x
                        if abs(error) < deadband / 2: error = 0
                        self.left_right_velocity = 0
                        self.for_back_velocity   = 0
                        self.up_down_velocity    = 0
                        self.yaw_velocity        = int(np.clip(-1 * int(gain_yaw * error), -100, 100))


            # draw Tello info
            fps.update()
            fps.stop()
            battery = self.tello.get_battery()
            temperature = self.tello.get_temperature()
            info = [
                ("FPS", "{:.2f}".format(fps.fps())), 
                ("Battery", "{}%".format(battery)), 
                ("Temp", "{}C".format(temperature)), 
                ("Auto Pilot", "ON" if self.auto_pilot else "OFF"), 
            ]
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)), 
                            cv2.FONT_HERSHEY_PLAIN, 1.3, (255, 0, 0), 1
                )
            # draw deadband
            for y in range(0, 31, 2):
                x = W // 2 - deadband // 2
                y1 = int(y / 31 * H)
                y2 = int((y + 1) / 31 * H)
                cv2.line(frame, (x, y1), (x, y2), (0, 127, 0), 2)
                x = W // 2 + deadband // 2
                cv2.line(frame, (x, y1), (x, y2), (0, 127, 0), 2)

            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame = cv2.flip(frame, 1)
            frame = pygame.surfarray.make_surface(frame)
            self.screen.blit(frame, (0, 0))
            pygame.display.update()

            # adjust FPS
            time.sleep(1 / self.fps)

            # key input detection
            for event in pygame.event.get():
                if event.type == pygame.USEREVENT + 1:
                    self.update()
                elif event.type == pygame.QUIT:
                    should_stop = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        should_stop = True
                    else:
                        self.keydown(event.key)
                elif event.type == pygame.KEYUP:
                    self.keyup(event.key)

        # Call it always before finishing. To deallocate resources.
        self.tello.end()

    def keydown(self, key):
        """ Update velocities based on key pressed
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP:  # set forward velocity
            self.for_back_velocity = S
        elif key == pygame.K_DOWN:  # set backward velocity
            self.for_back_velocity = -S
        elif key == pygame.K_LEFT:  # set left velocity
            self.left_right_velocity = -S
        elif key == pygame.K_RIGHT:  # set right velocity
            self.left_right_velocity = S
        elif key == pygame.K_w:  # set up velocity
            self.up_down_velocity = S
        elif key == pygame.K_s:  # set down velocity
            self.up_down_velocity = -S
        elif key == pygame.K_a:  # set yaw counter clockwise velocity
            self.yaw_velocity = -S
        elif key == pygame.K_d:  # set yaw clockwise velocity
            self.yaw_velocity = S

    def keyup(self, key):
        """ Update velocities based on key released
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP or key == pygame.K_DOWN:  # set zero forward/backward velocity
            self.for_back_velocity = 0
        elif key == pygame.K_LEFT or key == pygame.K_RIGHT:  # set zero left/right velocity
            self.left_right_velocity = 0
        elif key == pygame.K_w or key == pygame.K_s:  # set zero up/down velocity
            self.up_down_velocity = 0
        elif key == pygame.K_a or key == pygame.K_d:  # set zero yaw velocity
            self.yaw_velocity = 0
        elif key == pygame.K_t:  # takeoff
            try:
                self.tello.takeoff()
                self.send_rc_control = True
            except(TelloException):
                print(colorama.Fore.RED + '!Low Battery!')
        elif key == pygame.K_l:  # land
            not self.tello.land()
            self.send_rc_control = False
            self.auto_pilot = False
        elif key == pygame.K_p: # auto pilot(PID control, P term)
            if self.send_rc_control and not self.auto_pilot:
                self.auto_pilot = True
            elif self.auto_pilot:
                self.auto_pilot = False
                self.left_right_velocity = 0
                self.for_back_velocity   = 0
                self.up_down_velocity    = 0
                self.yaw_velocity        = 0

    def update(self):
        """ Update routine. Send velocities to Tello.
        """
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity,
                self.up_down_velocity, self.yaw_velocity)
    
    def _sendingCommand(self):
        """
        start a while loop that sends 'command' to Tello every 10 second
        """    
        while True:
            self.tello.send_control_command("command")
            time.sleep(10)


def main():
    frontend = FrontEnd()

    # run frontend
    frontend.run()


if __name__ == '__main__':
    main()
