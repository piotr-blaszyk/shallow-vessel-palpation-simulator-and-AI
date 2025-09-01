import threading
import time

import cv2

from difftactile.main.constants import *


class SynchVideoRecorder:
    def __init__(self):
        pass

    def go(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, -64)
        self.cap.set(cv2.CAP_PROP_CONTRAST, 48)
        self.cap.set(cv2.CAP_PROP_SATURATION, 0)
        self.cap.set(cv2.CAP_PROP_HUE, 0)
        self.cap.set(cv2.CAP_PROP_SHARPNESS, 3)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 5000)
        self.cap.set(cv2.CAP_PROP_GAMMA, 100)
        self.cap.set(cv2.CAP_PROP_GAIN, 0)
        self.cap.set(cv2.CAP_PROP_BACKLIGHT, 0)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, 100)
        print(f"Actual camera settings:")
        print(
            f"Resolution: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}"
        )
        print(f"FPS: {self.cap.get(cv2.CAP_PROP_FPS)}")
        print(f"Exposure: {self.cap.get(cv2.CAP_PROP_EXPOSURE)}")
        print(f"White Balance: {self.cap.get(cv2.CAP_PROP_WB_TEMPERATURE)}")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        output_path = SYSTEM_PARAMS.files.robot_video
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        self.frame_lock = threading.Lock()
        self.current_frame_number = 0
        video_thread = threading.Thread(target=self.capture_video, daemon=True)
        video_thread.start()
        for i in range(10):
            time.sleep(0.5)
            with self.frame_lock:
                print(f"frame: {self.current_frame_number}")
        self.cap.release()
        self.out.release()
        video_thread.join()
        cv2.destroyAllWindows()

    def capture_video(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            with self.frame_lock:
                self.current_frame_number += 1
            self.out.write(frame)


def main():
    svr = SynchVideoRecorder()
    svr.go()


if __name__ == "__main__":
    main()
