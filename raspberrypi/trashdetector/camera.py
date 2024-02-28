import cv2, threading

class Camera(threading.Thread):
    def __init__(self, cam_idx:int):
        self.cam = cv2.VideoCapture(cam_idx)
        self.running = False
        self.run = True
        self = threading.Thread(target=self.loop)
        self.start()

    def loop(self):
        while self.run:
            _, frame = self.cam.read()
            self.frame = frame
            if not self.running:
                self.running = True
        self.running = False
    
    def stop(self):
        while self.running:
            self.run = False
    
    def get_frame(self):
        return self.frame
