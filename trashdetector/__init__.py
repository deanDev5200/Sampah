import torch, cv2, datetime, time
print('torch %s %s' % (torch.__version__, "CUDA" if torch.cuda.is_available() else 'CPU'))
from ultralytics import YOLO
from notifypy import Notify
import os

# Create output directory
try:
    os.system('mkdir output')
except:
    pass

class TrashDetector:
    """
    Main module for Trash Detection with YOLOv8

    :param ``model_path``: Path to the trash detection model
    :param ``camera_index``: Camera index to use
    :param ``savefile``: File name to save the output as CSV
    :param ``notification_icon``: Path to the notification icon file (PNG)
    :param ``notification_audio``: Path to the notification audio file (WAV)
    :param ``confidence``: Minimum confidence for the model to detect
    """

    def __init__(self, model_path='trash.pt', camera_index=1, savefile='record.csv', notification_icon='icon.png', notification_audio='sfx.wav', confidence=0.78):
        self.model = YOLO(model_path)
        self.detects = []
        self.lastCount = 0
        self.video = cv2.VideoCapture(camera_index)
        self.lastHour = datetime.datetime.now().hour
        self.notification_icon = notification_icon
        self.notification_audio = notification_audio
        self.confidence = confidence
        self.img = None
        self.savename = savefile

    def notify(self, objects: int):

        # Create a Notify Object and set the values
        notification = Notify()
        notification.title = 'Sampah Terdeteksi!!'
        notification.message = str(objects) + ' sampah terdeteksi'
        notification.urgency = 'critical'
        notification.icon = self.notification_icon
        notification.audio = self.notification_audio
        notification.timeout = 999999

        # Send the notification
        notification.send()

    def resultProcess(self, objects: int, xyxy: list):

        # If there is object present
        if objects > 0:

                # Create a rectangle on detection bounding boxes
                for i in xyxy:
                    p1 = (int(i[0]), int(i[1]))
                    p2 = (int(i[2]), int(i[3]))
                    self.img = cv2.rectangle(self.img, p1, p2, (0, 0, 255), 5)

                # Get the current record. Format: 'month/day hour:minute, detections'
                now = ('0' if datetime.datetime.now().month<10 else '') + str(datetime.datetime.now().month) + ('/0' if datetime.datetime.now().day<10 else '/') + str(datetime.datetime.now().day) + (' 0' if datetime.datetime.now().hour<10 else ' ') + str(datetime.datetime.now().hour) + (':0' if datetime.datetime.now().minute<10 else ':') + str(datetime.datetime.now().minute) + ", " + str(objects)

                # If the current objects count isn't equal to the last objects count
                if objects != self.lastCount:

                    # Detect the values in detections list
                    isPresent = False
                    for i in self.detects:
                        if i == now:
                            isPresent = True

                    # If there is no value equals now
                    if not isPresent:

                        # Save the image with detections, append the record to detects, and show notification
                        cv2.imwrite('output/' + str(int(time.time())) + '.jpg', self.img)
                        self.detects.append(now)
                        self.lastCount = objects
                        self.notify(objects)
        else:

            # If there is no detection, set the last objects count to 0
            self.lastCount = 0

    def loop(self) -> bool:
        """
        Call this function in a loop to start detecting

        :return: Boolean value (False means the loop must stop)
        """

        # Get the current frame and assign self.img with frame
        _, frame = self.video.read()
        self.img = frame

        # Get the results from model's inference
        results = self.model(frame, stream=True, verbose=False, save=False, conf=self.confidence)
        for r in results:
            # Process the results
            self.resultProcess(len(r.boxes.xyxy.cpu().numpy()), r.boxes.xyxy.tolist())

        # Show the output on another window
        cv2.imshow('Output', self.img)

        # If 'Q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('--- Detections Until Now ---')

            # If detections aren't empty
            if len(self.detects) > 0:

                # Print the detections
                for d in self.detects:
                    print(d)

                print('Saving...')

                # Save the detections to 'output/self.savename'
                try:
                    with open('output/' + self.savename, 'x') as record:
                        tmp = ""
                        for d in self.detects:
                            tmp = tmp + d + "\n"
                        record.write(tmp)
                        record.close()
                except:
                    with open('output/' + self.savename, 'w') as record:
                        tmp = ""
                        for d in self.detects:
                            tmp = tmp + d + "\n"
                        record.write(tmp)
                        record.close()
                print(f"Saved as {self.savename}")

            else:

                # Print 'No Detection'
                print('------- No Detection -------')

            print('--- Detections Until Now ---')

        # If 'P' key is pressed
        elif cv2.waitKey(1) & 0xFF == ord('p'):

            # Return False value
            return False

        # If the hour has changed
        elif datetime.datetime.now().hour != self.lastHour:
            print('--- Detections Until Now ---')

            # If detections aren't empty
            if len(self.detects) > 0:

                # Print the detections
                for d in self.detects:
                    print(d)

                print("\nSaving...")

                # Save the detections to 'output/self.savename'
                try:
                    with open('output/' + self.savename, 'x') as record:
                        tmp = ""
                        for d in self.detects:
                            tmp = tmp + d + "\n"
                        record.write(tmp)
                        record.close()
                except:
                    with open('output/' + self.savename, 'w') as record:
                        tmp = ""
                        for d in self.detects:
                            tmp = tmp + d + "\n"
                        record.write(tmp)
                        record.close()
                print(f"Saved as {self.savename}")
                self.lastHour = datetime.datetime.now().hour

            else:

                # Print 'No Detection'
                print('------- No Detection -------')
            print('--- Detections Until Now ---')
        return True