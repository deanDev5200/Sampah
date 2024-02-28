from trashdetector import TrashDetector

trash = TrashDetector(model_path='raspberrypi/trash-s_openvino_model/')

while True:
    state = trash.loop()
    if not state:
        break
