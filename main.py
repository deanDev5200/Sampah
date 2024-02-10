from trashdetector import TrashDetector

trash = TrashDetector()

while True:
    state = trash.loop()
    if not state:
        break
