from trashdetector import TrashDetector

trash = TrashDetector(model_path='trash.pt')

while True:
    state = trash.loop()
    if not state:
        break
