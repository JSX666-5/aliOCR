from ultralytics import YOLO
a2=YOLO(r'D:\pythonProjectceshi\runs\detect\train\weights\best.pt')
a2('IMG_8402.JPG',show=True,save=True)

