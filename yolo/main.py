from ultralytics import YOLO
a2=YOLO(r'C:\Users\Lenovo\pythonProject\aliOCR\xuehao\runs\detect\train\weights\best.pt')
a2('IMG_8402.jpg',show=True,save=True)
