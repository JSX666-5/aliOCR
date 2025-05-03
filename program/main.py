from ultralytics import YOLO
a2=YOLO(r'D:\aliOCR\program\runs\detect\train\weights\best.pt')
a2('1540.JPG',show=True,save=True,show_labels=False, show_conf=False)