from ultralytics import YOLO
a1=YOLO(r'D:\aliOCR\select\runs\detect\train\weights\best.pt')
a1('IMG_8372.JPG',show=True,save=True,show_labels=False, show_conf=False)