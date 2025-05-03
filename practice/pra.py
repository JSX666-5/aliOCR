from ultralytics import YOLO
a1=YOLO(r'D:\aliOCR\practice\runs\detect\train\weights\best.pt')
a1('output_gray2.jpg',show=True,save=True,show_labels=False, show_conf=False)#show_labels=False, show_conf=False