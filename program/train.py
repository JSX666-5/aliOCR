from ultralytics import YOLO
a2=YOLO('yolov8n.pt')
a2.train(
    data= 'dddd.yaml',
    epochs=200,
    imgsz=640,
    batch=32,
    device='cpu'

)
print('model finished')