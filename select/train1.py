from ultralytics import YOLO
ac2=YOLO('yolov8n.pt')
ac2.train(
    data= 'cccc.yaml',
    epochs=500,
    imgsz=640,
    batch=32,
    device='cpu'

)
print('model finished')