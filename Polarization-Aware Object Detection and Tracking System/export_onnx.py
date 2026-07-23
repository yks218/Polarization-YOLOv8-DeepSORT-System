from ultralytics import YOLO

model = YOLO(r"data/yolov8cbam_xyd/weights/best.pt")

model.export(format="onnx", opset=12, dynamic=True)