import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import YOLO



if __name__ == '__main__':

    model = YOLO("yolov8n_cbam.yaml")
    model.load("yolov8n.pt")  # 🔥 加这一行

    model.train(data=r'data/data_xyd.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=8,
                close_mosaic=0,
                workers=2, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                optimizer='SGD', # using SGD
                project='data',
                name='yolov8cbam_xyd',
                )