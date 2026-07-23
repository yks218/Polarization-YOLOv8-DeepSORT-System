import os
import cv2
import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO


if __name__ == '__main__':

    # ================= 1. 模型路径 =================
    model = YOLO(
        r'data/yolov8_xyd2/weights/best.pt'
    )

    # ================= 2. 输入 & 输出路径 =================
    input_dir = r'data/images/val'
    save_dir = r'data/output'

    os.makedirs(save_dir, exist_ok=True)

    # ================= 3. 遍历图像 =================
    for img_name in os.listdir(input_dir):
        if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        img_path = os.path.join(input_dir, img_name)

        # ---------- (1) 读取原始彩色图 ----------
        img_color = cv2.imread(img_path)

        # ---------- (2) 直接用彩色图进行推理 ----------
        results = model.predict(
            source=img_color,
            imgsz=640,
            conf=0.25,
            verbose=False
        )

        # ---------- (3) 在原图上画检测框 ----------
        result_img = results[0].plot(img=img_color.copy())

        # ---------- (4) 保存结果 ----------
        cv2.imwrite(
            os.path.join(save_dir, img_name),
            result_img
        )

    print('目标检测完成')
    print(f'结果保存在：{save_dir}')
