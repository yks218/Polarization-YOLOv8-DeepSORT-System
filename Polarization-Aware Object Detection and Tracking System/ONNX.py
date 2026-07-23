import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("data/yolov8cbam_xyd/weights/best.onnx")

input_name = session.get_inputs()[0].name

dummy = np.random.randn(1, 3, 640, 640).astype(np.float32)

outputs = session.run(None, {input_name: dummy})

print("ONNX推理成功！输出shape：", [o.shape for o in outputs])