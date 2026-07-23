from collections import defaultdict, deque
import time
import cv2
from deep_sort.deep_sort import DeepSort
import numpy as np

deepsort = DeepSort("deep_sort/deep/checkpoint/ckpt.t7")

track_history = defaultdict(lambda: deque(maxlen=30))
track_last_y = {}

in_count = 0
out_count = 0
total_ids = set()

prev_time = 0
line_y = 500

def process_detection_results(model, frame, object_class=None):

    results = model(frame, verbose=False)

    xywhs = []
    confs = []

    if results[0].boxes is None:
        return [], [], frame

    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    for box, score, cls in zip(boxes, scores, classes):

        cls_name = model.names[int(cls)]

        # 👉 只检测人
        if object_class and cls_name not in object_class:
            continue

        x1, y1, x2, y2 = box

        w = x2 - x1
        h = y2 - y1
        x_c = x1 + w / 2
        y_c = y1 + h / 2

        xywhs.append([x_c, y_c, w, h])
        confs.append(score)

    if len(xywhs) == 0:
        return [], [], frame

    xywhs = np.array(xywhs)
    confs = np.array(confs)

    # 👉 DeepSORT
    classes = np.zeros(len(xywhs))  # 全部设为 person 类

    outputs = deepsort.update(xywhs, confs, classes, frame)

    bboxes_tracker = []

    for output in outputs:
        x1, y1, x2, y2, cls_id, track_id = output
        bboxes_tracker.append([x1, y1, x2, y2, cls_id, track_id])

        # 画框
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(frame, f"ID:{track_id}", (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    return xywhs, bboxes_tracker, frame

def process_video_frame(frame, model):

    global track_last_y, track_history
    global in_count, out_count, total_ids, prev_time

    bbox_coordinates, bboxes_tracker, frame_copy = process_detection_results(
        model, frame, object_class=['person']
    )

    for x1, y1, x2, y2, cls_id, track_id in bboxes_tracker:
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        if track_id in track_last_y:
            prev_y_id = track_last_y[track_id]

            if prev_y_id < line_y and cy >= line_y:
                in_count += 1
            elif prev_y_id > line_y and cy <= line_y:
                out_count += 1

        track_last_y[track_id] = cy
        track_history[track_id].append((cx, cy))

        # 画轨迹
        for i in range(1, len(track_history[track_id])):
            cv2.line(frame_copy,
                     track_history[track_id][i - 1],
                     track_history[track_id][i],
                     (0, 255, 255), 2)

    current_ids = set()

    for _, _, _, _, _, track_id in bboxes_tracker:
        current_ids.add(track_id)
        total_ids.add(track_id)

    # 画线
    cv2.line(frame_copy, (0, line_y), (frame_copy.shape[1], line_y), (255, 0, 255), 2)

    # 统计信息
    person_count = len(current_ids)

    cv2.putText(frame_copy, f"In: {in_count}", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame_copy, f"Out: {out_count}", (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame_copy, f"Count: {person_count}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame_copy, f"Total: {len(total_ids)}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    cv2.putText(frame_copy, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame_copy, (fps, person_count, len(total_ids), in_count, out_count)