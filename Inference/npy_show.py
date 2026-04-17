import cv2
import numpy as np
import os

# -----------------------------
# 1️⃣ 配置路径
# -----------------------------
video_path = "/home/wbr12589/Project/Improve/raw_video/test/test/test.mp4"  # 原视频
npy_path = "/home/wbr12589/Project/Improve/preprocessed_output/detect_speaker/test_test_test/pywork/best_persons.npy"
output_path = "/home/wbr12589/Project/Improve/preprocessed_output/test_test_test_with_bbox.mp4"  # MP4

# -----------------------------
# 2️⃣ 检查文件
# -----------------------------
if not os.path.isfile(video_path):
    raise FileNotFoundError(f"视频文件不存在: {video_path}")
if not os.path.isfile(npy_path):
    raise FileNotFoundError(f"npy 文件不存在: {npy_path}")

# -----------------------------
# 3️⃣ 读取 npy
# -----------------------------
best_persons = np.load(npy_path)
print("best_persons shape:", best_persons.shape)

# -----------------------------
# 4️⃣ 打开视频
# -----------------------------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("视频无法打开，请检查路径和文件是否可读")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 使用 MP4 codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 也可以尝试 'H264' 或 'avc1'
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# -----------------------------
# 5️⃣ 遍历视频帧并画 bbox
# -----------------------------
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx < len(best_persons):
        bbox = best_persons[frame_idx]  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width-1, x2), min(height-1, y2)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 3)
            cv2.putText(frame, "BEST_PERSON", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print(f"✅ 可视化完成，MP4 输出保存到: {output_path}")