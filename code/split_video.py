import cv2
import os

input_path = 'video/video_0000.avi'
output_dir = 'video'

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("无法打开视频")
    exit(1)

total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 每段帧数
part_size = total // 4  # 159
splits = [
    (0, part_size),           # part1: 0-158
    (part_size, part_size*2), # part2: 159-317
    (part_size*2, part_size*3), # part3: 318-476
    (part_size*3, total),     # part4: 477-637
]

# 用 MP4V 编码（兼容性好）
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

writers = []
for i in range(4):
    out_path = os.path.join(output_dir, f'video_part{i+1}.mp4')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    writers.append(writer)
    print(f"Part {i+1}: 0-{total-1} 帧 -> {out_path}")

frame_idx = 0
part_idx = 0
next_boundary = splits[0][1]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 写入当前段
    writers[part_idx].write(frame)

    frame_idx += 1

    # 到达分段边界
    if frame_idx >= next_boundary and part_idx < 3:
        part_idx += 1
        next_boundary = splits[part_idx][1]

cap.release()
for w in writers:
    w.release()

print(f"\n完成！共 {frame_idx} 帧，切割为 4 段。")
print(f"输出: video_part1.mp4 ~ video_part4.mp4")
