import cv2
import os
import time

# ===== 初始化摄像头（与你目前能用的版本完全一致）=====
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

os.makedirs("save/left", exist_ok=True)
os.makedirs("save/right", exist_ok=True)
os.makedirs("video", exist_ok=True)
cnt = 0
video_cnt = 0
recording = False
out = None

# 帧率计算
prev_time = time.time()
fps = 0.0

# 创建左右独立窗口（可缩放）
cv2.namedWindow("Left Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Left Camera", 640, 360)
cv2.namedWindow("Right Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Right Camera", 640, 360)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 计算帧率
    curr_time = time.time()
    delta = curr_time - prev_time
    if delta > 0:
        fps = 1.0 / delta
    prev_time = curr_time

    h, w = frame.shape[:2]
    left, right = frame[:, :w//2], frame[:, w//2:]

    # 在左图上添加文字提示（不影响原始图像）
    left_disp = left.copy()
    cv2.putText(left_disp, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    if recording:
        cv2.putText(left_disp, "REC", (left_disp.shape[1]-100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    # 显示左右独立窗口
    cv2.imshow("Left Camera", left_disp)
    cv2.imshow("Right Camera", right)

    # 录像：写入原始拼接帧（不含文字）
    if recording and out is not None:
        out.write(frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 32:   # 空格：拍照
        cv2.imwrite(f"save/left/{cnt:04d}.png", left)
        cv2.imwrite(f"save/right/{cnt:04d}.png", right)
        print(f"Saved {cnt}")
        cnt += 1

    elif key == ord('r') or key == ord('R'):   # R 键：录像
        if not recording:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            video_path = f"video/video_{video_cnt:04d}.avi"
            out = cv2.VideoWriter(video_path, fourcc, 30,
                                  (frame.shape[1], frame.shape[0]))
            if out.isOpened():
                recording = True
                print(f"开始录像 -> {video_path}")
            else:
                print("无法创建视频文件")
        else:
            recording = False
            out.release()
            out = None
            print(f"录像已保存 -> video/video_{video_cnt:04d}.avi")
            video_cnt += 1

    elif key == 27:  # ESC 退出
        break

# 释放资源
if recording and out is not None:
    out.release()
cap.release()
cv2.destroyAllWindows()