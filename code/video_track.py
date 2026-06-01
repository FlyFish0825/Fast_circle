import cv2
import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import time
# ========== 椭圆点采样与法向量 ==========

def get_ellipse_points_and_normals(center, axes, angle_deg, num_points=40):
    cx, cy = center
    a, b = axes  # 半轴长
    angle_rad = np.deg2rad(angle_deg)
    t = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    x = cx + a * np.cos(t) * np.cos(angle_rad) - b * np.sin(t) * np.sin(angle_rad)
    y = cy + a * np.cos(t) * np.sin(angle_rad) + b * np.sin(t) * np.cos(angle_rad)
    dx = -a * np.sin(t) * np.cos(angle_rad) - b * np.cos(t) * np.sin(angle_rad)
    dy = -a * np.sin(t) * np.sin(angle_rad) + b * np.cos(t) * np.cos(angle_rad)
    norm = np.hypot(dx, dy)
    nx = -dy / norm
    ny = dx / norm
    return np.column_stack((x, y)), np.column_stack((nx, ny))


def detect_ring_centerline(img_gray, pt, normal, search_length, template):
    h, w = img_gray.shape
    x0, y0 = pt
    nx, ny = normal

    distances = np.arange(-search_length, search_length + 1)
    sample_x = x0 + distances * nx
    sample_y = y0 + distances * ny

    valid = (sample_x >= 0) & (sample_x < w-1) & (sample_y >= 0) & (sample_y < h-1)
    if not np.any(valid):
        return None

    vx, vy = sample_x[valid], sample_y[valid]
    vals = img_gray[vy.astype(int), vx.astype(int)].astype(float)

    if len(vals) < len(template) + 4:
        return None

    response = np.convolve(vals, template, mode='valid')

    pos_idx = np.argmax(response)
    neg_idx = np.argmin(response)

    pos_val = response[pos_idx]
    neg_val = response[neg_idx]

    edge_thresh = 15
    if pos_val < edge_thresh or neg_val > -edge_thresh:
        return None

    # 返回外边缘点（pos_idx 对应黑→白，即从内到外的法线方向上的外边缘）
    # 这里仍然返回外边缘点，保持与左相机一致
    best_x = vx[pos_idx]
    best_y = vy[pos_idx]

    return (best_x, best_y)


# ========== 第一帧手动标定 ==========

def mark_ellipse_on_frame(frame, window_name="Mark Ellipse"):
    points = []
    img = frame.copy()

    def mouse_cb(event, x, y, flags, param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            temp = frame.copy()
            for p in points:
                cv2.circle(temp, p, 4, (0, 255, 0), -1)
            if len(points) >= 5:
                pts_arr = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
                cv2.ellipse(temp, cv2.fitEllipse(pts_arr), (0, 255, 0), 2)
            cv2.imshow(window_name, temp)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 600)
    cv2.setMouseCallback(window_name, mouse_cb)

    while True:
        cv2.imshow(window_name, img)
        key = cv2.waitKey(50) & 0xFF
        if key == 13 and len(points) >= 5:
            pts_arr = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
            ell = cv2.fitEllipse(pts_arr)
            cv2.destroyWindow(window_name)
            return ell
        elif key == ord('c'):
            points.clear()
            img = frame.copy()
        elif key == 27 or key == ord('q'):
            cv2.destroyWindow(window_name)
            return None
    cv2.destroyWindow(window_name)
    return None


def process_side(gray, prev_ellipse, num_samples, search_length, template):
    """
    处理单个视图：返回 (ellipse, detected_pts)
    逻辑与原来完全一致，只是封装为函数便于线程调用
    """
    (cx, cy), (a, b), angle = prev_ellipse
    # 预测椭圆直接使用上一帧椭圆（这里可加入运动模型，目前保持原样）
    pred_cx, pred_cy = cx, cy
    pts, normals = get_ellipse_points_and_normals(
        (pred_cx, pred_cy), (a/2, b/2), angle, num_samples
    )
    detected_pts = []
    for pt, n in zip(pts, normals):
        best_pt = detect_ring_centerline(gray, pt, n, search_length, template)
        if best_pt is not None:
            detected_pts.append(best_pt)

    ellipse = prev_ellipse
    if len(detected_pts) >= 5:
        pts_arr = np.array(detected_pts, dtype=np.float32).reshape(-1, 1, 2)
        ellipse = cv2.fitEllipse(pts_arr)
    return ellipse, detected_pts


def main():
    video_path = 'video/video_0000.avi'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频 FPS: {fps}, 总帧数: {total_frames}")

    ret, frame = cap.read()
    if not ret:
        return

    h, w = frame.shape[:2]
    w_left = w // 2
    left_frame = frame[:, :w_left]
    right_frame = frame[:, w_left:]

    # 左右初始标定
    print("\n请先在左图的椭圆边缘点击至少 5 个点，然后按 Enter 拟合。")
    init_ellipse_L = mark_ellipse_on_frame(left_frame, "Mark Left Ellipse")
    if init_ellipse_L is None:
        cap.release()
        return
    print("\n请在右图的椭圆边缘点击至少 5 个点，然后按 Enter 拟合。")
    init_ellipse_R = mark_ellipse_on_frame(right_frame, "Mark Right Ellipse")
    if init_ellipse_R is None:
        cap.release()
        return

    # 参数
    NUM_SAMPLES = 50
    SEARCH_LENGTH = 50
    TEMPLATE = np.array([-1, -2, -4, -8, -16, 0, 16, 8, 4, 2, 1])

    prev_ellipse_L = init_ellipse_L
    prev_ellipse_R = init_ellipse_R

    # 输出视频
    output_path = 'output_tracking_fast.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # 线程池（2个线程，处理左右视图）
    executor = ThreadPoolExecutor(max_workers=2)
    t = time.perf_counter() # um
    frame_idx = 1
    while True:
        
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        left_half = frame[:, :w_left]
        right_half = frame[:, w_left:]
        gray_L = cv2.cvtColor(left_half, cv2.COLOR_BGR2GRAY)
        gray_R = cv2.cvtColor(right_half, cv2.COLOR_BGR2GRAY)

        # # 并行提交左右处理任务
        # future_L = executor.submit(process_side, gray_L, prev_ellipse_L,
        #                            NUM_SAMPLES, SEARCH_LENGTH, TEMPLATE)
        # future_R = executor.submit(process_side, gray_R, prev_ellipse_R,
        #                            NUM_SAMPLES, SEARCH_LENGTH, TEMPLATE)

    
        # 获取结果（等待两个线程都完成）
        prev_ellipse_L, detected_pts_L = process_side( gray_L, prev_ellipse_L,
                                   NUM_SAMPLES, SEARCH_LENGTH, TEMPLATE)
        prev_ellipse_R, detected_pts_R = process_side( gray_R, prev_ellipse_R,
                                   NUM_SAMPLES, SEARCH_LENGTH, TEMPLATE)


        # 绘图
        display_L = left_half.copy()
        for (x, y) in detected_pts_L:
            cv2.circle(display_L, (int(x), int(y)), 2, (0, 0, 255), -1)
        cv2.ellipse(display_L, prev_ellipse_L, (0, 255, 0), 2)

        display_R = right_half.copy()
        for (x, y) in detected_pts_R:
            cv2.circle(display_R, (int(x), int(y)), 2, (0, 0, 255), -1)
        cv2.ellipse(display_R, prev_ellipse_R, (0, 255, 0), 2)

        combined = np.hstack([display_L, display_R])
        cv2.line(combined, (w_left, 0), (w_left, h), (100, 100, 100), 1)
        out.write(combined)

        cv2.imshow("Stereo Tracking (Threaded)", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        if frame_idx % 30 == 0:
            fps = 30 / (time.perf_counter() - t)
            t = time.perf_counter() # um
            
            print(f"帧 {frame_idx}/{total_frames}, 左点: {len(detected_pts_L)}, 右点: {len(detected_pts_R)}, 估计 FPS: {fps:.2f}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    executor.shutdown()
    print(f"输出视频已保存: {output_path}")

if __name__ == "__main__":
    main()