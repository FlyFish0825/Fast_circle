import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

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

# ========== 第一帧手动标定 ==========

def mark_ellipse_on_frame(frame):
    window_name = "Mark Ellipse — Click points, Enter to fit, C to clear, Esc to quit"
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

# ========== 核心：黑色圆环中线检测 ==========

def detect_ring_centerline(img_gray, pt, normal, search_length, template):
    """
    沿法线方向检测黑色圆环的两条边缘（白→黑、黑→白），
    返回两条边缘的中线点（即环的中心线）。

    返回 (best_x, best_y), score 或 (None, 0)
    score 取两条边缘响应强度的平均值。
    """
    h, w = img_gray.shape
    x0, y0 = pt
    nx, ny = normal

    distances = np.arange(-search_length, search_length + 1)
    sample_x = x0 + distances * nx
    sample_y = y0 + distances * ny

    valid = (sample_x >= 0) & (sample_x < w-1) & (sample_y >= 0) & (sample_y < h-1)
    if not np.any(valid):
        return None, 0

    vx, vy = sample_x[valid], sample_y[valid]
    vals = img_gray[vy.astype(int), vx.astype(int)].astype(float)

    if len(vals) < len(template) + 4:
        return None, 0

    # 一维卷积响应
    response = np.convolve(vals, template, mode='valid')

    # 找最强正峰（黑→白，外边缘）和最强负峰（白→黑，内边缘）
    pos_idx = np.argmax(response)   # 最强正响应 = 外边缘
    neg_idx = np.argmin(response)   # 最强负响应 = 内边缘

    pos_val = response[pos_idx]
    neg_val = response[neg_idx]

    # 两条边缘都需足够强
    edge_thresh = 15
    if pos_val < edge_thresh or neg_val > -edge_thresh:
        return None, 0

    # 两条边缘的中点 → 环的中心线
    inner_idx = min(pos_idx, neg_idx)
    outer_idx = max(pos_idx, neg_idx)
    mid_offset = (inner_idx + outer_idx) // 2 + len(template) // 2

    best_x = vx[mid_offset]
    best_y = vy[mid_offset]
    score = (pos_val - neg_val) / 2  # 正负响应平均强度

    return (best_x, best_y), score


# ========== 点位到椭圆边界的归一化距离（离群剔除用） ==========

def normalized_radius(pt, ellipse):
    (cx, cy), (a_full, b_full), angle = ellipse
    angle_rad = np.deg2rad(angle)
    dx = pt[0] - cx
    dy = pt[1] - cy
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    dx_r = dx * cos_a - dy * sin_a
    dy_r = dx * sin_a + dy * cos_a
    return np.sqrt((dx_r / (a_full/2))**2 + (dy_r / (b_full/2))**2)


# ========== 椭圆形状约束 ==========

def constrain_ellipse(new_ellipse, prev_ellipse, max_center_drift=0.3, max_size_change=0.2):
    (cx, cy), (a, b), angle = new_ellipse
    (pcx, pcy), (pa, pb), pangle = prev_ellipse

    # 对齐 a/b 表示
    diff_current = abs(a - pa) + abs(b - pb)
    diff_swapped = abs(b - pa) + abs(a - pb)
    if diff_swapped < diff_current:
        a, b = b, a
        angle = (angle + 90) % 180

    # 中心约束
    max_drift = max(pa, pb) * max_center_drift
    dx = cx - pcx
    dy = cy - pcy
    drift = np.hypot(dx, dy)
    if drift > max_drift > 0:
        scale = max_drift / drift
        cx = pcx + dx * scale
        cy = pcy + dy * scale

    # 尺寸约束
    a = np.clip(a, pa * (1 - max_size_change), pa * (1 + max_size_change))
    b = np.clip(b, pb * (1 - max_size_change), pb * (1 + max_size_change))

    # 角度约束
    da = angle - pangle
    if da > 90:
        angle = pangle + 90
    elif da < -90:
        angle = pangle - 90
    else:
        angle = pangle + np.clip(da, -15, 15)

    return ((cx, cy), (a, b), np.round(angle, 1))


# ========== 主程序 ==========

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
    # ---- 第一帧标定 ----
    print("\n请在第一帧的椭圆边缘点击至少 5 个点，然后按 Enter 拟合。")
    initial_ellipse = mark_ellipse_on_frame(left_frame)
    if initial_ellipse is None:
        cap.release()
        return

    (init_cx, init_cy), (init_a, init_b), init_angle = initial_ellipse
    print(f"初始椭圆: 中心=({init_cx:.1f}, {init_cy:.1f}), "
          f"轴=({init_a:.1f}, {init_b:.1f}), 角度={init_angle:.1f}")

    # ---- 算法参数 ----
    TEMPLATE = np.array([-1, -2, -4, -2, -1, 0, 1, 2, 4, 2, 1])
    RING_MIN_SCORE = 15        # 环检测最小响应
    OUTLIER_R_SIGMA = 2.5      # 离群阈值（归一化半径偏离倍数）
    VEL_EMA_ALPHA = 0.6        # 速度平滑系数

    # ---- 跟踪状态 ----
    prev_ellipse = initial_ellipse
    smooth_vx, smooth_vy = 0.0, 0.0

    speeds = []
    speed_buffer = deque(maxlen=5)
    frame_idx = 1
    lost_count = 0

    # ---- 逐帧处理 ----
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        left_half = frame[:, :w_left].copy()
        gray = cv2.cvtColor(left_half, cv2.COLOR_BGR2GRAY)
        (p_cx, p_cy), (p_a, p_b), p_angle = prev_ellipse
        ellipse_size = np.sqrt(p_a * p_b)

        # 1. 运动预测
        pred_cx = p_cx + smooth_vx
        pred_cy = p_cy + smooth_vy
        pred_ellipse = ((pred_cx, pred_cy), (p_a, p_b), p_angle)

        # 2. 自适应搜索参数（环大小变化时自动缩放）
        search_length = max(20, int(ellipse_size * 0.12))
        if lost_count >= 3:
            search_length = int(search_length * (1.0 + 0.3 * lost_count))
        num_samples = max(40, min(100, int(ellipse_size * 0.10)))

        # 3. 在预测椭圆上采样并检测环中线
        pts, normals = get_ellipse_points_and_normals(
            (pred_cx, pred_cy), (p_a / 2, p_b / 2), p_angle, num_samples
        )

        detected_pts = []
        scores = []
        for pt, n in zip(pts, normals):
            best_pt, score = detect_ring_centerline(gray, pt, n, search_length, TEMPLATE)
            if best_pt is not None and score > RING_MIN_SCORE:
                detected_pts.append(best_pt)
                scores.append(score)

        # 4. 离群剔除
        inliers = []
        if len(detected_pts) >= 5:
            pts_arr = np.array(detected_pts)
            radii = np.array([normalized_radius(pt, pred_ellipse) for pt in pts_arr])
            r_med = np.median(radii)
            r_mad = np.median(np.abs(radii - r_med))
            if r_mad < 0.05:
                r_mad = 0.15
            inlier_mask = np.abs(radii - 1.0) < OUTLIER_R_SIGMA * r_mad
            inliers = [detected_pts[i] for i in range(len(detected_pts)) if inlier_mask[i]]

        # 5. 椭圆拟合 + 约束
        current_ellipse = None
        fit_ok = False

        if len(inliers) >= 5:
            inliers_arr = np.array(inliers, dtype=np.float32)
            try:
                raw_ellipse = cv2.fitEllipse(inliers_arr.reshape(-1, 1, 2))
                current_ellipse = constrain_ellipse(raw_ellipse, prev_ellipse)

                # 质量检查：约束后的中心偏移不能太大
                (ncx, ncy), _, _ = current_ellipse
                drift = np.hypot(ncx - pred_cx, ncy - pred_cy)
                if drift < search_length * 2.0:
                    fit_ok = True
                    lost_count = max(0, lost_count - 1)
                else:
                    current_ellipse = None
            except cv2.error:
                pass

        if current_ellipse is None:
            current_ellipse = pred_ellipse
            smooth_vx *= 0.85
            smooth_vy *= 0.85
            disp = 0.0
            lost_count += 1
        else:
            # 更新速度和状态
            new_cx, new_cy = current_ellipse[0]
            raw_vx = new_cx - p_cx
            raw_vy = new_cy - p_cy
            smooth_vx = VEL_EMA_ALPHA * raw_vx + (1 - VEL_EMA_ALPHA) * smooth_vx
            smooth_vy = VEL_EMA_ALPHA * raw_vy + (1 - VEL_EMA_ALPHA) * smooth_vy
            max_v = max(p_a, p_b) * 0.5
            smooth_vx = np.clip(smooth_vx, -max_v, max_v)
            smooth_vy = np.clip(smooth_vy, -max_v, max_v)
            prev_ellipse = current_ellipse
            disp = np.hypot(raw_vx, raw_vy)

        speeds.append(disp)
        speed_buffer.append(disp)

        # ---- 可视化 ----
        display = frame.copy()
        avg_speed = np.mean(speed_buffer) if speed_buffer else 0

        if current_ellipse is not None:
            cv2.ellipse(display, current_ellipse, (0, 255, 0), 2)

        # 内点（青色）vs 外点（红色）
        for pt in inliers:
            cv2.circle(display, (int(pt[0]), int(pt[1])), 3, (255, 255, 0), -1)
        for pt in detected_pts:
            if pt not in inliers:
                cv2.circle(display, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)

        cv2.line(display, (w_left, 0), (w_left, h), (100, 100, 100), 1)

        status = "TRACK" if fit_ok else "PREDICT"
        info = [
            f"Frame: {frame_idx}/{total_frames}  [{status}]",
            f"Speed: {avg_speed:.1f} px/fr",
            f"Inliers: {len(inliers)}/{len(detected_pts)}",
            f"Center: ({current_ellipse[0][0]:.0f}, {current_ellipse[0][1]:.0f})",
        ]
        color = (0, 255, 0) if fit_ok else (0, 200, 255)
        for i, text in enumerate(info):
            cv2.putText(display, text, (10, 30 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(display, "Right (unused)", (w_left + 20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)

        cv2.imshow('Ellipse Tracking', display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # ---- 速度曲线 ----
    if speeds:
        plt.figure(figsize=(14, 5))
        plt.plot(speeds, 'b-', alpha=0.5, label='Instant')
        if len(speeds) > 15:
            smooth = np.convolve(speeds, np.ones(15)/15, mode='valid')
            plt.plot(np.arange(14, 14 + len(smooth)), smooth, 'r-', linewidth=2, label='Smoothed')
        plt.xlabel('Frame')
        plt.ylabel('Speed (px/frame)')
        plt.title('Ellipse Center Speed')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    print(f"\n完成。共处理 {frame_idx} 帧。")


if __name__ == "__main__":
    main()
