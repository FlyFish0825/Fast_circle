import cv2
import numpy as np
import os
from collections import deque

# ========== 复用已有的工具函数 ==========

def get_ellipse_points_and_normals(center, axes, angle_deg, num_points=40):
    cx, cy = center
    a, b = axes
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

def search_along_normal(img_gray, pt, normal, search_length, template):
    h, w = img_gray.shape
    x0, y0 = pt
    nx, ny = normal
    distances = np.arange(-search_length, search_length + 1)
    sample_x = x0 + distances * nx
    sample_y = y0 + distances * ny
    valid_mask = (sample_x >= 0) & (sample_x < w-1) & (sample_y >= 0) & (sample_y < h-1)
    if not np.any(valid_mask):
        return None, 0
    v_x, v_y = sample_x[valid_mask], sample_y[valid_mask]
    pixel_values = img_gray[v_y.astype(int), v_x.astype(int)].astype(float)
    if len(pixel_values) < len(template):
        return None, 0
    response = np.convolve(pixel_values, template, mode='valid')
    max_idx = np.argmax(np.abs(response))
    max_response = np.abs(response[max_idx])
    offset_idx = max_idx + len(template) // 2
    best_x = v_x[offset_idx]
    best_y = v_y[offset_idx]
    return (best_x, best_y), max_response

def normalized_radius(pt, ellipse):
    (cx, cy), (a_full, b_full), angle = ellipse
    angle_rad = np.deg2rad(angle)
    dx = pt[0] - cx
    dy = pt[1] - cy
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    dx_r = dx * cos_a - dy * sin_a
    dy_r = dx * sin_a + dy * cos_a
    r = np.sqrt((dx_r / (a_full/2))**2 + (dy_r / (b_full/2))**2)
    return r

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
                ell = cv2.fitEllipse(pts_arr)
                (cx, cy), (a, b), angle = ell
                cv2.ellipse(temp, ell, (0, 255, 0), 2)
                overlay = temp.copy()
                band = int(np.sqrt(a * b) * 0.15)
                cv2.ellipse(overlay, (int(cx), int(cy)),
                            (int(a/2 + band), int(b/2 + band)), angle, 0, 360, (255, 200, 100), -1)
                cv2.ellipse(overlay, (int(cx), int(cy)),
                            (max(0, int(a/2 - band)), max(0, int(b/2 - band))), angle, 0, 360, (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.3, temp, 0.7, 0, temp)
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

# ========== 标注绘图 ==========

def draw_debug_image(frame, prev_ellipse, pred_ellipse, sampling_pts, normals,
                     search_length, detected_pts, inliers, outliers, frame_idx, output_dir):
    """在一张图上叠加所有调试信息"""
    vis = frame.copy()
    h, w = vis.shape[:2]

    # 1. 上一帧椭圆（绿色实线）
    if prev_ellipse is not None:
        cv2.ellipse(vis, prev_ellipse, (0, 255, 0), 2)

    # 2. 搜索带（半透明灰色环）
    if pred_ellipse is not None:
        (cx, cy), (a, b), angle = pred_ellipse
        overlay = vis.copy()
        outer_a = int(a/2 + search_length)
        outer_b = int(b/2 + search_length)
        inner_a = max(0, int(a/2 - search_length))
        inner_b = max(0, int(b/2 - search_length))
        cv2.ellipse(overlay, (int(cx), int(cy)), (outer_a, outer_b), angle, 0, 360, (200, 200, 100), -1)
        cv2.ellipse(overlay, (int(cx), int(cy)), (inner_a, inner_b), angle, 0, 360, (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.2, vis, 0.8, 0, vis)
        # 预测椭圆轮廓（黄色虚线）
        cv2.ellipse(vis, pred_ellipse, (0, 255, 255), 1, lineType=cv2.LINE_AA)

    # 3. 法向量采样线（灰色短线）
    if sampling_pts is not None and normals is not None:
        n_to_draw = min(30, len(sampling_pts))
        step = max(1, len(sampling_pts) // n_to_draw)
        for i, (pt, n) in enumerate(zip(sampling_pts[::step], normals[::step])):
            p1 = (int(pt[0] - n[0] * search_length), int(pt[1] - n[1] * search_length))
            p2 = (int(pt[0] + n[0] * search_length), int(pt[1] + n[1] * search_length))
            cv2.line(vis, p1, p2, (180, 180, 180), 1, lineType=cv2.LINE_AA)

    # 4. 内点（青色大点）和外点（红色小点）
    for pt in inliers:
        cv2.circle(vis, (int(pt[0]), int(pt[1])), 4, (255, 255, 0), -1)
    for pt in outliers:
        cv2.circle(vis, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)

    # 5. 帧号
    cv2.putText(vis, f"Frame {frame_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 保存
    out_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
    cv2.imwrite(out_path, vis)
    return out_path

# ========== 主程序 ==========

def main():
    output_dir = "debug_frames"
    os.makedirs(output_dir, exist_ok=True)

    video_path = 'video/video_0000.avi'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"总帧数: {total_frames}")

    # 第一帧
    ret, frame = cap.read()
    if not ret:
        return
    h, w = frame.shape[:2]
    w_left = w // 2
    left_frame = frame[:, :w_left]

    print("请在第一帧标定椭圆...")
    initial_ellipse = mark_ellipse_on_frame(left_frame)
    if initial_ellipse is None:
        print("未标定")
        cap.release()
        return
    print(f"初始椭圆: center={initial_ellipse[0]}, axes={initial_ellipse[1]}, angle={initial_ellipse[2]:.1f}")

    # 参数
    TEMPLATE = np.array([-1, -2, -4, -2, -1, 0, 1, 2, 4, 2, 1])
    RESPONSE_THRESH = 10
    MIN_SEARCH = 25
    OUTLIER_R_SIGMA = 2.5
    VEL_EMA_ALPHA = 0.6

    prev_ellipse = initial_ellipse
    smooth_vx, smooth_vy = 0.0, 0.0
    lost_count = 0
    frame_idx = 1

    MAX_SAVE = 150  # 最多保存前 150 帧

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        left_half = frame[:, :w_left].copy()
        gray = cv2.cvtColor(left_half, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        (p_cx, p_cy), (p_a, p_b), p_angle = prev_ellipse
        ellipse_size = np.sqrt(p_a * p_b)

        # 预测
        pred_cx = p_cx + smooth_vx
        pred_cy = p_cy + smooth_vy
        pred_ellipse = ((pred_cx, pred_cy), (p_a, p_b), p_angle)

        # 自适应搜索
        search_length = max(MIN_SEARCH, int(ellipse_size * 0.15))
        num_samples = max(40, min(100, int(ellipse_size * 0.12)))

        if lost_count >= 3:
            search_length = int(search_length * (1.0 + 0.5 * lost_count))

        # 采样与搜索
        sampling_pts, normals = get_ellipse_points_and_normals(
            (pred_cx, pred_cy), (p_a / 2, p_b / 2), p_angle, num_samples
        )

        detected_pts = []
        scores = []
        for pt, n in zip(sampling_pts, normals):
            best_pt, score = search_along_normal(gray, pt, n, search_length, TEMPLATE)
            if best_pt is not None and score > RESPONSE_THRESH:
                detected_pts.append(best_pt)
                scores.append(score)

        # 离群剔除
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
        outliers = [p for p in detected_pts if p not in inliers]

        # 拟合
        fit_ok = False
        if len(inliers) >= 5:
            inliers_arr = np.array(inliers, dtype=np.float32)
            try:
                current_ellipse = cv2.fitEllipse(inliers_arr.reshape(-1, 1, 2))
                new_cx, new_cy = current_ellipse[0]
                drift = np.hypot(new_cx - pred_cx, new_cy - pred_cy)
                if drift < search_length * 2.5:
                    fit_ok = True
                    lost_count = 0
                    raw_vx = new_cx - p_cx
                    raw_vy = new_cy - p_cy
                    smooth_vx = VEL_EMA_ALPHA * raw_vx + (1 - VEL_EMA_ALPHA) * smooth_vx
                    smooth_vy = VEL_EMA_ALPHA * raw_vy + (1 - VEL_EMA_ALPHA) * smooth_vy
                    max_v = search_length
                    smooth_vx = np.clip(smooth_vx, -max_v, max_v)
                    smooth_vy = np.clip(smooth_vy, -max_v, max_v)
                    prev_ellipse = current_ellipse
                else:
                    lost_count += 1
            except cv2.error:
                lost_count += 1
        else:
            lost_count += 1

        if not fit_ok:
            current_ellipse = pred_ellipse
            smooth_vx *= 0.85
            smooth_vy *= 0.85

        # 保存调试图
        if frame_idx <= MAX_SAVE:
            # 在完整帧的左半边绘图
            full_vis = frame.copy()
            h_f, w_f = full_vis.shape[:2]
            wl = w_f // 2

            # 在左半部分绘图
            left_vis = full_vis[:, :wl].copy()
            if prev_ellipse is not None:
                cv2.ellipse(left_vis, prev_ellipse, (0, 255, 0), 2)

            # 搜索带
            (cx, cy), (a, b), angle = pred_ellipse
            overlay = left_vis.copy()
            outer_a = int(a/2 + search_length)
            outer_b = int(b/2 + search_length)
            inner_a = max(0, int(a/2 - search_length))
            inner_b = max(0, int(b/2 - search_length))
            cv2.ellipse(overlay, (int(cx), int(cy)), (outer_a, outer_b), angle, 0, 360, (200, 200, 100), -1)
            cv2.ellipse(overlay, (int(cx), int(cy)), (inner_a, inner_b), angle, 0, 360, (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.2, left_vis, 0.8, 0, left_vis)
            cv2.ellipse(left_vis, pred_ellipse, (0, 255, 255), 1)

            # 法向量线
            n_draw = min(20, len(sampling_pts))
            step = max(1, len(sampling_pts) // n_draw)
            for i, (pt, n) in enumerate(zip(sampling_pts[::step], normals[::step])):
                p1 = (int(pt[0] - n[0] * search_length), int(pt[1] - n[1] * search_length))
                p2 = (int(pt[0] + n[0] * search_length), int(pt[1] + n[1] * search_length))
                cv2.line(left_vis, p1, p2, (180, 180, 180), 1)

            # 内点与外点
            for pt in inliers:
                cv2.circle(left_vis, (int(pt[0]), int(pt[1])), 4, (255, 255, 0), -1)
            for pt in outliers:
                cv2.circle(left_vis, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)

            full_vis[:, :wl] = left_vis
            cv2.line(full_vis, (wl, 0), (wl, h_f), (100, 100, 100), 1)

            # 状态文字
            status = "TRACK" if fit_ok else "PREDICT"
            info = [
                f"Frame {frame_idx}  [{status}]",
                f"Inliers: {len(inliers)}/{len(detected_pts)}",
                f"Search: {search_length}px  Vel: ({smooth_vx:.1f},{smooth_vy:.1f})",
                f"Center: ({prev_ellipse[0][0]:.0f}, {prev_ellipse[0][1]:.0f})",
            ]
            color = (0, 255, 0) if fit_ok else (0, 200, 255)
            for i, t in enumerate(info):
                cv2.putText(full_vis, t, (10, 30 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            out_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
            cv2.imwrite(out_path, full_vis)

    cap.release()
    print(f"\n调试图已保存到 {output_dir}/")
    print(f"共处理 {frame_idx} 帧，最终 lost_count={lost_count}")

if __name__ == "__main__":
    main()
