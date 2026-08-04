import cv2
import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import time
from StereoParams import ours_params
import matplotlib.pyplot as plt





'''
求圆心
'''
def solve_projected_center_from_Q_and_normal(Q, n):
    """
    由 Q 和圆平面法向量 n 求真实圆心投影 q_c。

    公式：
        q_c ∼ Q^{-1} n

    输入：
        Q: 3x3 归一化椭圆矩阵
        n: 3维法向量，当前相机坐标系下

    输出：
        q: [x, y, 1]，归一化相机坐标
    """
    Q = np.asarray(Q, dtype=np.float64)
    Q = 0.5 * (Q + Q.T)

    n = np.asarray(n, dtype=np.float64).reshape(3)

    q = np.linalg.solve(Q, n)

    if abs(q[2]) < 1e-12:
        raise ValueError("圆心投影 q[2] 接近 0，无法归一化")

    q = q / q[2]

    return q

def triangulate_center_from_normalized_points(qL, qR, R_LR, t_LR):
    """
    使用左右真实圆心投影 qL, qR 三角化三维圆心。

    输入：
        qL: 左图归一化圆心投影 [xL, yL, 1]
        qR: 右图归一化圆心投影 [xR, yR, 1]
        R_LR, t_LR: 双目外参，满足 X_R = R_LR X_L + t_LR

    输出：
        C_L: 左相机坐标系下的三维圆心 [X, Y, Z]
        C_R: 右相机坐标系下的三维圆心
        valid_depth: 是否满足左右相机正深度
    """
    qL = np.asarray(qL, dtype=np.float64).reshape(3)
    qR = np.asarray(qR, dtype=np.float64).reshape(3)

    R_LR = np.asarray(R_LR, dtype=np.float64).reshape(3, 3)
    t_LR = np.asarray(t_LR, dtype=np.float64).reshape(3, 1)

    # 左相机投影矩阵 P_L = [I | 0]
    P_L = np.hstack([
        np.eye(3, dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64)
    ])

    # 右相机投影矩阵 P_R = [R | t]
    P_R = np.hstack([
        R_LR,
        t_LR
    ])

    xL, yL, _ = qL
    xR, yR, _ = qR

    A = np.zeros((4, 4), dtype=np.float64)

    A[0, :] = xL * P_L[2, :] - P_L[0, :]
    A[1, :] = yL * P_L[2, :] - P_L[1, :]
    A[2, :] = xR * P_R[2, :] - P_R[0, :]
    A[3, :] = yR * P_R[2, :] - P_R[1, :]

    _, _, Vt = np.linalg.svd(A)

    X_h = Vt[-1, :]

    if abs(X_h[3]) < 1e-12:
        raise ValueError("三角化结果 W 接近 0，无法归一化")

    C_L = X_h[:3] / X_h[3]

    C_R = R_LR @ C_L.reshape(3, 1) + t_LR
    C_R = C_R.reshape(3)

    valid_depth = (C_L[2] > 0) and (C_R[2] > 0)

    return C_L, C_R, valid_depth
'''
把归一化圆心投影转成像素坐标，方便画图
'''
def normalized_to_pixel(q, K):
    """
    归一化坐标 [x, y, 1] -> 像素坐标 [u, v]
    """
    q = np.asarray(q, dtype=np.float64).reshape(3)
    p = K @ q
    p = p / p[2]
    return p[:2]












'''
添加绘图函数
'''
def save_metric_plots(frame_ids, score_list, angle_list):
    """
    保存 score 和 angle_deg 曲线图
    """
    if len(frame_ids) == 0:
        print("没有可绘制的数据")
        return

    # score 曲线
    plt.figure(figsize=(10, 4))
    plt.plot(frame_ids, score_list, label="score")
    plt.axhline(0.99, linestyle="--", linewidth=1, label="score = 0.99")
    plt.axhline(0.98, linestyle=":", linewidth=1, label="score = 0.98")
    plt.xlabel("Frame")
    plt.ylabel("score")
    plt.title("Stereo Normal Matching Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("score_curve.png", dpi=200)
    plt.close()

    # angle_deg 曲线
    plt.figure(figsize=(10, 4))
    plt.plot(frame_ids, angle_list, label="angle_deg")
    plt.axhline(3.0, linestyle="--", linewidth=1, label="3 deg")
    plt.axhline(8.0, linestyle=":", linewidth=1, label="8 deg")
    plt.xlabel("Frame")
    plt.ylabel("angle_deg")
    plt.title("Stereo Normal Matching Angle Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("angle_deg_curve.png", dpi=200)
    plt.close()

    print("已保存曲线图: score_curve.png, angle_deg_curve.png")


'''
双目消歧义
'''
def disambiguate_normals_by_stereo(normals_L, normals_R, R_LR):
    """
    利用双目旋转关系，从左右各两个法向量候选中选出一致的一组。

    输入:
        normals_L: [nL1, nL2]，左相机坐标系下的两个法向量候选
        normals_R: [nR1, nR2]，右相机坐标系下的两个法向量候选
        R_LR: 左相机到右相机的旋转矩阵，满足 X_R = R_LR X_L + t_LR

    输出:
        best_nL: 左相机下最终法向量
        best_nR: 右相机下最终法向量，已经和 best_nL 方向对齐
        best_info: 匹配信息
    """
    R_LR = np.asarray(R_LR, dtype=np.float64).reshape(3, 3)

    best_score = -1.0
    best_nL = None
    best_nR = None
    best_info = None

    for i, nL in enumerate(normals_L):
        nL = np.asarray(nL, dtype=np.float64).reshape(3)
        nL = nL / np.linalg.norm(nL)

        # 左相机法向量旋转到右相机坐标系
        nL_to_R = R_LR @ nL
        nL_to_R = nL_to_R / np.linalg.norm(nL_to_R)

        for j, nR in enumerate(normals_R):
            nR = np.asarray(nR, dtype=np.float64).reshape(3)
            nR = nR / np.linalg.norm(nR)

            dot = float(np.dot(nL_to_R, nR))

            # 法向量正负都代表同一平面，所以用 abs
            score = abs(dot)

            if score > best_score:
                best_score = score

                # 如果 dot < 0，说明 nR 和 nL_to_R 方向相反，把 nR 翻转
                if dot < 0:
                    nR_aligned = -nR
                else:
                    nR_aligned = nR

                best_nL = nL
                best_nR = nR_aligned

                best_info = {
                    "idx_L": i,
                    "idx_R": j,
                    "dot_raw": dot,
                    "score": score,
                    "angle_deg": np.rad2deg(np.arccos(np.clip(score, -1.0, 1.0)))
                }

    return best_nL, best_nR, best_info



'''
求法向量候选
'''
def solve_normal_candidates_from_Q(Q):
    """
    由归一化椭圆锥矩阵 Q 求空间圆平面的两个法向量候选。

    输入:
        Q: 3x3 对称矩阵，满足 x.T @ Q @ x = 0

    输出:
        normals: 长度为 2 的 list
                 [n1, n2]
                 每个 n 是 3 维单位向量，位于当前相机坐标系下
    """
    Q = np.asarray(Q, dtype=np.float64)
    Q = 0.5 * (Q + Q.T)

    # 特征值分解，eigh 专门用于实对称矩阵
    eigvals, eigvecs = np.linalg.eigh(Q)

    # Q 是齐次矩阵，整体正负不影响椭圆。
    # 正常圆锥矩阵应整理成两个正特征值、一个负特征值。
    num_pos = np.sum(eigvals > 0)
    num_neg = np.sum(eigvals < 0)

    if num_pos == 1 and num_neg == 2:
        Q = -Q
        eigvals, eigvecs = np.linalg.eigh(Q)
    elif num_pos != 2 or num_neg != 1:
        raise ValueError(f"Q 特征值符号异常: {eigvals}")

    # 重新排序，使 lambda1 >= lambda2 > 0 > lambda3
    pos_idx = np.where(eigvals > 0)[0]
    neg_idx = np.where(eigvals < 0)[0]

    # 正特征值从大到小排序
    pos_idx = pos_idx[np.argsort(eigvals[pos_idx])[::-1]]

    idx1 = pos_idx[0]
    idx2 = pos_idx[1]
    idx3 = neg_idx[0]

    lam1 = eigvals[idx1]
    lam2 = eigvals[idx2]
    lam3 = eigvals[idx3]

    V = np.column_stack([
        eigvecs[:, idx1],
        eigvecs[:, idx2],
        eigvecs[:, idx3]
    ])

    # 保证 V 是右手坐标系，避免后续符号混乱
    if np.linalg.det(V) < 0:
        V[:, 2] *= -1.0

    den = lam1 - lam3
    if abs(den) < 1e-12:
        raise ValueError("特征值退化，无法稳定求法向量")

    alpha2 = (lam1 - lam2) / den
    beta2 = (lam2 - lam3) / den

    # 数值保护，避免浮点误差导致 -1e-16 这种情况
    alpha2 = max(alpha2, 0.0)
    beta2 = max(beta2, 0.0)

    alpha = np.sqrt(alpha2)
    beta = np.sqrt(beta2)

    n_bar_1 = np.array([ alpha, 0.0, beta], dtype=np.float64)
    n_bar_2 = np.array([-alpha, 0.0, beta], dtype=np.float64)

    n1 = V @ n_bar_1
    n2 = V @ n_bar_2

    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)

    return [n1, n2]



def ellipse_to_conic(ellipse):
    """
    OpenCV fitEllipse 椭圆参数 -> 像素坐标下的椭圆矩阵 C_img

    ellipse = ((cx, cy), (width, height), angle_deg)

    返回 C_img，使得：
        [u, v, 1]^T C_img [u, v, 1] = 0
    """
    (cx, cy), (width, height), angle_deg = ellipse

    # fitEllipse 返回的是完整轴长，因此除以 2
    a = width / 2.0
    b = height / 2.0

    if a <= 1e-9 or b <= 1e-9:
        raise ValueError("椭圆半轴太小，无法构造椭圆矩阵")

    theta = np.deg2rad(angle_deg)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    R2 = np.array([
        [cos_t, -sin_t],
        [sin_t,  cos_t]
    ], dtype=np.float64)

    D = np.diag([
        1.0 / (a * a),
        1.0 / (b * b)
    ])

    A = R2 @ D @ R2.T

    c = np.array([[cx], [cy]], dtype=np.float64)

    C_img = np.zeros((3, 3), dtype=np.float64)
    C_img[:2, :2] = A
    C_img[:2, 2:3] = -A @ c
    C_img[2:3, :2] = (-A @ c).T
    C_img[2, 2] = float(c.T @ A @ c - 1.0)

    C_img = 0.5 * (C_img + C_img.T)

    return C_img


def conic_to_Q(C_img, K):
    """
    像素椭圆矩阵 C_img -> 归一化相机坐标下的 Q

    公式：
        Q = K.T @ C_img @ K
    """
    Q = K.T @ C_img @ K

    # 保证对称
    Q = 0.5 * (Q + Q.T)

    # 齐次矩阵尺度归一化
    norm = np.linalg.norm(Q)
    if norm > 1e-12:
        Q = Q / norm

    return Q





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



    params = ours_params

    K_L = params.K_left
    D_L = params.D_left
    K_R = params.K_right
    D_R = params.D_right
    R_LR = params.R
    t_LR = params.T

    print("使用相机参数：")
    print(params)

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


    metric_frame_ids = []
    metric_scores = []
    metric_angles = []  
    center_history = []  
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

      
        C_img_L = ellipse_to_conic(prev_ellipse_L)
        C_img_R = ellipse_to_conic(prev_ellipse_R)

        Q_L = conic_to_Q(C_img_L, K_L)
        Q_R = conic_to_Q(C_img_R, K_R)  

        normal_candidates_L = solve_normal_candidates_from_Q(Q_L)
        normal_candidates_R = solve_normal_candidates_from_Q(Q_R)

        nL1, nL2 = normal_candidates_L
        nR1, nR2 = normal_candidates_R

        best_nL, best_nR, normal_match_info = disambiguate_normals_by_stereo(
            normal_candidates_L,
            normal_candidates_R,
            R_LR
        )
        score = normal_match_info["score"]
        angle = normal_match_info["angle_deg"]

        valid_normal = (score > 0.99) and (angle < 8.0)

        if not valid_normal:
            print(f"帧 {frame_idx}: 法向量不稳定，跳过三角化, score={score:.4f}, angle={angle:.2f}")
            continue

        try:
            qL_center = solve_projected_center_from_Q_and_normal(Q_L, best_nL)
            qR_center = solve_projected_center_from_Q_and_normal(Q_R, best_nR)

            C_L, C_R, valid_depth = triangulate_center_from_normalized_points(
                qL_center,
                qR_center,
                R_LR,
                t_LR
            )
            
            if not valid_depth:
                print(f"帧 {frame_idx}: 三角化圆心不满足正深度, C_L={C_L}, C_R={C_R}")
                continue
            center_history.append([frame_idx, C_L[0], C_L[1], C_L[2], score, angle])        
            pL_center = normalized_to_pixel(qL_center, K_L)
            pR_center = normalized_to_pixel(qR_center, K_R)

        except Exception as e:
            print(f"帧 {frame_idx}: 圆心三角化失败: {e}")
            continue

        metric_frame_ids.append(frame_idx)
        metric_scores.append(normal_match_info["score"])
        metric_angles.append(normal_match_info["angle_deg"])

    

                # 绘图
        display_L = left_half.copy()
        display_R = right_half.copy()

        if pL_center is not None:
            cv2.circle(display_L, (int(pL_center[0]), int(pL_center[1])), 6, (255, 0, 0), -1)
            cv2.putText(display_L, "center", (int(pL_center[0]) + 8, int(pL_center[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        if pR_center is not None:
            cv2.circle(display_R, (int(pR_center[0]), int(pR_center[1])), 6, (255, 0, 0), -1)
            cv2.putText(display_R, "center", (int(pR_center[0]) + 8, int(pR_center[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
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
    save_metric_plots(metric_frame_ids, metric_scores, metric_angles)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    executor.shutdown()
    print(f"输出视频已保存: {output_path}")

if __name__ == "__main__":
    main()