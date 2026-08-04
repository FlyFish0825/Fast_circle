import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
import importlib.util
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# =========================
# 路径与参数加载
# =========================
BASE_DIR = Path(__file__).resolve().parent


def load_stereo_params():
    """
    自动加载同目录下的 StereoParams.py 或 StereoParams(1).py
    并返回其中的 ours_params
    """
    candidates = [
        BASE_DIR / "StereoParams.py",
        BASE_DIR / "StereoParams(1).py",
    ]

    param_file = None
    for p in candidates:
        if p.exists():
            param_file = p
            break

    if param_file is None:
        raise FileNotFoundError(
            "未找到 StereoParams.py 或 StereoParams(1).py，请确认文件与本脚本在同一目录。"
        )

    spec = importlib.util.spec_from_file_location("stereo_params_module", str(param_file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "ours_params"):
        raise AttributeError(f"{param_file.name} 中未找到 ours_params")

    print(f"已加载参数文件: {param_file.name}")
    return module.ours_params


# =========================
# 解析法相关函数
# =========================
def ellipse_to_conic(ellipse):
    """
    OpenCV fitEllipse 椭圆参数 -> 像素坐标下的椭圆矩阵 C_img

    ellipse = ((cx, cy), (width, height), angle_deg)

    返回:
        C_img，使得 [u, v, 1]^T C_img [u, v, 1] = 0
    """
    (cx, cy), (width, height), angle_deg = ellipse

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

    Q = K.T @ C_img @ K
    """
    Q = K.T @ C_img @ K
    Q = 0.5 * (Q + Q.T)

    norm = np.linalg.norm(Q)
    if norm > 1e-12:
        Q = Q / norm

    return Q


def solve_normal_candidates_from_Q(Q):
    """
    由归一化椭圆锥矩阵 Q 求空间圆平面的两个法向量候选
    """
    Q = np.asarray(Q, dtype=np.float64)
    Q = 0.5 * (Q + Q.T)

    eigvals, eigvecs = np.linalg.eigh(Q)

    num_pos = np.sum(eigvals > 0)
    num_neg = np.sum(eigvals < 0)

    if num_pos == 1 and num_neg == 2:
        Q = -Q
        eigvals, eigvecs = np.linalg.eigh(Q)
    elif num_pos != 2 or num_neg != 1:
        raise ValueError(f"Q 特征值符号异常: {eigvals}")

    pos_idx = np.where(eigvals > 0)[0]
    neg_idx = np.where(eigvals < 0)[0]

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

    if np.linalg.det(V) < 0:
        V[:, 2] *= -1.0

    den = lam1 - lam3
    if abs(den) < 1e-12:
        raise ValueError("特征值退化，无法稳定求法向量")

    alpha2 = (lam1 - lam2) / den
    beta2 = (lam2 - lam3) / den

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


def disambiguate_normals_by_stereo(normals_L, normals_R, R_LR):
    """
    利用双目旋转关系，从左右各两个法向量候选中选出一致的一组
    """
    R_LR = np.asarray(R_LR, dtype=np.float64).reshape(3, 3)

    best_score = -1.0
    best_nL = None
    best_nR = None
    best_info = None

    for i, nL in enumerate(normals_L):
        nL = np.asarray(nL, dtype=np.float64).reshape(3)
        nL = nL / np.linalg.norm(nL)

        nL_to_R = R_LR @ nL
        nL_to_R = nL_to_R / np.linalg.norm(nL_to_R)

        for j, nR in enumerate(normals_R):
            nR = np.asarray(nR, dtype=np.float64).reshape(3)
            nR = nR / np.linalg.norm(nR)

            dot = float(np.dot(nL_to_R, nR))
            score = abs(dot)

            if score > best_score:
                best_score = score

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


def solve_projected_center_from_Q_and_normal(Q, n):
    """
    由 Q 和圆平面法向量 n 求真实圆心投影 q_c
    q_c ∼ Q^{-1} n
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
    使用左右真实圆心投影 qL, qR 三角化三维圆心
    """
    qL = np.asarray(qL, dtype=np.float64).reshape(3)
    qR = np.asarray(qR, dtype=np.float64).reshape(3)

    R_LR = np.asarray(R_LR, dtype=np.float64).reshape(3, 3)
    t_LR = np.asarray(t_LR, dtype=np.float64).reshape(3, 1)

    P_L = np.hstack([
        np.eye(3, dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64)
    ])

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
    C_R = (R_LR @ C_L.reshape(3, 1) + t_LR).reshape(3)

    valid_depth = (C_L[2] > 0) and (C_R[2] > 0)
    return C_L, C_R, valid_depth


def normalized_to_pixel(q, K):
    """
    归一化坐标 [x, y, 1] -> 像素坐标 [u, v]
    """
    q = np.asarray(q, dtype=np.float64).reshape(3)
    p = K @ q
    p = p / p[2]
    return p[:2]


# =========================
# 绘图与保存函数
# =========================
def save_metric_plots(frame_ids, score_list, angle_list):
    """
    保存 score 和 angle_deg 曲线图
    """
    if len(frame_ids) == 0:
        print("没有可绘制的 score/angle 数据")
        return

    score_path = BASE_DIR / "score_curve.png"
    angle_path = BASE_DIR / "angle_deg_curve.png"

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
    plt.savefig(score_path, dpi=200)
    plt.close()

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
    plt.savefig(angle_path, dpi=200)
    plt.close()

    print(f"已保存曲线图: {score_path.name}, {angle_path.name}")


def save_relative_pose_plots(pose_history, video_fps):
    """
    以时间为横轴绘制相对位置/朝向曲线
    """
    if len(pose_history) == 0:
        print("没有有效 pose_history，无法绘制相对位姿曲线")
        return

    arr = np.array(pose_history, dtype=np.float64)

    frame_ids = arr[:, 0]
    t = frame_ids / video_fps

    C = arr[:, 1:4]
    n = arr[:, 4:7]
    score = arr[:, 7]
    angle_deg = arr[:, 8]

    d = np.linalg.norm(C, axis=1)
    h = -np.sum(n * C, axis=1)
    rho2 = np.clip(d ** 2 - h ** 2, 0.0, None)
    rho = np.sqrt(rho2)

    theta = np.rad2deg(np.arccos(np.clip(np.abs(n[:, 2]), 0.0, 1.0)))

    out1 = BASE_DIR / "pose_distance_to_center.png"
    out2 = BASE_DIR / "pose_normal_distance_h.png"
    out3 = BASE_DIR / "pose_inplane_offset_rho.png"
    out4 = BASE_DIR / "pose_tilt_angle_theta.png"
    out5 = BASE_DIR / "pose_match_score.png"
    out6 = BASE_DIR / "pose_match_angle.png"

    plt.figure(figsize=(10, 4))
    plt.plot(t, d, label="distance_to_center")
    plt.xlabel("Time (s)")
    plt.ylabel("Distance (mm)")
    plt.title("Camera-Circle Center Distance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out1, dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(t, h, label="normal_distance_h")
    plt.xlabel("Time (s)")
    plt.ylabel("h (mm)")
    plt.title("Camera Height Relative to Circle Plane")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out2, dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(t, rho, label="in_plane_offset_rho")
    plt.xlabel("Time (s)")
    plt.ylabel("rho (mm)")
    plt.title("Camera In-plane Offset Relative to Circle Center")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out3, dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(t, theta, label="tilt_angle_theta")
    plt.xlabel("Time (s)")
    plt.ylabel("theta (deg)")
    plt.title("Angle Between Optical Axis and Circle Normal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out4, dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(t, score, label="match_score")
    plt.axhline(0.99, linestyle="--", linewidth=1, label="0.99")
    plt.axhline(0.98, linestyle=":", linewidth=1, label="0.98")
    plt.xlabel("Time (s)")
    plt.ylabel("score")
    plt.title("Stereo Normal Matching Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out5, dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(t, angle_deg, label="normal_match_angle")
    plt.axhline(3.0, linestyle="--", linewidth=1, label="3 deg")
    plt.axhline(8.0, linestyle=":", linewidth=1, label="8 deg")
    plt.xlabel("Time (s)")
    plt.ylabel("angle (deg)")
    plt.title("Stereo Normal Matching Angle")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out6, dpi=200)
    plt.close()

    print("已保存相对位姿曲线：")
    print(f"  {out1.name}")
    print(f"  {out2.name}")
    print(f"  {out3.name}")
    print(f"  {out4.name}")
    print(f"  {out5.name}")
    print(f"  {out6.name}")


def save_pose_strip_3d(pose_history, video_fps, step=10):
    """
    3D 条带图：
    x轴 = 时间
    y轴 = 相机在圆平面内偏移 rho
    z轴 = 相机相对圆平面法向距离 h

    每个时间切片上：
    - 圆心固定在 (t, 0, 0)
    - 圆法向量画成沿 +z 的绿色箭头
    - 相机画在 (t, rho, h)
    - 相机朝向用朝向圆心的蓝色箭头表示
    """
    if len(pose_history) == 0:
        print("没有有效 pose_history，无法绘制 3D pose strip")
        return

    arr = np.array(pose_history, dtype=np.float64)

    frame_ids = arr[:, 0]
    t = frame_ids / video_fps

    C = arr[:, 1:4]
    n = arr[:, 4:7]

    d = np.linalg.norm(C, axis=1)
    h = -np.sum(n * C, axis=1)
    rho = np.sqrt(np.clip(d**2 - h**2, 0.0, None))

    out_path = BASE_DIR / "pose_strip_3d.png"

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(t, rho, h, linewidth=1.5, label="camera trajectory")

    z_arrow_len = max(np.max(np.abs(h)) * 0.15, 10.0)
    cam_arrow_len = max(np.max(d) * 0.12, 10.0)

    for k in range(0, len(t), step):
        tk = t[k]
        yk = rho[k]
        zk = h[k]

        ax.scatter([tk], [0.0], [0.0], c='r', s=20)
        ax.quiver(tk, 0.0, 0.0,
                  0.0, 0.0, z_arrow_len,
                  color='g', arrow_length_ratio=0.2)

        ax.scatter([tk], [yk], [zk], c='b', s=20)

        dy = -yk
        dz = -zk
        norm = np.hypot(dy, dz)
        if norm > 1e-9:
            dy = dy / norm * cam_arrow_len
            dz = dz / norm * cam_arrow_len
            ax.quiver(tk, yk, zk,
                      0.0, dy, dz,
                      color='b', arrow_length_ratio=0.25)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("In-plane Offset rho (mm)")
    ax.set_zlabel("Normal Distance h (mm)")
    ax.set_title("Relative Pose Strip (Circle Fixed, Camera Moving)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"已保存 3D 位姿条带图: {out_path.name}")


# =========================
# 椭圆边缘跟踪
# =========================
def get_ellipse_points_and_normals(center, axes, angle_deg, num_points=40):
    cx, cy = center
    a, b = axes
    angle_rad = np.deg2rad(angle_deg)
    t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

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

    valid = (sample_x >= 0) & (sample_x < w - 1) & (sample_y >= 0) & (sample_y < h - 1)
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

    best_x = vx[pos_idx]
    best_y = vy[pos_idx]
    return (best_x, best_y)


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
        if key == 13 and len(points) >= 5:  # Enter
            pts_arr = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
            ell = cv2.fitEllipse(pts_arr)
            cv2.destroyWindow(window_name)
            return ell
        elif key == ord('c'):
            points.clear()
            img = frame.copy()
        elif key == 27 or key == ord('q'):  # ESC or q
            cv2.destroyWindow(window_name)
            return None


def process_side(gray, prev_ellipse, num_samples, search_length, template):
    """
    处理单个视图：返回 (ellipse, detected_pts)
    """
    (cx, cy), (a, b), angle = prev_ellipse
    pred_cx, pred_cy = cx, cy

    pts, normals = get_ellipse_points_and_normals(
        (pred_cx, pred_cy), (a / 2, b / 2), angle, num_samples
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


# =========================
# 主程序
# =========================
def main():
    params = load_stereo_params()

    K_L = params.K_left
    D_L = params.D_left
    K_R = params.K_right
    D_R = params.D_right
    R_LR = params.R
    t_LR = params.T

    print("使用相机参数：")
    print(params)

    video_path = BASE_DIR.parent / "video" / "video_0000.avi"
    if not video_path.exists():
        print(f"找不到视频文件: {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频 FPS: {video_fps}, 总帧数: {total_frames}")

    ret, frame = cap.read()
    if not ret:
        print("无法读取视频第一帧")
        cap.release()
        return

    h, w = frame.shape[:2]
    w_left = w // 2
    left_frame = frame[:, :w_left]
    right_frame = frame[:, w_left:]

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

    NUM_SAMPLES = 50
    SEARCH_LENGTH = 50
    TEMPLATE = np.array([-1, -2, -4, -8, -16, 0, 16, 8, 4, 2, 1])

    prev_ellipse_L = init_ellipse_L
    prev_ellipse_R = init_ellipse_R

    output_path = BASE_DIR / "output_tracking_fast.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(output_path), fourcc, video_fps, (w, h))

    t_proc = time.perf_counter()
    frame_idx = 1

    metric_frame_ids = []
    metric_scores = []
    metric_angles = []

    pose_history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        left_half = frame[:, :w_left]
        right_half = frame[:, w_left:]
        gray_L = cv2.cvtColor(left_half, cv2.COLOR_BGR2GRAY)
        gray_R = cv2.cvtColor(right_half, cv2.COLOR_BGR2GRAY)

        # 每帧初始化，避免残留上一帧变量
        pL_center = None
        pR_center = None
        C_L = None
        C_R = None

        try:
            prev_ellipse_L, detected_pts_L = process_side(
                gray_L, prev_ellipse_L, NUM_SAMPLES, SEARCH_LENGTH, TEMPLATE
            )
            prev_ellipse_R, detected_pts_R = process_side(
                gray_R, prev_ellipse_R, NUM_SAMPLES, SEARCH_LENGTH, TEMPLATE
            )

            C_img_L = ellipse_to_conic(prev_ellipse_L)
            C_img_R = ellipse_to_conic(prev_ellipse_R)

            Q_L = conic_to_Q(C_img_L, K_L)
            Q_R = conic_to_Q(C_img_R, K_R)

            normal_candidates_L = solve_normal_candidates_from_Q(Q_L)
            normal_candidates_R = solve_normal_candidates_from_Q(Q_R)

            best_nL, best_nR, normal_match_info = disambiguate_normals_by_stereo(
                normal_candidates_L,
                normal_candidates_R,
                R_LR
            )

            score = normal_match_info["score"]
            angle = normal_match_info["angle_deg"]

            # 所有帧都记录 score / angle，便于完整分析
            metric_frame_ids.append(frame_idx)
            metric_scores.append(score)
            metric_angles.append(angle)

            valid_normal = (score > 0.99) and (angle < 8.0)

            if valid_normal:
                qL_center = solve_projected_center_from_Q_and_normal(Q_L, best_nL)
                qR_center = solve_projected_center_from_Q_and_normal(Q_R, best_nR)

                C_L, C_R, valid_depth = triangulate_center_from_normalized_points(
                    qL_center,
                    qR_center,
                    R_LR,
                    t_LR
                )

                if valid_depth:
                    # 统一法向量方向：让法向量朝向相机
                    if np.dot(best_nL, C_L) > 0:
                        best_nL = -best_nL
                        best_nR = -best_nR

                    pL_center = normalized_to_pixel(qL_center, K_L)
                    pR_center = normalized_to_pixel(qR_center, K_R)

                    pose_history.append([
                        frame_idx,
                        C_L[0], C_L[1], C_L[2],
                        best_nL[0], best_nL[1], best_nL[2],
                        score, angle
                    ])
                else:
                    print(f"帧 {frame_idx}: 三角化圆心不满足正深度")
            else:
                print(f"帧 {frame_idx}: 法向量不稳定, score={score:.4f}, angle={angle:.2f}")

        except Exception as e:
            print(f"帧 {frame_idx}: 处理失败: {e}")
            detected_pts_L = []
            detected_pts_R = []

        # ===== 绘图 =====
        display_L = left_half.copy()
        display_R = right_half.copy()

        for (x, y) in detected_pts_L:
            cv2.circle(display_L, (int(x), int(y)), 2, (0, 0, 255), -1)
        cv2.ellipse(display_L, prev_ellipse_L, (0, 255, 0), 2)

        for (x, y) in detected_pts_R:
            cv2.circle(display_R, (int(x), int(y)), 2, (0, 0, 255), -1)
        cv2.ellipse(display_R, prev_ellipse_R, (0, 255, 0), 2)

        if pL_center is not None:
            cv2.circle(display_L, (int(pL_center[0]), int(pL_center[1])), 6, (255, 0, 0), -1)
            cv2.putText(display_L, "center",
                        (int(pL_center[0]) + 8, int(pL_center[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        if pR_center is not None:
            cv2.circle(display_R, (int(pR_center[0]), int(pR_center[1])), 6, (255, 0, 0), -1)
            cv2.putText(display_R, "center",
                        (int(pR_center[0]) + 8, int(pR_center[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        if len(metric_scores) > 0:
            last_score = metric_scores[-1]
            last_angle = metric_angles[-1]
            cv2.putText(display_L, f"score={last_score:.4f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(display_L, f"angle={last_angle:.2f} deg",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if C_L is not None:
            cv2.putText(display_L, f"Cx={C_L[0]:.1f} mm",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
            cv2.putText(display_L, f"Cy={C_L[1]:.1f} mm",
                        (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
            cv2.putText(display_L, f"Cz={C_L[2]:.1f} mm",
                        (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        combined = np.hstack([display_L, display_R])
        cv2.line(combined, (w_left, 0), (w_left, h), (100, 100, 100), 1)

        out.write(combined)
        cv2.imshow("Stereo Tracking + Circle Pose", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        if frame_idx % 30 == 0:
            proc_fps = 30 / (time.perf_counter() - t_proc)
            t_proc = time.perf_counter()
            print(
                f"帧 {frame_idx}/{total_frames}, "
                f"左点: {len(detected_pts_L)}, 右点: {len(detected_pts_R)}, "
                f"处理FPS: {proc_fps:.2f}"
            )

    # ===== 保存输出 =====
    save_metric_plots(metric_frame_ids, metric_scores, metric_angles)
    save_relative_pose_plots(pose_history, video_fps)
    save_pose_strip_3d(pose_history, video_fps, step=10)

    pose_csv = BASE_DIR / "pose_history.csv"
    if len(pose_history) > 0:
        pose_arr = np.array(pose_history, dtype=np.float64)
        np.savetxt(
            pose_csv,
            pose_arr,
            delimiter=",",
            header="frame,Cx_mm,Cy_mm,Cz_mm,nx,ny,nz,score,angle_deg",
            comments=""
        )
        print(f"已保存 pose_history.csv: {pose_csv.name}")
    else:
        print("没有有效 pose_history 可保存")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"输出视频已保存: {output_path.name}")


if __name__ == "__main__":
    main()