import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
from StereoParams import ours_params

# ========================= 配置 =========================
SCRIPT_DIR = Path(__file__).resolve().parent
VIDEO_CANDIDATES = [
    SCRIPT_DIR / "video" / "video_0000.avi",
    SCRIPT_DIR.parent / "video" / "video_0000.avi",
]

NUM_SAMPLES = 100
SEARCH_LENGTH = 50
EDGE_TEMPLATE = np.array([-1, -2, -4, -8, -16, 0, 16, 8, 4, 2, 1], dtype=np.float64)

# 固定数组只计算一次
NORMAL_DISTANCES = np.arange(
    -SEARCH_LENGTH,
    SEARCH_LENGTH + 1,
    dtype=np.float64,
)
EDGE_TEMPLATE_KERNEL = EDGE_TEMPLATE[::-1].copy()

MIN_NORMAL_SCORE = 0.99
MAX_NORMAL_ANGLE_DEG = 8.0

# 如果视频已经去畸变或立体校正，请保持 False
USE_UNDISTORT = False

# 3D 图里最多画多少个相机模型
MAX_CAMERA_MODELS = 18


# ========================= 基础工具 =========================
def find_video_path():
    for path in VIDEO_CANDIDATES:
        if path.exists():
            return path
    tried = "\n".join(f"  {p}" for p in VIDEO_CANDIDATES)
    raise FileNotFoundError("找不到 video_0000.avi，已尝试：\n" + tried)


def normalize_vector(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("向量模长过小")
    return v / n


def skew(v):
    x, y, z = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def rotation_align_vectors(source, target):
    """
    求 R，使 R @ source = target。
    使用最小旋转；单圆不可观测的绕法向量偏航被固定为零。
    """
    a = normalize_vector(source)
    b = normalize_vector(target)

    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    v = np.cross(a, b)
    s = np.linalg.norm(v)

    if s < 1e-12:
        if c > 0:
            return np.eye(3)

        helper = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        axis = normalize_vector(np.cross(a, helper))
        return -np.eye(3) + 2.0 * np.outer(axis, axis)

    K = skew(v)
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))


def resize_for_display(image, max_width=1600, max_height=900):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale >= 1.0:
        return image
    return cv2.resize(
        image,
        (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


# ========================= 椭圆与解析定位 =========================
def ellipse_to_conic(ellipse):
    (cx, cy), (width, height), angle_deg = ellipse
    a = width / 2.0
    b = height / 2.0

    if a <= 1e-9 or b <= 1e-9:
        raise ValueError("椭圆半轴过小")

    theta = np.deg2rad(angle_deg)
    ct, st = np.cos(theta), np.sin(theta)

    R2 = np.array([[ct, -st], [st, ct]], dtype=np.float64)
    D = np.diag([1.0 / (a * a), 1.0 / (b * b)])
    A = R2 @ D @ R2.T

    c = np.array([[cx], [cy]], dtype=np.float64)

    C_img = np.zeros((3, 3), dtype=np.float64)
    C_img[:2, :2] = A
    C_img[:2, 2:3] = -A @ c
    C_img[2:3, :2] = (-A @ c).T
    C_img[2, 2] = float((c.T @ A @ c - 1.0).item())

    return 0.5 * (C_img + C_img.T)


def conic_to_Q(C_img, K):
    Q = K.T @ C_img @ K
    Q = 0.5 * (Q + Q.T)

    scale = np.linalg.norm(Q)
    if scale < 1e-12:
        raise ValueError("Q 尺度过小")

    return Q / scale


def solve_normal_candidates_from_Q(Q):
    Q = np.asarray(Q, dtype=np.float64).reshape(3, 3)
    Q = 0.5 * (Q + Q.T)

    eigvals, eigvecs = np.linalg.eigh(Q)
    eps = 1e-10

    num_pos = int(np.sum(eigvals > eps))
    num_neg = int(np.sum(eigvals < -eps))

    if num_pos == 1 and num_neg == 2:
        Q = -Q
        eigvals, eigvecs = np.linalg.eigh(Q)
        num_pos = int(np.sum(eigvals > eps))
        num_neg = int(np.sum(eigvals < -eps))

    if num_pos != 2 or num_neg != 1:
        raise ValueError(f"Q 特征值符号异常: {eigvals}")

    pos_idx = np.where(eigvals > eps)[0]
    neg_idx = np.where(eigvals < -eps)[0]
    pos_idx = pos_idx[np.argsort(eigvals[pos_idx])[::-1]]

    idx1 = int(pos_idx[0])
    idx2 = int(pos_idx[1])
    idx3 = int(neg_idx[0])

    lam1 = float(eigvals[idx1])
    lam2 = float(eigvals[idx2])
    lam3 = float(eigvals[idx3])

    V = np.column_stack([
        eigvecs[:, idx1],
        eigvecs[:, idx2],
        eigvecs[:, idx3],
    ])

    if np.linalg.det(V) < 0:
        V[:, 2] *= -1.0

    den = lam1 - lam3
    if abs(den) < 1e-12:
        raise ValueError("特征值退化")

    alpha = np.sqrt(max((lam1 - lam2) / den, 0.0))
    beta = np.sqrt(max((lam2 - lam3) / den, 0.0))

    n1 = normalize_vector(V @ np.array([alpha, 0.0, beta]))
    n2 = normalize_vector(V @ np.array([-alpha, 0.0, beta]))

    return [n1, n2]


def disambiguate_normals_by_stereo(normals_L, normals_R, R_LR):
    best_score = -1.0
    best_nL = None
    best_nR = None
    best_info = None

    for idx_L, nL in enumerate(normals_L):
        nL = normalize_vector(nL)
        nL_to_R = normalize_vector(R_LR @ nL)

        for idx_R, nR in enumerate(normals_R):
            nR = normalize_vector(nR)

            dot_raw = float(np.dot(nL_to_R, nR))
            score = abs(dot_raw)

            if score > best_score:
                best_score = score
                best_nL = nL
                best_nR = nR if dot_raw >= 0 else -nR
                best_info = {
                    "idx_L": idx_L,
                    "idx_R": idx_R,
                    "score": score,
                    "angle_deg": float(
                        np.rad2deg(np.arccos(np.clip(score, -1.0, 1.0)))
                    ),
                }

    return best_nL, best_nR, best_info


def solve_projected_center_from_Q_and_normal(Q, normal):
    Q = 0.5 * (Q + Q.T)
    q = np.linalg.solve(Q, normalize_vector(normal))

    if abs(q[2]) < 1e-12:
        raise ValueError("圆心投影第三维接近 0")

    return q / q[2]


def triangulate_center_from_normalized_points(qL, qR, R_LR, t_LR):
    qL = np.asarray(qL, dtype=np.float64).reshape(3)
    qR = np.asarray(qR, dtype=np.float64).reshape(3)
    t_LR = np.asarray(t_LR, dtype=np.float64).reshape(3, 1)

    P_L = np.hstack([np.eye(3), np.zeros((3, 1))])
    P_R = np.hstack([R_LR, t_LR])

    A = np.vstack([
        qL[0] * P_L[2] - P_L[0],
        qL[1] * P_L[2] - P_L[1],
        qR[0] * P_R[2] - P_R[0],
        qR[1] * P_R[2] - P_R[1],
    ])

    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]

    if abs(X_h[3]) < 1e-12:
        raise ValueError("三角化齐次坐标 W 接近 0")

    C_L = X_h[:3] / X_h[3]
    C_R = (R_LR @ C_L.reshape(3, 1) + t_LR).reshape(3)

    valid_depth = bool(C_L[2] > 0.0 and C_R[2] > 0.0)
    return C_L, C_R, valid_depth


def normalized_to_pixel(q, K):
    p = K @ np.asarray(q, dtype=np.float64).reshape(3)

    if abs(p[2]) < 1e-12:
        raise ValueError("像素齐次坐标第三维接近 0")

    p /= p[2]
    return p[:2]


# ========================= 椭圆跟踪 =========================
def get_ellipse_points_and_normals(center, axes, angle_deg, num_points=40):
    cx, cy = center
    a, b = axes
    angle = np.deg2rad(angle_deg)

    t = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)

    x = cx + a * np.cos(t) * np.cos(angle) - b * np.sin(t) * np.sin(angle)
    y = cy + a * np.cos(t) * np.sin(angle) + b * np.sin(t) * np.cos(angle)

    dx = -a * np.sin(t) * np.cos(angle) - b * np.cos(t) * np.sin(angle)
    dy = -a * np.sin(t) * np.sin(angle) + b * np.cos(t) * np.cos(angle)

    norm = np.hypot(dx, dy)
    norm[norm < 1e-12] = 1.0

    nx = -dy / norm
    ny = dx / norm

    return np.column_stack((x, y)), np.column_stack((nx, ny))


def detect_ring_centerlines(gray, points, normals, distances, template_kernel):
    """
    同时沿着全部法线方向提取一维像素，并使用模板矩阵化卷积寻找边缘。
    """
    h, w = gray.shape

    sample_x = points[:, 0, None] + distances[None, :] * normals[:, 0, None]
    sample_y = points[:, 1, None] + distances[None, :] * normals[:, 1, None]

    valid = (
        (sample_x >= 0)
        & (sample_x < w - 1)
        & (sample_y >= 0)
        & (sample_y < h - 1)
    )

    # 为了安全读取先限制索引；越界位置随后由 valid_windows 屏蔽
    sample_x_int = np.clip(sample_x.astype(np.int32), 0, w - 1)
    sample_y_int = np.clip(sample_y.astype(np.int32), 0, h - 1)

    values = gray[sample_y_int, sample_x_int].astype(np.float64)

    template_length = len(template_kernel)

    value_windows = np.lib.stride_tricks.sliding_window_view(
        values,
        template_length,
        axis=1,
    )
    valid_windows = np.lib.stride_tricks.sliding_window_view(
        valid,
        template_length,
        axis=1,
    )

    valid_windows = np.all(valid_windows, axis=-1)

    # 保持原代码要求：一条法线至少要有 len(template) + 4 个有效采样点
    enough_values = np.sum(valid, axis=1) >= template_length + 4
    valid_windows &= enough_values[:, None]

    # 与 np.convolve(values, template, mode="valid") 保持相同方向
    response = value_windows @ template_kernel

    pos_response = np.where(valid_windows, response, -np.inf)
    neg_response = np.where(valid_windows, response, np.inf)

    line_valid = np.any(valid_windows, axis=1)

    pos_idx = np.argmax(pos_response, axis=1)
    neg_idx = np.argmin(neg_response, axis=1)
    row_idx = np.arange(len(points))

    pos_value = pos_response[row_idx, pos_idx]
    neg_value = neg_response[row_idx, neg_idx]

    detected_valid = (
        line_valid
        & (pos_value >= 15.0)
        & (neg_value <= -15.0)
    )

    best_x = sample_x[row_idx, pos_idx]
    best_y = sample_y[row_idx, pos_idx]

    detected = np.column_stack((best_x, best_y))
    return detected[detected_valid].tolist()


def mark_ellipse_on_frame(frame, window_name):
    points = []
    shown = resize_for_display(frame)
    scale_x = frame.shape[1] / shown.shape[1]
    scale_y = frame.shape[0] / shown.shape[0]

    def redraw():
        temp = frame.copy()

        for p in points:
            cv2.circle(temp, p, 4, (0, 255, 0), -1)

        if len(points) >= 5:
            pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
            cv2.ellipse(temp, cv2.fitEllipse(pts), (0, 255, 0), 2)

        cv2.putText(
            temp,
            "Left click: mark | Enter: confirm | C: clear | Q/Esc: quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        cv2.imshow(window_name, resize_for_display(temp))

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((
                int(round(x * scale_x)),
                int(round(y * scale_y)),
            ))
            redraw()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, shown.shape[1], shown.shape[0])
    cv2.setMouseCallback(window_name, mouse_callback)
    redraw()

    while True:
        key = cv2.waitKey(50) & 0xFF

        if key == 13 and len(points) >= 5:
            pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
            ellipse = cv2.fitEllipse(pts)
            cv2.destroyWindow(window_name)
            return ellipse

        if key == ord("c"):
            points.clear()
            redraw()

        if key in (ord("q"), 27):
            cv2.destroyWindow(window_name)
            return None


def process_side(gray, prev_ellipse):
    (cx, cy), (width, height), angle = prev_ellipse

    points, normals = get_ellipse_points_and_normals(
        (cx, cy),
        (width / 2.0, height / 2.0),
        angle,
        NUM_SAMPLES,
    )

    detected = detect_ring_centerlines(
        gray,
        points,
        normals,
        NORMAL_DISTANCES,
        EDGE_TEMPLATE_KERNEL,
    )

    ellipse = prev_ellipse

    if len(detected) >= 5:
        pts = np.asarray(detected, dtype=np.float32).reshape(-1, 1, 2)
        ellipse = cv2.fitEllipse(pts)

    return ellipse, detected


# ========================= 交互式 3D 绘制 =========================
def observation_to_circle_fixed_pose(center_camera, normal_camera):
    """
    圆心固定为世界原点，圆法向量固定为 +Z。
    单个圆无法确定绕法向量偏航，因此使用最小旋转零偏航规范。
    """
    world_normal = np.array([0.0, 0.0, 1.0])
    R_world_from_camera = rotation_align_vectors(normal_camera, world_normal)
    camera_position_world = R_world_from_camera @ (-center_camera)

    return camera_position_world, R_world_from_camera


def draw_camera_frustum(ax, position, rotation, scale):
    depth = scale
    half_w = 0.55 * scale
    half_h = 0.38 * scale

    corners_local = np.array([
        [-half_w, -half_h, depth],
        [half_w, -half_h, depth],
        [half_w, half_h, depth],
        [-half_w, half_h, depth],
    ])

    corners_world = position.reshape(1, 3) + (rotation @ corners_local.T).T

    for corner in corners_world:
        ax.plot(
            [position[0], corner[0]],
            [position[1], corner[1]],
            [position[2], corner[2]],
            linewidth=1.0,
        )

    closed = np.vstack([corners_world, corners_world[0]])
    ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], linewidth=1.0)

    optical_axis = rotation[:, 2]
    ax.quiver(
        position[0],
        position[1],
        position[2],
        optical_axis[0],
        optical_axis[1],
        optical_axis[2],
        length=1.5 * scale,
        normalize=True,
        arrow_length_ratio=0.2,
    )


def set_axes_equal_3d(ax, points):
    points = np.asarray(points, dtype=np.float64)
    pmin = np.min(points, axis=0)
    pmax = np.max(points, axis=0)
    center = (pmin + pmax) / 2.0
    radius = max(float(np.max(pmax - pmin)) / 2.0, 1.0)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except Exception:
        pass


def show_interactive_3d_pose(pose_history, video_fps):
    """
    弹出两个可拖动 3D 窗口，不保存任何图像、CSV 或视频。
    """
    if len(pose_history) == 0:
        print("没有有效位姿，无法显示 3D 图")
        return

    history = np.asarray(pose_history, dtype=np.float64)
    frame_ids = history[:, 0]
    t = frame_ids / video_fps

    centers_camera = history[:, 1:4]
    normals_camera = history[:, 4:7]

    camera_positions = []
    camera_rotations = []

    for center, normal in zip(centers_camera, normals_camera):
        pos, rot = observation_to_circle_fixed_pose(center, normal)
        camera_positions.append(pos)
        camera_rotations.append(rot)

    camera_positions = np.asarray(camera_positions)
    camera_rotations = np.asarray(camera_rotations)

    distance = np.linalg.norm(camera_positions, axis=1)
    median_distance = float(np.median(distance))

    # 仅为显示比例，不参与计算
    circle_radius = max(20.0, 0.10 * median_distance)
    normal_length = max(30.0, 0.16 * median_distance)
    camera_scale = max(12.0, 0.055 * median_distance)

    draw_count = min(MAX_CAMERA_MODELS, len(camera_positions))
    draw_indices = np.unique(
        np.linspace(0, len(camera_positions) - 1, draw_count, dtype=int)
    )

    # ---------- 图 1：圆固定的真实 3D 规范图 ----------
    fig1 = plt.figure(figsize=(11, 8))
    ax1 = fig1.add_subplot(111, projection="3d")

    theta = np.linspace(0.0, 2.0 * np.pi, 240)
    circle_x = circle_radius * np.cos(theta)
    circle_y = circle_radius * np.sin(theta)
    circle_z = np.zeros_like(theta)

    ax1.plot(circle_x, circle_y, circle_z, linewidth=2.0, label="Fixed circle")
    ax1.scatter([0.0], [0.0], [0.0], s=60, label="Circle center")

    ax1.quiver(
        0.0, 0.0, 0.0,
        0.0, 0.0, 1.0,
        length=normal_length,
        normalize=True,
        arrow_length_ratio=0.16,
        label="Circle normal",
    )

    ax1.plot(
        camera_positions[:, 0],
        camera_positions[:, 1],
        camera_positions[:, 2],
        linewidth=1.8,
        label="Camera trajectory",
    )

    scatter = ax1.scatter(
        camera_positions[:, 0],
        camera_positions[:, 1],
        camera_positions[:, 2],
        c=t,
        cmap="viridis",
        s=18,
    )

    cbar = fig1.colorbar(scatter, ax=ax1, shrink=0.72, pad=0.08)
    cbar.set_label("Time (s)")

    for idx in draw_indices:
        draw_camera_frustum(
            ax1,
            camera_positions[idx],
            camera_rotations[idx],
            camera_scale,
        )
        ax1.text(
            camera_positions[idx, 0],
            camera_positions[idx, 1],
            camera_positions[idx, 2],
            f"{t[idx]:.1f}s",
            fontsize=8,
        )

    support_points = np.vstack([
        camera_positions,
        np.array([
            [circle_radius, 0.0, 0.0],
            [-circle_radius, 0.0, 0.0],
            [0.0, circle_radius, 0.0],
            [0.0, -circle_radius, 0.0],
            [0.0, 0.0, normal_length],
        ]),
    ])

    set_axes_equal_3d(ax1, support_points)

    ax1.set_xlabel("Circle-frame X (mm)")
    ax1.set_ylabel("Circle-frame Y (mm)")
    ax1.set_zlabel("Circle normal Z (mm)")
    ax1.set_title(
        "Interactive Relative Pose - Circle Fixed at Origin\n"
        "Yaw about circle normal is gauge-fixed"
    )
    ax1.legend(loc="upper left")
    ax1.view_init(elev=24, azim=-58)

    # ---------- 图 2：时间为横轴的 3D 条带图 ----------
    fig2 = plt.figure(figsize=(12, 8))
    ax2 = fig2.add_subplot(111, projection="3d")

    xw = camera_positions[:, 0]
    yw = camera_positions[:, 1]
    zw = camera_positions[:, 2]

    rho = np.hypot(xw, yw)
    h = zw
    optical_axes = camera_rotations[:, :, 2]

    ax2.plot(t, rho, h, linewidth=2.0, label="Camera trajectory")
    ax2.scatter(t, rho, h, c=t, cmap="viridis", s=18)

    # 圆心在空间里固定；在时间条带上形成一条中心线
    ax2.plot(
        t,
        np.zeros_like(t),
        np.zeros_like(t),
        linewidth=1.5,
        label="Fixed circle center",
    )

    strip_radius = max(10.0, 0.65 * circle_radius)
    strip_normal_length = max(20.0, 0.75 * normal_length)
    camera_arrow_length = max(20.0, 1.8 * camera_scale)

    for idx in draw_indices:
        ti = t[idx]

        # 圆平面在该时间切片的一条直径
        ax2.plot(
            [ti, ti],
            [-strip_radius, strip_radius],
            [0.0, 0.0],
            linewidth=1.2,
        )

        # 圆法向量
        ax2.quiver(
            ti, 0.0, 0.0,
            0.0, 0.0, 1.0,
            length=strip_normal_length,
            normalize=True,
            arrow_length_ratio=0.2,
        )

        if rho[idx] > 1e-9:
            radial_unit = np.array([xw[idx] / rho[idx], yw[idx] / rho[idx], 0.0])
        else:
            radial_unit = np.array([1.0, 0.0, 0.0])

        optical = optical_axes[idx]

        # 相机光轴投影到“平面内径向-圆法向”平面
        radial_component = float(np.dot(optical, radial_unit))
        normal_component = float(optical[2])

        projected_norm = np.hypot(radial_component, normal_component)

        if projected_norm > 1e-9:
            radial_component /= projected_norm
            normal_component /= projected_norm

            ax2.quiver(
                ti,
                rho[idx],
                h[idx],
                0.0,
                radial_component,
                normal_component,
                length=camera_arrow_length,
                normalize=True,
                arrow_length_ratio=0.23,
            )

        ax2.scatter([ti], [rho[idx]], [h[idx]], s=35)

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("In-plane offset from center (mm)")
    ax2.set_zlabel("Distance along circle normal (mm)")
    ax2.set_title(
        "Interactive 3D Time Strip\n"
        "Circle fixed; camera position and projected optical axis"
    )
    ax2.legend(loc="upper left")
    ax2.view_init(elev=25, azim=-62)

    try:
        ax2.set_box_aspect((2.4, 1.25, 1.45))
    except Exception:
        pass

    print("\n3D 图操作：")
    print("  左键拖动：旋转")
    print("  滚轮：缩放")
    print("  关闭两个 3D 窗口后程序结束\n")

    plt.show(block=True)


# ========================= 主程序 =========================
def main():
    params = ours_params

    K_L = np.asarray(params.K_left, dtype=np.float64)
    D_L = np.asarray(params.D_left, dtype=np.float64)
    K_R = np.asarray(params.K_right, dtype=np.float64)
    D_R = np.asarray(params.D_right, dtype=np.float64)
    R_LR = np.asarray(params.R, dtype=np.float64)
    t_LR = np.asarray(params.T, dtype=np.float64)

    print("使用相机参数：")
    print(params)

    try:
        video_path = find_video_path()
    except FileNotFoundError as e:
        print(e)
        return

    print(f"视频文件: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 1e-6:
        fps = 30.0
        print("无法读取有效 FPS，使用 30 FPS")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频 FPS: {fps:.3f}, 总帧数: {total_frames}")

    ok, first_frame = cap.read()

    if not ok:
        print("无法读取视频第一帧")
        cap.release()
        return

    h, w = first_frame.shape[:2]
    w_left = w // 2

    first_left = first_frame[:, :w_left]
    first_right = first_frame[:, w_left:]

    if USE_UNDISTORT:
        first_left = cv2.undistort(first_left, K_L, D_L, None, K_L)
        first_right = cv2.undistort(first_right, K_R, D_R, None, K_R)

    print("\n请在左图椭圆边缘点击至少 5 个点，然后按 Enter。")
    prev_ellipse_L = mark_ellipse_on_frame(first_left, "Mark Left Ellipse")
    if prev_ellipse_L is None:
        cap.release()
        return

    print("\n请在右图椭圆边缘点击至少 5 个点，然后按 Enter。")
    prev_ellipse_R = mark_ellipse_on_frame(first_right, "Mark Right Ellipse")
    if prev_ellipse_R is None:
        cap.release()
        return

    frame_idx = 1
    timer = time.perf_counter()

    # frame, Cx, Cy, Cz, nx, ny, nz, score, stereo_angle
    pose_history = []

    print("\n开始处理。实时窗口按 Q 或 Esc 结束并显示交互式 3D 图。")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1

        left_frame = frame[:, :w_left]
        right_frame = frame[:, w_left:]

        if USE_UNDISTORT:
            left_frame = cv2.undistort(left_frame, K_L, D_L, None, K_L)
            right_frame = cv2.undistort(right_frame, K_R, D_R, None, K_R)

        gray_L = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        gray_R = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

        detected_L = []
        detected_R = []

        pixel_center_L = None
        pixel_center_R = None
        C_L = None
        score = None
        angle_deg = None
        current_valid = False

        try:
            prev_ellipse_L, detected_L = process_side(gray_L, prev_ellipse_L)
            prev_ellipse_R, detected_R = process_side(gray_R, prev_ellipse_R)

            C_img_L = ellipse_to_conic(prev_ellipse_L)
            C_img_R = ellipse_to_conic(prev_ellipse_R)

            Q_L = conic_to_Q(C_img_L, K_L)
            Q_R = conic_to_Q(C_img_R, K_R)

            normals_L = solve_normal_candidates_from_Q(Q_L)
            normals_R = solve_normal_candidates_from_Q(Q_R)

            best_nL, best_nR, match_info = disambiguate_normals_by_stereo(
                normals_L,
                normals_R,
                R_LR,
            )

            score = float(match_info["score"])
            angle_deg = float(match_info["angle_deg"])

            valid_normal = (
                score >= MIN_NORMAL_SCORE
                and angle_deg <= MAX_NORMAL_ANGLE_DEG
            )

            if valid_normal:
                qL = solve_projected_center_from_Q_and_normal(Q_L, best_nL)
                qR = solve_projected_center_from_Q_and_normal(Q_R, best_nR)

                C_L, C_R, valid_depth = triangulate_center_from_normalized_points(
                    qL,
                    qR,
                    R_LR,
                    t_LR,
                )

                if valid_depth:
                    # 法向量统一为“圆心指向相机”
                    if np.dot(best_nL, C_L) > 0:
                        best_nL = -best_nL
                        best_nR = -best_nR

                    pixel_center_L = normalized_to_pixel(qL, K_L)
                    pixel_center_R = normalized_to_pixel(qR, K_R)

                    pose_history.append([
                        frame_idx,
                        C_L[0], C_L[1], C_L[2],
                        best_nL[0], best_nL[1], best_nL[2],
                        score,
                        angle_deg,
                    ])

                    current_valid = True

        except Exception as e:
            print(f"帧 {frame_idx}: 处理失败: {e}")

        # ---------- 实时显示，不保存 ----------
        display_L = left_frame.copy()
        display_R = right_frame.copy()

        for x, y in detected_L:
            cv2.circle(display_L, (int(round(x)), int(round(y))), 2, (0, 0, 255), -1)

        for x, y in detected_R:
            cv2.circle(display_R, (int(round(x)), int(round(y))), 2, (0, 0, 255), -1)

        cv2.ellipse(display_L, prev_ellipse_L, (0, 255, 0), 2)
        cv2.ellipse(display_R, prev_ellipse_R, (0, 255, 0), 2)

        if pixel_center_L is not None:
            p = (int(round(pixel_center_L[0])), int(round(pixel_center_L[1])))
            cv2.circle(display_L, p, 6, (255, 0, 0), -1)
            cv2.putText(
                display_L,
                "projected center",
                (p[0] + 8, p[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )

        if pixel_center_R is not None:
            p = (int(round(pixel_center_R[0])), int(round(pixel_center_R[1])))
            cv2.circle(display_R, p, 6, (255, 0, 0), -1)
            cv2.putText(
                display_R,
                "projected center",
                (p[0] + 8, p[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )

        if score is not None:
            cv2.putText(
                display_L,
                f"score: {score:.4f}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                display_L,
                f"angle: {angle_deg:.2f} deg",
                (12, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 255),
                2,
            )

        status = "VALID" if current_valid else "INVALID"
        status_color = (0, 255, 0) if current_valid else (0, 0, 255)

        cv2.putText(
            display_L,
            status,
            (12, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            status_color,
            2,
        )

        if C_L is not None and current_valid:
            cv2.putText(
                display_L,
                f"C: [{C_L[0]:.1f}, {C_L[1]:.1f}, {C_L[2]:.1f}] mm",
                (12, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (255, 255, 0),
                2,
            )

        combined = np.hstack([display_L, display_R])
        cv2.line(
            combined,
            (w_left, 0),
            (w_left, combined.shape[0]),
            (120, 120, 120),
            1,
        )

        cv2.imshow(
            "Stereo Circle Tracking - Q/Esc to finish",
            resize_for_display(combined),
        )

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break

        if frame_idx % 30 == 0:
            elapsed = time.perf_counter() - timer
            processing_fps = 30.0 / elapsed if elapsed > 1e-9 else 0.0
            timer = time.perf_counter()

            print(
                f"帧 {frame_idx}/{total_frames}, "
                f"左点 {len(detected_L)}, "
                f"右点 {len(detected_R)}, "
                f"有效位姿 {len(pose_history)}, "
                f"处理 FPS {processing_fps:.2f}"
            )

    cap.release()
    cv2.destroyAllWindows()

    print(f"视频处理结束，共获得 {len(pose_history)} 个有效位姿")

    # 只显示交互式 3D 图，不保存任何结果
    show_interactive_3d_pose(pose_history, fps)

    print("程序结束")


if __name__ == "__main__":
    main()