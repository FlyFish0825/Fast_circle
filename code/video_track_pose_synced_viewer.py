import cv2
import numpy as np
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from StereoParams import ours_params


# ============================================================
# 配置
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent

VIDEO_CANDIDATES = [
    SCRIPT_DIR / "video" / "video_0000.avi",
    SCRIPT_DIR.parent / "video" / "video_0000.avi",
]

NUM_SAMPLES = 50
SEARCH_LENGTH = 50
EDGE_TEMPLATE = np.array(
    [-1, -2, -4, -8, -16, 0, 16, 8, 4, 2, 1],
    dtype=np.float64
)

MIN_NORMAL_SCORE = 0.99
MAX_NORMAL_ANGLE_DEG = 8.0

# 原始视频若未去畸变，保持 True。
# 若视频本身已完成去畸变或立体校正，改为 False。
USE_UNDISTORT = True

# 左视频画面缓存为 JPEG，降低内存占用。
LEFT_VIEW_MAX_WIDTH = 960
JPEG_QUALITY = 85

# 3D 轨迹拖尾长度，单位：帧。
TRAIL_LENGTH_FRAMES = 90

# 播放速度倍率。
PLAYBACK_SPEED = 1.0

# 自动显示的圆半径只用于 3D 示意，不参与计算。
# 设为具体毫米值可固定显示半径；None 表示自动估计。
DISPLAY_CIRCLE_RADIUS_MM = None


# ============================================================
# 通用工具
# ============================================================
def find_video_path():
    for path in VIDEO_CANDIDATES:
        if path.exists():
            return path

    tried = "\n".join(f"  {p}" for p in VIDEO_CANDIDATES)
    raise FileNotFoundError(
        "找不到 video_0000.avi，已尝试：\n" + tried
    )


def normalize_vector(vector, eps=1e-12):
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    length = np.linalg.norm(vector)

    if length < eps:
        raise ValueError("向量模长过小，无法归一化")

    return vector / length


def skew(vector):
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)

    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0]
    ], dtype=np.float64)


def orthonormalize_rotation(rotation):
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)

    U, _, Vt = np.linalg.svd(rotation)
    result = U @ Vt

    if np.linalg.det(result) < 0.0:
        U[:, -1] *= -1.0
        result = U @ Vt

    return result


def rotation_align_vectors(source, target):
    """
    计算最小旋转 R，使 R @ source = target。
    """
    source = normalize_vector(source)
    target = normalize_vector(target)

    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    cross = np.cross(source, target)
    sine = np.linalg.norm(cross)

    if sine < 1e-12:
        if cosine > 0.0:
            return np.eye(3, dtype=np.float64)

        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(source[0]) > 0.9:
            helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        axis = normalize_vector(np.cross(source, helper))
        return -np.eye(3) + 2.0 * np.outer(axis, axis)

    K = skew(cross)

    return (
        np.eye(3)
        + K
        + K @ K * ((1.0 - cosine) / (sine * sine))
    )


def rotation_about_z(angle):
    cosine = np.cos(angle)
    sine = np.sin(angle)

    return np.array([
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)


def choose_continuous_circle_frame(normal_camera, previous_rotation):
    """
    构造 R_world_from_camera，使圆法向量映射到世界 +Z。

    单个圆无法观测绕法向量的绝对旋转。这里选择与上一有效帧
    最接近的偏航规范，使 3D 显示连续，不产生无意义的跳转。
    """
    world_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    base_rotation = rotation_align_vectors(
        normal_camera,
        world_normal
    )

    if previous_rotation is None:
        return orthonormalize_rotation(base_rotation)

    B = base_rotation @ previous_rotation.T

    a = B[0, 0] + B[1, 1]
    b = B[0, 1] - B[1, 0]
    yaw = np.arctan2(b, a)

    result = rotation_about_z(yaw) @ base_rotation
    return orthonormalize_rotation(result)


def resize_to_width(image, max_width):
    height, width = image.shape[:2]

    if width <= max_width:
        return image

    scale = max_width / width
    new_size = (
        int(round(width * scale)),
        int(round(height * scale))
    )

    return cv2.resize(
        image,
        new_size,
        interpolation=cv2.INTER_AREA
    )


def encode_bgr_frame(image):
    image = resize_to_width(image, LEFT_VIEW_MAX_WIDTH)

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )

    if not success:
        raise RuntimeError("视频帧 JPEG 编码失败")

    return encoded.tobytes()


def decode_rgb_frame(encoded_bytes):
    array = np.frombuffer(encoded_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)

    if bgr is None:
        raise RuntimeError("视频帧 JPEG 解码失败")

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ============================================================
# 椭圆矩阵、法向量和三角化
# ============================================================
def ellipse_to_conic(ellipse):
    (cx, cy), (width, height), angle_deg = ellipse

    a = width / 2.0
    b = height / 2.0

    if a <= 1e-9 or b <= 1e-9:
        raise ValueError("椭圆半轴过小")

    theta = np.deg2rad(angle_deg)
    cosine = np.cos(theta)
    sine = np.sin(theta)

    rotation_2d = np.array([
        [cosine, -sine],
        [sine, cosine]
    ], dtype=np.float64)

    diagonal = np.diag([
        1.0 / (a * a),
        1.0 / (b * b)
    ])

    A = rotation_2d @ diagonal @ rotation_2d.T
    center = np.array([[cx], [cy]], dtype=np.float64)

    conic = np.zeros((3, 3), dtype=np.float64)
    conic[:2, :2] = A
    conic[:2, 2:3] = -A @ center
    conic[2:3, :2] = (-A @ center).T
    conic[2, 2] = float(
        (center.T @ A @ center - 1.0).item()
    )

    return 0.5 * (conic + conic.T)


def conic_to_Q(conic, camera_matrix):
    camera_matrix = np.asarray(
        camera_matrix,
        dtype=np.float64
    ).reshape(3, 3)

    conic = np.asarray(
        conic,
        dtype=np.float64
    ).reshape(3, 3)

    Q = camera_matrix.T @ conic @ camera_matrix
    Q = 0.5 * (Q + Q.T)

    scale = np.linalg.norm(Q)

    if scale < 1e-12:
        raise ValueError("Q 的尺度过小")

    return Q / scale


def solve_normal_candidates_from_Q(Q):
    Q = np.asarray(Q, dtype=np.float64).reshape(3, 3)
    Q = 0.5 * (Q + Q.T)

    eigenvalues, eigenvectors = np.linalg.eigh(Q)

    epsilon = 1e-10
    positive_count = int(np.sum(eigenvalues > epsilon))
    negative_count = int(np.sum(eigenvalues < -epsilon))

    if positive_count == 1 and negative_count == 2:
        Q = -Q
        eigenvalues, eigenvectors = np.linalg.eigh(Q)
        positive_count = int(np.sum(eigenvalues > epsilon))
        negative_count = int(np.sum(eigenvalues < -epsilon))

    if positive_count != 2 or negative_count != 1:
        raise ValueError(
            f"Q 特征值符号异常: {eigenvalues}"
        )

    positive_indices = np.where(
        eigenvalues > epsilon
    )[0]

    negative_indices = np.where(
        eigenvalues < -epsilon
    )[0]

    positive_indices = positive_indices[
        np.argsort(
            eigenvalues[positive_indices]
        )[::-1]
    ]

    index_1 = int(positive_indices[0])
    index_2 = int(positive_indices[1])
    index_3 = int(negative_indices[0])

    lambda_1 = float(eigenvalues[index_1])
    lambda_2 = float(eigenvalues[index_2])
    lambda_3 = float(eigenvalues[index_3])

    V = np.column_stack([
        eigenvectors[:, index_1],
        eigenvectors[:, index_2],
        eigenvectors[:, index_3]
    ])

    if np.linalg.det(V) < 0.0:
        V[:, 2] *= -1.0

    denominator = lambda_1 - lambda_3

    if abs(denominator) < 1e-12:
        raise ValueError("特征值退化")

    alpha = np.sqrt(
        max(
            (lambda_1 - lambda_2) / denominator,
            0.0
        )
    )

    beta = np.sqrt(
        max(
            (lambda_2 - lambda_3) / denominator,
            0.0
        )
    )

    candidate_1 = normalize_vector(
        V @ np.array([alpha, 0.0, beta])
    )

    candidate_2 = normalize_vector(
        V @ np.array([-alpha, 0.0, beta])
    )

    return [candidate_1, candidate_2]


def disambiguate_normals_by_stereo(
    normals_left,
    normals_right,
    rotation_left_to_right
):
    rotation_left_to_right = np.asarray(
        rotation_left_to_right,
        dtype=np.float64
    ).reshape(3, 3)

    best_score = -1.0
    best_left = None
    best_right = None
    best_info = None

    for left_index, normal_left in enumerate(normals_left):
        normal_left = normalize_vector(normal_left)

        left_in_right = normalize_vector(
            rotation_left_to_right @ normal_left
        )

        for right_index, normal_right in enumerate(normals_right):
            normal_right = normalize_vector(normal_right)

            raw_dot = float(
                np.dot(left_in_right, normal_right)
            )

            score = abs(raw_dot)

            if score > best_score:
                aligned_right = (
                    normal_right
                    if raw_dot >= 0.0
                    else -normal_right
                )

                best_score = score
                best_left = normal_left
                best_right = aligned_right

                best_info = {
                    "idx_L": left_index,
                    "idx_R": right_index,
                    "score": score,
                    "angle_deg": float(
                        np.rad2deg(
                            np.arccos(
                                np.clip(
                                    score,
                                    -1.0,
                                    1.0
                                )
                            )
                        )
                    )
                }

    return best_left, best_right, best_info


def solve_projected_center_from_Q_and_normal(Q, normal):
    Q = np.asarray(Q, dtype=np.float64).reshape(3, 3)
    Q = 0.5 * (Q + Q.T)

    normal = normalize_vector(normal)

    projected_center = np.linalg.solve(Q, normal)

    if abs(projected_center[2]) < 1e-12:
        raise ValueError("圆心投影第三维接近 0")

    return projected_center / projected_center[2]


def triangulate_center_from_normalized_points(
    projected_left,
    projected_right,
    rotation_left_to_right,
    translation_left_to_right
):
    projected_left = np.asarray(
        projected_left,
        dtype=np.float64
    ).reshape(3)

    projected_right = np.asarray(
        projected_right,
        dtype=np.float64
    ).reshape(3)

    rotation_left_to_right = np.asarray(
        rotation_left_to_right,
        dtype=np.float64
    ).reshape(3, 3)

    translation_left_to_right = np.asarray(
        translation_left_to_right,
        dtype=np.float64
    ).reshape(3, 1)

    projection_left = np.hstack([
        np.eye(3),
        np.zeros((3, 1))
    ])

    projection_right = np.hstack([
        rotation_left_to_right,
        translation_left_to_right
    ])

    x_left = projected_left[0]
    y_left = projected_left[1]
    x_right = projected_right[0]
    y_right = projected_right[1]

    A = np.vstack([
        x_left * projection_left[2] - projection_left[0],
        y_left * projection_left[2] - projection_left[1],
        x_right * projection_right[2] - projection_right[0],
        y_right * projection_right[2] - projection_right[1]
    ])

    _, _, Vt = np.linalg.svd(A)
    homogeneous = Vt[-1]

    if abs(homogeneous[3]) < 1e-12:
        raise ValueError("三角化齐次坐标 W 接近 0")

    center_left = homogeneous[:3] / homogeneous[3]

    center_right = (
        rotation_left_to_right
        @ center_left.reshape(3, 1)
        + translation_left_to_right
    ).reshape(3)

    valid_depth = bool(
        center_left[2] > 0.0
        and center_right[2] > 0.0
    )

    return center_left, center_right, valid_depth


def normalized_to_pixel(projected_center, camera_matrix):
    projected_center = np.asarray(
        projected_center,
        dtype=np.float64
    ).reshape(3)

    pixel = camera_matrix @ projected_center

    if abs(pixel[2]) < 1e-12:
        raise ValueError("像素齐次坐标第三维接近 0")

    pixel /= pixel[2]
    return pixel[:2]


# ============================================================
# 椭圆跟踪
# ============================================================
def get_ellipse_points_and_normals(
    center,
    axes,
    angle_deg,
    num_points=40
):
    cx, cy = center
    a, b = axes

    angle = np.deg2rad(angle_deg)

    parameter = np.linspace(
        0.0,
        2.0 * np.pi,
        num_points,
        endpoint=False
    )

    x = (
        cx
        + a * np.cos(parameter) * np.cos(angle)
        - b * np.sin(parameter) * np.sin(angle)
    )

    y = (
        cy
        + a * np.cos(parameter) * np.sin(angle)
        + b * np.sin(parameter) * np.cos(angle)
    )

    dx = (
        -a * np.sin(parameter) * np.cos(angle)
        - b * np.cos(parameter) * np.sin(angle)
    )

    dy = (
        -a * np.sin(parameter) * np.sin(angle)
        + b * np.cos(parameter) * np.cos(angle)
    )

    tangent_length = np.hypot(dx, dy)
    tangent_length[tangent_length < 1e-12] = 1.0

    normal_x = -dy / tangent_length
    normal_y = dx / tangent_length

    return (
        np.column_stack((x, y)),
        np.column_stack((normal_x, normal_y))
    )


def detect_ring_centerline(
    gray_image,
    point,
    normal,
    search_length,
    template
):
    image_height, image_width = gray_image.shape

    x0, y0 = point
    normal_x, normal_y = normal

    distances = np.arange(
        -search_length,
        search_length + 1,
        dtype=np.float64
    )

    sample_x = x0 + distances * normal_x
    sample_y = y0 + distances * normal_y

    valid = (
        (sample_x >= 0)
        & (sample_x < image_width - 1)
        & (sample_y >= 0)
        & (sample_y < image_height - 1)
    )

    if not np.any(valid):
        return None

    valid_x = sample_x[valid]
    valid_y = sample_y[valid]

    values = gray_image[
        valid_y.astype(np.int32),
        valid_x.astype(np.int32)
    ].astype(np.float64)

    if len(values) < len(template) + 4:
        return None

    response = np.convolve(
        values,
        template,
        mode="valid"
    )

    positive_index = int(np.argmax(response))
    negative_index = int(np.argmin(response))

    positive_value = float(response[positive_index])
    negative_value = float(response[negative_index])

    edge_threshold = 15.0

    if (
        positive_value < edge_threshold
        or negative_value > -edge_threshold
    ):
        return None

    return (
        float(valid_x[positive_index]),
        float(valid_y[positive_index])
    )


def mark_ellipse_on_frame(frame, window_name):
    points = []

    shown = resize_to_width(frame, 1200)
    scale_x = frame.shape[1] / shown.shape[1]
    scale_y = frame.shape[0] / shown.shape[0]

    def redraw():
        preview = frame.copy()

        for point in points:
            cv2.circle(
                preview,
                point,
                4,
                (0, 255, 0),
                -1
            )

        if len(points) >= 5:
            point_array = np.asarray(
                points,
                dtype=np.float32
            ).reshape(-1, 1, 2)

            fitted = cv2.fitEllipse(point_array)
            cv2.ellipse(
                preview,
                fitted,
                (0, 255, 0),
                2
            )

        cv2.putText(
            preview,
            "Left click: mark | Enter: confirm | C: clear | Q/Esc: quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            window_name,
            resize_to_width(preview, 1200)
        )

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            original_x = int(round(x * scale_x))
            original_y = int(round(y * scale_y))

            points.append((original_x, original_y))
            redraw()

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        window_name,
        shown.shape[1],
        shown.shape[0]
    )

    cv2.setMouseCallback(
        window_name,
        mouse_callback
    )

    redraw()

    while True:
        key = cv2.waitKey(50) & 0xFF

        if key == 13 and len(points) >= 5:
            point_array = np.asarray(
                points,
                dtype=np.float32
            ).reshape(-1, 1, 2)

            fitted = cv2.fitEllipse(point_array)
            cv2.destroyWindow(window_name)
            return fitted

        if key == ord("c"):
            points.clear()
            redraw()

        if key in (27, ord("q")):
            cv2.destroyWindow(window_name)
            return None


def process_side(
    gray_image,
    previous_ellipse,
    num_samples,
    search_length,
    template
):
    (cx, cy), (width, height), angle_deg = previous_ellipse

    sample_points, sample_normals = (
        get_ellipse_points_and_normals(
            (cx, cy),
            (width / 2.0, height / 2.0),
            angle_deg,
            num_samples
        )
    )

    detected_points = []

    for point, normal in zip(
        sample_points,
        sample_normals
    ):
        detected = detect_ring_centerline(
            gray_image,
            point,
            normal,
            search_length,
            template
        )

        if detected is not None:
            detected_points.append(detected)

    fitted_ellipse = previous_ellipse

    if len(detected_points) >= 5:
        point_array = np.asarray(
            detected_points,
            dtype=np.float32
        ).reshape(-1, 1, 2)

        fitted_ellipse = cv2.fitEllipse(point_array)

    return fitted_ellipse, detected_points


# ============================================================
# 预处理视频：计算每帧位姿并缓存左画面
# ============================================================
def preprocess_video(
    video_path,
    K_left,
    D_left,
    K_right,
    D_right,
    rotation_left_to_right,
    translation_left_to_right
):
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    video_fps = float(
        capture.get(cv2.CAP_PROP_FPS)
    )

    if not np.isfinite(video_fps) or video_fps <= 1e-6:
        video_fps = 30.0

    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    success, first_frame = capture.read()

    if not success:
        capture.release()
        raise RuntimeError("无法读取视频第一帧")

    frame_height, frame_width = first_frame.shape[:2]
    left_width = frame_width // 2

    first_left = first_frame[:, :left_width]
    first_right = first_frame[:, left_width:]

    if USE_UNDISTORT:
        map_left_1, map_left_2 = cv2.initUndistortRectifyMap(
            K_left,
            D_left,
            None,
            K_left,
            (first_left.shape[1], first_left.shape[0]),
            cv2.CV_32FC1
        )

        map_right_1, map_right_2 = cv2.initUndistortRectifyMap(
            K_right,
            D_right,
            None,
            K_right,
            (first_right.shape[1], first_right.shape[0]),
            cv2.CV_32FC1
        )

        first_left = cv2.remap(
            first_left,
            map_left_1,
            map_left_2,
            cv2.INTER_LINEAR
        )

        first_right = cv2.remap(
            first_right,
            map_right_1,
            map_right_2,
            cv2.INTER_LINEAR
        )
    else:
        map_left_1 = None
        map_left_2 = None
        map_right_1 = None
        map_right_2 = None

    print()
    print("请在左图椭圆边缘点击至少 5 个点，然后按 Enter。")

    previous_left_ellipse = mark_ellipse_on_frame(
        first_left,
        "Mark Left Ellipse"
    )

    if previous_left_ellipse is None:
        capture.release()
        return None

    print()
    print("请在右图椭圆边缘点击至少 5 个点，然后按 Enter。")

    previous_right_ellipse = mark_ellipse_on_frame(
        first_right,
        "Mark Right Ellipse"
    )

    if previous_right_ellipse is None:
        capture.release()
        return None

    # 从第 0 帧重新开始统一处理。
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cached_left_frames = []
    frame_results = []

    start_time = time.perf_counter()

    frame_index = 0

    while True:
        success, frame = capture.read()

        if not success:
            break

        left_frame = frame[:, :left_width]
        right_frame = frame[:, left_width:]

        if USE_UNDISTORT:
            left_frame = cv2.remap(
                left_frame,
                map_left_1,
                map_left_2,
                cv2.INTER_LINEAR
            )

            right_frame = cv2.remap(
                right_frame,
                map_right_1,
                map_right_2,
                cv2.INTER_LINEAR
            )

        gray_left = cv2.cvtColor(
            left_frame,
            cv2.COLOR_BGR2GRAY
        )

        gray_right = cv2.cvtColor(
            right_frame,
            cv2.COLOR_BGR2GRAY
        )

        detected_left = []
        detected_right = []

        result = {
            "valid": False,
            "center_left": None,
            "normal_left": None,
            "score": np.nan,
            "angle_deg": np.nan,
            "pixel_center_left": None
        }

        try:
            previous_left_ellipse, detected_left = process_side(
                gray_left,
                previous_left_ellipse,
                NUM_SAMPLES,
                SEARCH_LENGTH,
                EDGE_TEMPLATE
            )

            previous_right_ellipse, detected_right = process_side(
                gray_right,
                previous_right_ellipse,
                NUM_SAMPLES,
                SEARCH_LENGTH,
                EDGE_TEMPLATE
            )

            conic_left = ellipse_to_conic(
                previous_left_ellipse
            )

            conic_right = ellipse_to_conic(
                previous_right_ellipse
            )

            Q_left = conic_to_Q(
                conic_left,
                K_left
            )

            Q_right = conic_to_Q(
                conic_right,
                K_right
            )

            normal_candidates_left = (
                solve_normal_candidates_from_Q(Q_left)
            )

            normal_candidates_right = (
                solve_normal_candidates_from_Q(Q_right)
            )

            best_left_normal, best_right_normal, match_info = (
                disambiguate_normals_by_stereo(
                    normal_candidates_left,
                    normal_candidates_right,
                    rotation_left_to_right
                )
            )

            score = float(match_info["score"])
            angle_deg = float(match_info["angle_deg"])

            result["score"] = score
            result["angle_deg"] = angle_deg

            normal_valid = (
                score >= MIN_NORMAL_SCORE
                and angle_deg <= MAX_NORMAL_ANGLE_DEG
            )

            if normal_valid:
                projected_left = (
                    solve_projected_center_from_Q_and_normal(
                        Q_left,
                        best_left_normal
                    )
                )

                projected_right = (
                    solve_projected_center_from_Q_and_normal(
                        Q_right,
                        best_right_normal
                    )
                )

                center_left, center_right, depth_valid = (
                    triangulate_center_from_normalized_points(
                        projected_left,
                        projected_right,
                        rotation_left_to_right,
                        translation_left_to_right
                    )
                )

                if depth_valid:
                    # 法向量统一为从圆心指向相机。
                    if np.dot(
                        best_left_normal,
                        center_left
                    ) > 0.0:
                        best_left_normal = -best_left_normal
                        best_right_normal = -best_right_normal

                    pixel_center_left = normalized_to_pixel(
                        projected_left,
                        K_left
                    )

                    result["valid"] = True
                    result["center_left"] = center_left
                    result["normal_left"] = best_left_normal
                    result["pixel_center_left"] = pixel_center_left

        except Exception as error:
            print(
                f"\n帧 {frame_index}: 计算失败: {error}"
            )

        # ----------------------------------------------------
        # 生成左图叠加画面，供交互式播放器显示。
        # ----------------------------------------------------
        display_left = left_frame.copy()

        for x, y in detected_left:
            cv2.circle(
                display_left,
                (int(round(x)), int(round(y))),
                2,
                (0, 0, 255),
                -1
            )

        cv2.ellipse(
            display_left,
            previous_left_ellipse,
            (0, 255, 0),
            2
        )

        if result["pixel_center_left"] is not None:
            pixel_center = result["pixel_center_left"]

            center_point = (
                int(round(pixel_center[0])),
                int(round(pixel_center[1]))
            )

            cv2.circle(
                display_left,
                center_point,
                6,
                (255, 0, 0),
                -1
            )

            cv2.putText(
                display_left,
                "projected center",
                (center_point[0] + 8, center_point[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1
            )

        status_text = (
            "VALID"
            if result["valid"]
            else "INVALID"
        )

        status_color = (
            (0, 255, 0)
            if result["valid"]
            else (0, 0, 255)
        )

        cv2.putText(
            display_left,
            status_text,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            status_color,
            2
        )

        if np.isfinite(result["score"]):
            cv2.putText(
                display_left,
                f"score: {result['score']:.4f}",
                (12, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 255),
                2
            )

            cv2.putText(
                display_left,
                f"angle: {result['angle_deg']:.2f} deg",
                (12, 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 255),
                2
            )

        if result["center_left"] is not None:
            center = result["center_left"]

            cv2.putText(
                display_left,
                (
                    f"C: [{center[0]:.1f}, "
                    f"{center[1]:.1f}, "
                    f"{center[2]:.1f}] mm"
                ),
                (12, 116),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (255, 255, 0),
                2
            )

        cv2.putText(
            display_left,
            (
                f"frame {frame_index}  "
                f"time {frame_index / video_fps:.3f} s"
            ),
            (12, display_left.shape[0] - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cached_left_frames.append(
            encode_bgr_frame(display_left)
        )

        frame_results.append(result)

        frame_index += 1

        if (
            frame_index % 30 == 0
            or frame_index == total_frames
        ):
            elapsed = time.perf_counter() - start_time
            speed = frame_index / max(elapsed, 1e-9)

            print(
                (
                    f"\r预处理 {frame_index}/{total_frames}, "
                    f"{speed:.2f} FPS"
                ),
                end="",
                flush=True
            )

    capture.release()

    print()
    print(
        f"预处理完成，共 {len(frame_results)} 帧。"
    )

    return {
        "fps": video_fps,
        "frames": cached_left_frames,
        "results": frame_results
    }


# ============================================================
# 将每帧圆观测转换为固定圆坐标系下的相机位姿
# ============================================================
def build_circle_fixed_pose_sequence(frame_results):
    frame_count = len(frame_results)

    camera_positions = np.full(
        (frame_count, 3),
        np.nan,
        dtype=np.float64
    )

    camera_rotations = np.full(
        (frame_count, 3, 3),
        np.nan,
        dtype=np.float64
    )

    valid_mask = np.zeros(
        frame_count,
        dtype=bool
    )

    previous_rotation = None

    for index, result in enumerate(frame_results):
        if not result["valid"]:
            continue

        center_camera = np.asarray(
            result["center_left"],
            dtype=np.float64
        ).reshape(3)

        normal_camera = normalize_vector(
            result["normal_left"]
        )

        rotation_world_from_camera = (
            choose_continuous_circle_frame(
                normal_camera,
                previous_rotation
            )
        )

        camera_position_world = (
            rotation_world_from_camera
            @ (-center_camera)
        )

        camera_positions[index] = camera_position_world
        camera_rotations[index] = rotation_world_from_camera
        valid_mask[index] = True

        previous_rotation = rotation_world_from_camera

    return camera_positions, camera_rotations, valid_mask


# ============================================================
# 同步交互式查看器
# ============================================================
class SynchronizedPoseViewer:
    def __init__(
        self,
        cached_frames,
        frame_results,
        camera_positions,
        camera_rotations,
        valid_mask,
        video_fps
    ):
        self.cached_frames = cached_frames
        self.frame_results = frame_results
        self.camera_positions = camera_positions
        self.camera_rotations = camera_rotations
        self.valid_mask = valid_mask
        self.video_fps = video_fps

        self.frame_count = len(cached_frames)
        self.current_index = 0
        self.playing = False

        self.valid_indices = np.where(valid_mask)[0]

        self._prepare_geometry()
        self._create_figure()
        self._create_timer()
        self.update_frame(0)

    def _prepare_geometry(self):
        valid_positions = self.camera_positions[
            self.valid_mask
        ]

        if len(valid_positions) == 0:
            raise RuntimeError(
                "没有有效三维位姿，无法打开查看器"
            )

        distances = np.linalg.norm(
            valid_positions,
            axis=1
        )

        median_distance = float(
            np.median(distances)
        )

        if DISPLAY_CIRCLE_RADIUS_MM is None:
            self.circle_radius = max(
                20.0,
                0.10 * median_distance
            )
        else:
            self.circle_radius = float(
                DISPLAY_CIRCLE_RADIUS_MM
            )

        self.normal_length = max(
            30.0,
            0.18 * median_distance
        )

        self.camera_scale = max(
            12.0,
            0.055 * median_distance
        )

        circle_extent = np.array([
            [self.circle_radius, 0.0, 0.0],
            [-self.circle_radius, 0.0, 0.0],
            [0.0, self.circle_radius, 0.0],
            [0.0, -self.circle_radius, 0.0],
            [0.0, 0.0, self.normal_length]
        ])

        all_points = np.vstack([
            valid_positions,
            circle_extent
        ])

        minimum = np.min(all_points, axis=0)
        maximum = np.max(all_points, axis=0)

        center = 0.5 * (minimum + maximum)

        radius = 0.55 * max(
            maximum[0] - minimum[0],
            maximum[1] - minimum[1],
            maximum[2] - minimum[2],
            1.0
        )

        self.axis_center = center
        self.axis_radius = radius

    def _create_figure(self):
        self.figure = plt.figure(
            figsize=(16, 8.5)
        )

        grid = self.figure.add_gridspec(
            1,
            2,
            width_ratios=[1.08, 1.0],
            left=0.035,
            right=0.98,
            top=0.93,
            bottom=0.16,
            wspace=0.06
        )

        self.video_axis = self.figure.add_subplot(
            grid[0, 0]
        )

        self.pose_axis = self.figure.add_subplot(
            grid[0, 1],
            projection="3d"
        )

        first_rgb = decode_rgb_frame(
            self.cached_frames[0]
        )

        self.video_artist = self.video_axis.imshow(
            first_rgb
        )

        self.video_axis.set_axis_off()
        self.video_title = self.video_axis.set_title(
            "Left video"
        )

        # 固定圆
        parameter = np.linspace(
            0.0,
            2.0 * np.pi,
            240
        )

        circle_x = (
            self.circle_radius * np.cos(parameter)
        )

        circle_y = (
            self.circle_radius * np.sin(parameter)
        )

        circle_z = np.zeros_like(parameter)

        self.pose_axis.plot(
            circle_x,
            circle_y,
            circle_z,
            linewidth=2.0,
            label="Fixed circle"
        )

        self.pose_axis.scatter(
            [0.0],
            [0.0],
            [0.0],
            s=60,
            label="Circle center"
        )

        self.pose_axis.quiver(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            length=self.normal_length,
            normalize=True,
            arrow_length_ratio=0.16,
            label="Circle normal"
        )

        self.trail_line, = self.pose_axis.plot(
            [],
            [],
            [],
            linewidth=2.3,
            label="Camera trail"
        )

        self.current_point = self.pose_axis.scatter(
            [],
            [],
            [],
            s=70,
            label="Current camera"
        )

        self.frustum_lines = []

        # 4 条中心到角点 + 4 条矩形边
        for _ in range(8):
            line, = self.pose_axis.plot(
                [],
                [],
                [],
                linewidth=1.4
            )
            self.frustum_lines.append(line)

        self.axis_lines = []

        for _ in range(3):
            line, = self.pose_axis.plot(
                [],
                [],
                [],
                linewidth=2.0
            )
            self.axis_lines.append(line)

        self.optical_axis_line, = self.pose_axis.plot(
            [],
            [],
            [],
            linewidth=2.2,
            label="Optical axis"
        )

        self.camera_text = self.pose_axis.text(
            0.0,
            0.0,
            0.0,
            ""
        )

        center = self.axis_center
        radius = self.axis_radius

        self.pose_axis.set_xlim(
            center[0] - radius,
            center[0] + radius
        )

        self.pose_axis.set_ylim(
            center[1] - radius,
            center[1] + radius
        )

        self.pose_axis.set_zlim(
            center[2] - radius,
            center[2] + radius
        )

        try:
            self.pose_axis.set_box_aspect(
                (1.0, 1.0, 1.0)
            )
        except Exception:
            pass

        self.pose_axis.set_xlabel(
            "Circle-frame X (mm)"
        )

        self.pose_axis.set_ylabel(
            "Circle-frame Y (mm)"
        )

        self.pose_axis.set_zlabel(
            "Circle normal Z (mm)"
        )

        self.pose_axis.set_title(
            "3D relative position and pose"
        )

        self.pose_axis.legend(
            loc="upper left"
        )

        self.pose_axis.view_init(
            elev=24,
            azim=-58
        )

        # 时间条
        slider_axis = self.figure.add_axes(
            [0.15, 0.065, 0.70, 0.035]
        )

        self.time_slider = Slider(
            ax=slider_axis,
            label="Time (s)",
            valmin=0.0,
            valmax=max(
                (self.frame_count - 1)
                / self.video_fps,
                1e-6
            ),
            valinit=0.0,
            valstep=1.0 / self.video_fps
        )

        self.time_slider.on_changed(
            self._on_slider_changed
        )

        play_axis = self.figure.add_axes(
            [0.035, 0.052, 0.085, 0.052]
        )

        self.play_button = Button(
            play_axis,
            "Play"
        )

        self.play_button.on_clicked(
            self._toggle_play
        )

        self.status_text = self.figure.text(
            0.88,
            0.069,
            "",
            ha="left",
            va="center"
        )

        self.figure.canvas.mpl_connect(
            "key_press_event",
            self._on_key_press
        )

        self.figure.canvas.mpl_connect(
            "close_event",
            self._on_close
        )

    def _create_timer(self):
        interval_ms = max(
            10,
            int(
                round(
                    1000.0
                    / (
                        self.video_fps
                        * PLAYBACK_SPEED
                    )
                )
            )
        )

        self.timer = self.figure.canvas.new_timer(
            interval=interval_ms
        )

        self.timer.add_callback(
            self._on_timer
        )

        self.timer.start()

    def _nearest_previous_valid_index(self, frame_index):
        candidates = self.valid_indices[
            self.valid_indices <= frame_index
        ]

        if len(candidates) == 0:
            return None

        return int(candidates[-1])

    def _update_trail(self, frame_index):
        trail_start = max(
            0,
            frame_index - TRAIL_LENGTH_FRAMES + 1
        )

        indices = np.arange(
            trail_start,
            frame_index + 1
        )

        indices = indices[
            self.valid_mask[indices]
        ]

        if len(indices) == 0:
            self.trail_line.set_data_3d(
                [],
                [],
                []
            )
            return

        positions = self.camera_positions[indices]

        self.trail_line.set_data_3d(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2]
        )

    def _camera_frustum_segments(
        self,
        position,
        rotation
    ):
        depth = self.camera_scale
        half_width = 0.55 * self.camera_scale
        half_height = 0.38 * self.camera_scale

        local_corners = np.array([
            [-half_width, -half_height, depth],
            [half_width, -half_height, depth],
            [half_width, half_height, depth],
            [-half_width, half_height, depth]
        ])

        world_corners = (
            position.reshape(1, 3)
            + (rotation @ local_corners.T).T
        )

        segments = []

        for corner in world_corners:
            segments.append(
                (position, corner)
            )

        for index in range(4):
            segments.append(
                (
                    world_corners[index],
                    world_corners[(index + 1) % 4]
                )
            )

        return segments

    def _hide_camera(self):
        self.current_point._offsets3d = (
            [],
            [],
            []
        )

        for line in self.frustum_lines:
            line.set_data_3d([], [], [])

        for line in self.axis_lines:
            line.set_data_3d([], [], [])

        self.optical_axis_line.set_data_3d(
            [],
            [],
            []
        )

        self.camera_text.set_text("")

    def _update_camera(self, pose_index):
        position = self.camera_positions[
            pose_index
        ]

        rotation = self.camera_rotations[
            pose_index
        ]

        self.current_point._offsets3d = (
            [position[0]],
            [position[1]],
            [position[2]]
        )

        segments = self._camera_frustum_segments(
            position,
            rotation
        )

        for line, segment in zip(
            self.frustum_lines,
            segments
        ):
            start, end = segment

            line.set_data_3d(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]]
            )

        axis_length = 1.25 * self.camera_scale

        for axis_index, line in enumerate(
            self.axis_lines
        ):
            endpoint = (
                position
                + axis_length
                * rotation[:, axis_index]
            )

            line.set_data_3d(
                [position[0], endpoint[0]],
                [position[1], endpoint[1]],
                [position[2], endpoint[2]]
            )

        optical_endpoint = (
            position
            + 2.0
            * self.camera_scale
            * rotation[:, 2]
        )

        self.optical_axis_line.set_data_3d(
            [position[0], optical_endpoint[0]],
            [position[1], optical_endpoint[1]],
            [position[2], optical_endpoint[2]]
        )

        self.camera_text.set_position(
            (position[0], position[1])
        )

        self.camera_text.set_3d_properties(
            position[2]
        )

        self.camera_text.set_text(
            f"t={pose_index / self.video_fps:.2f}s"
        )

    def update_frame(self, frame_index):
        frame_index = int(
            np.clip(
                frame_index,
                0,
                self.frame_count - 1
            )
        )

        self.current_index = frame_index

        rgb_frame = decode_rgb_frame(
            self.cached_frames[frame_index]
        )

        self.video_artist.set_data(rgb_frame)

        current_time = (
            frame_index / self.video_fps
        )

        self.video_title.set_text(
            (
                f"Left video — frame {frame_index}/"
                f"{self.frame_count - 1}, "
                f"time {current_time:.3f} s"
            )
        )

        self._update_trail(frame_index)

        if self.valid_mask[frame_index]:
            pose_index = frame_index
            status = "VALID"
        else:
            pose_index = (
                self._nearest_previous_valid_index(
                    frame_index
                )
            )
            status = "INVALID"

        if pose_index is None:
            self._hide_camera()
        else:
            self._update_camera(pose_index)

        result = self.frame_results[
            frame_index
        ]

        if np.isfinite(result["score"]):
            self.status_text.set_text(
                (
                    f"{status}  "
                    f"score={result['score']:.4f}  "
                    f"angle={result['angle_deg']:.2f}°"
                )
            )
        else:
            self.status_text.set_text(status)

        self.figure.canvas.draw_idle()

    def _on_slider_changed(self, time_value):
        frame_index = int(
            round(
                float(time_value)
                * self.video_fps
            )
        )

        self.update_frame(frame_index)

    def _toggle_play(self, event=None):
        self.playing = not self.playing

        self.play_button.label.set_text(
            "Pause" if self.playing else "Play"
        )

        self.figure.canvas.draw_idle()

    def _on_timer(self):
        if not self.playing:
            return

        next_index = self.current_index + 1

        if next_index >= self.frame_count:
            next_index = 0

        self.time_slider.set_val(
            next_index / self.video_fps
        )

    def _on_key_press(self, event):
        if event.key == " ":
            self._toggle_play()
            return

        if event.key == "right":
            next_index = min(
                self.current_index + 1,
                self.frame_count - 1
            )

            self.time_slider.set_val(
                next_index / self.video_fps
            )
            return

        if event.key == "left":
            previous_index = max(
                self.current_index - 1,
                0
            )

            self.time_slider.set_val(
                previous_index / self.video_fps
            )
            return

        if event.key in ("q", "escape"):
            plt.close(self.figure)

    def _on_close(self, event):
        self.playing = False

        try:
            self.timer.stop()
        except Exception:
            pass

    def show(self):
        print()
        print("同步查看器操作：")
        print("  Play/Pause 按钮或空格：播放/暂停")
        print("  拖动底部时间条：跳到任意时间")
        print("  左右方向键：逐帧")
        print("  3D 区域鼠标左键拖动：旋转视角")
        print("  3D 区域滚轮：缩放")
        print("  Q 或 Esc：退出")
        print()

        plt.show(block=True)


# ============================================================
# 主程序
# ============================================================
def main():
    params = ours_params

    K_left = np.asarray(
        params.K_left,
        dtype=np.float64
    )

    D_left = np.asarray(
        params.D_left,
        dtype=np.float64
    )

    K_right = np.asarray(
        params.K_right,
        dtype=np.float64
    )

    D_right = np.asarray(
        params.D_right,
        dtype=np.float64
    )

    rotation_left_to_right = orthonormalize_rotation(
        params.R
    )

    translation_left_to_right = np.asarray(
        params.T,
        dtype=np.float64
    ).reshape(3)

    print("使用相机参数：")
    print(params)

    try:
        video_path = find_video_path()
    except FileNotFoundError as error:
        print(error)
        return

    print(f"视频文件: {video_path}")

    processed = preprocess_video(
        video_path,
        K_left,
        D_left,
        K_right,
        D_right,
        rotation_left_to_right,
        translation_left_to_right
    )

    if processed is None:
        print("用户取消，程序结束")
        return

    camera_positions, camera_rotations, valid_mask = (
        build_circle_fixed_pose_sequence(
            processed["results"]
        )
    )

    valid_count = int(
        np.sum(valid_mask)
    )

    print(
        f"有效三维位姿: {valid_count}/"
        f"{len(valid_mask)}"
    )

    if valid_count == 0:
        print(
            "没有有效三维位姿，请检查椭圆跟踪、相机外参和阈值。"
        )
        return

    viewer = SynchronizedPoseViewer(
        cached_frames=processed["frames"],
        frame_results=processed["results"],
        camera_positions=camera_positions,
        camera_rotations=camera_rotations,
        valid_mask=valid_mask,
        video_fps=processed["fps"]
    )

    viewer.show()


if __name__ == "__main__":
    main()
