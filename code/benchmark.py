"""性能测试：测量椭圆检测处理一帧图像的平均耗时（独立脚本，不依赖 matplotlib）"""
import time
import os
import cv2
import numpy as np


# ==== 从 test.py 复制核心函数，避免 import test.py 触发 matplotlib ====

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


def search_along_normal(img_gray, pt, normal, search_length, template):
    h, w = img_gray.shape
    x0, y0 = pt
    nx, ny = normal

    distances = np.arange(-search_length, search_length + 1)
    sample_x = x0 + distances * nx
    sample_y = y0 + distances * ny

    valid_mask = (sample_x >= 0) & (sample_x < w - 1) & (sample_y >= 0) & (sample_y < h - 1)
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


# ==== 处理流程 ====

def run_once(img_gray, prior_center, prior_axes, prior_angle,
             num_samples=100, search_length=60, template=None, response_thresh=0):
    """执行一次完整的处理流程（不含可视化和图像加载/预处理）"""
    pts, normals = get_ellipse_points_and_normals(
        prior_center, prior_axes, prior_angle, num_samples
    )

    detected_points = []

    for pt, normal in zip(pts, normals):
        best_pt, score = search_along_normal(
            img_gray, pt, normal, search_length, template
        )
        if best_pt is not None and score > response_thresh:
            detected_points.append(best_pt)

    detected_points = np.array(detected_points, dtype=np.float32)

    if len(detected_points) >= 5:
        final_ellipse = cv2.fitEllipse(detected_points)
    else:
        final_ellipse = None

    return final_ellipse, len(detected_points)


# ==== 主函数 ====

def main():
    NUM_RUNS = 50
    WARMUP_RUNS = 5

    # 图片路径（相对项目根目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, '..', 'images', '2.png')

    # 先验参数（与 test.py 保持一致）
    prior_center = (790, 500)
    prior_axes = (750 / 2, 800 / 2)
    prior_angle = 12
    NUM_SAMPLES = 100
    SEARCH_LENGTH = 60
    TEMPLATE = np.array([-1, -2, -3, -4, -3, -2, -1, 0, 1, 2, 3, 4, 3, 2, 1])
    RESPONSE_THRESH = 0

    # --- 加载图像（不计入处理时间）---
    t0 = time.perf_counter()
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (7, 7), 0)
    t_load = time.perf_counter() - t0

    print(f"图像路径: {image_path}")
    print(f"图像尺寸: {img.shape[1]}x{img.shape[0]}  |  通道: {img.shape[2]}")
    print(f"图像加载+预处理耗时: {t_load*1000:.2f} ms")
    print(f"采样点数: {NUM_SAMPLES}  |  搜索半长: {SEARCH_LENGTH} px")
    print(f"测试次数: {NUM_RUNS} (预热 {WARMUP_RUNS} 次)")
    print("-" * 50)

    # --- 预热 ---
    for i in range(WARMUP_RUNS):
        run_once(img_gray, prior_center, prior_axes, prior_angle,
                 NUM_SAMPLES, SEARCH_LENGTH, TEMPLATE, RESPONSE_THRESH)

    # --- 正式测试 ---
    times = []
    for i in range(NUM_RUNS):
        t_start = time.perf_counter()
        ellipse, n_pts = run_once(img_gray, prior_center, prior_axes, prior_angle,
                                  NUM_SAMPLES, SEARCH_LENGTH, TEMPLATE, RESPONSE_THRESH)
        t_elapsed = time.perf_counter() - t_start
        times.append(t_elapsed)

        if (i + 1) % 10 == 0:
            print(f"  已完成 {i+1}/{NUM_RUNS} ...")

    times = np.array(times)
    times_ms = times * 1000

    # --- 统计 ---
    print(f"\n{'='*55}")
    print(f"                 处理速度统计")
    print(f"{'='*55}")
    print(f"  平均 (mean):      {np.mean(times_ms):.2f} ms")
    print(f"  中位数 (median):  {np.median(times_ms):.2f} ms")
    print(f"  最小 (min):       {np.min(times_ms):.2f} ms")
    print(f"  最大 (max):       {np.max(times_ms):.2f} ms")
    print(f"  标准差 (std):     {np.std(times_ms):.2f} ms")
    print(f"{'='*55}")
    print(f"  等效 FPS:         {1000 / np.mean(times_ms):.1f} fps")
    print(f"{'='*55}")
    print(f"\n  最终拟合椭圆: {ellipse}")
    print(f"  检测到的边缘点数: {n_pts}")


if __name__ == "__main__":
    main()
