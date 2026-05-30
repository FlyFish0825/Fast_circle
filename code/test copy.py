import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def search_along_normal(img_gray, pt, normal, search_length, template):
    h, w = img_gray.shape
    x0, y0 = pt
    nx, ny = normal
    distances = np.arange(-search_length, search_length + 1)
    sample_x = x0 + distances * nx
    sample_y = y0 + distances * ny
    valid_mask = (sample_x >= 0) & (sample_x < w-1) & (sample_y >= 0) & (sample_y < h-1)
    if not np.any(valid_mask):
        return None, 0, None, None, None, None
    v_x, v_y = sample_x[valid_mask], sample_y[valid_mask]
    valid_dist = distances[valid_mask]
    pixel_values = img_gray[v_y.astype(int), v_x.astype(int)].astype(float)
    if len(pixel_values) < len(template):
        return None, 0, None, None, None, None
    response = np.convolve(pixel_values, template, mode='valid')
    resp_offsets = np.array([valid_dist[i + len(template)//2] for i in range(len(response))])
    max_idx = np.argmax(np.abs(response))
    max_resp = np.abs(response[max_idx])
    offset_idx = max_idx + len(template)//2
    best_x = v_x[offset_idx]
    best_y = v_y[offset_idx]
    # 返回原始灰度剖面和对应的物理偏移
    return (best_x, best_y), max_resp, response, resp_offsets, pixel_values, valid_dist

def main():
    image_path = 'images/2.png'  # 修改为你的图片
    img = cv2.imread(image_path)
    if img is None:
        print("图像未找到")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7,7), 0)

    prior_center = (790, 500)
    prior_axes = (750/2, 800/2)   # 半轴长
    prior_angle = 12

    NUM_SAMPLES = 100
    SEARCH_LENGTH = 100
    TEMPLATE = np.array([-1, -1, -1, -1, -1, 0, 1, 1, 1, 1, 1])
    RESPONSE_THRESH = 0.5

    pts, normals = get_ellipse_points_and_normals(prior_center, prior_axes, prior_angle, NUM_SAMPLES)

    detected_pts_all = []
    responses_dict = {}          # 存储每个采样点的 (resp_seq, offsets, best_pt, score)
    profiles_dict = {}           # 存储 (pixel_values, valid_dist, offset_of_best)

    for idx, (pt, normal) in enumerate(zip(pts, normals)):
        best_pt, score, resp_seq, offsets, pix_vals, valid_dist = \
            search_along_normal(gray, pt, normal, SEARCH_LENGTH, TEMPLATE)
        if best_pt is not None and score > RESPONSE_THRESH:
            detected_pts_all.append(best_pt)
        # 保存响应曲线数据
        responses_dict[idx] = (resp_seq, offsets, best_pt, score)
        # 保存灰度剖面数据（不管是否检测到点，都保存，方便绘图）
        if pix_vals is not None:
            # 计算最佳点对应的偏移（如果有）
            offset_best = None
            if best_pt is not None:
                vec = np.array(best_pt) - pt
                offset_best = np.dot(vec, normal)
            profiles_dict[idx] = (pix_vals, valid_dist, offset_best)
        else:
            profiles_dict[idx] = None

    detected_pts_all = np.array(detected_pts_all, dtype=np.float32) if detected_pts_all else np.empty((0,2), dtype=np.float32)

    # 拟合椭圆
    final_ellipse = None
    if len(detected_pts_all) >= 5:
        pts4fit = detected_pts_all.reshape(-1, 1, 2)
        final_ellipse = cv2.fitEllipse(pts4fit)

    # ========== 最终结果图 ==========
    plt.figure(figsize=(12,9))
    prior_ell_pts, _ = get_ellipse_points_and_normals(prior_center, prior_axes, prior_angle, 100)
    plt.plot(prior_ell_pts[:,0], prior_ell_pts[:,1], 'r--', label='Prior ellipse')
    for i, (pt, normal) in enumerate(zip(pts, normals)):
        if i % 10 != 0:
            continue
        p1 = pt - normal * SEARCH_LENGTH
        p2 = pt + normal * SEARCH_LENGTH
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'y-', alpha=0.3)
    if len(detected_pts_all) > 0:
        plt.scatter(detected_pts_all[:,0], detected_pts_all[:,1], c='cyan', s=15, zorder=5, label='Detected points')
    if final_ellipse is not None:
        cv2.ellipse(img, final_ellipse, (0,255,0), 3)
        plt.plot([], [], 'g-', linewidth=2, label='Fitted ellipse')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.legend()
    plt.axis('off')
    plt.title("Edge detection along normals")
    plt.show()

    # ========== 响应曲线窗口（原有） ==========
    step = max(1, NUM_SAMPLES // 12)
    fig, axes = plt.subplots(3, 4, figsize=(18, 10))
    axes = axes.flatten()
    for plot_i, sample_i in enumerate(range(0, NUM_SAMPLES, step)):
        if plot_i >= len(axes):
            break
        ax = axes[plot_i]
        resp_seq, offsets, best_pt, score = responses_dict[sample_i]
        if resp_seq is None:
            ax.set_title(f"Sample {sample_i}: no data")
            continue
        ax.plot(offsets, resp_seq, 'b-')
        ax.set_title(f"Sample {sample_i} (response)")
        ax.set_xlabel("Offset (pix)")
        ax.set_ylabel("Conv response")
        if best_pt is not None:
            vec = np.array(best_pt) - pts[sample_i]
            offset_best = np.dot(vec, normals[sample_i])
            idx_closest = np.argmin(np.abs(offsets - offset_best))
            ax.plot(offsets[idx_closest], resp_seq[idx_closest], 'ro', markersize=8, label='selected')
            ax.legend()
        ax.grid(True, alpha=0.3)
    for idx in range(plot_i+1, len(axes)):
        axes[idx].set_visible(False)
    plt.suptitle("1D convolution response along normals")
    plt.tight_layout()
    plt.show()

    # ========== 新增：绘制“偏离很多”的灰度剖面 ==========
    # 计算每个检测点的偏移量
    offsets_all = []
    for idx in range(NUM_SAMPLES):
        best_pt = responses_dict[idx][2]  # best_pt
        if best_pt is not None:
            vec = np.array(best_pt) - pts[idx]
            offset = np.dot(vec, normals[idx])
        else:
            offset = 0.0
        offsets_all.append(offset)
    offsets_all = np.array(offsets_all)

    # 设定偏离阈值（像素），可根据实际情况调整
    DEVIATION_THRESH = 30  
    outlier_indices = np.where(np.abs(offsets_all) > DEVIATION_THRESH)[0]

    if len(outlier_indices) > 0:
        # 选择最多6个偏离最严重的点进行显示
        selected = sorted(outlier_indices, key=lambda i: abs(offsets_all[i]), reverse=True)[:6]
        num_plots = len(selected)
        cols = min(3, num_plots)
        rows = int(np.ceil(num_plots / cols))
        fig2, axes2 = plt.subplots(rows, cols, figsize=(15, 4*rows))
        if num_plots == 1:
            axes2 = [axes2]
        else:
            axes2 = axes2.flatten()

        for i, idx in enumerate(selected):
            ax = axes2[i]
            if profiles_dict[idx] is None:
                ax.set_title(f"Sample {idx}: no profile")
                continue
            pix_vals, valid_dist, offset_best = profiles_dict[idx]
            # 绘制灰度值随偏移的变化
            ax.plot(valid_dist, pix_vals, 'k-', label='Gray level')
            ax.set_xlabel("Offset along normal (pix)")
            ax.set_ylabel("Pixel intensity")
            ax.set_title(f"Sample {idx} (offset = {offsets_all[idx]:.1f} px)")
            # 标记检测到的边缘点位置
            if offset_best is not None:
                y_min, y_max = ax.get_ylim()
                ax.vlines(offset_best, y_min, y_max, color='r', linestyle='--', label='Detected edge')
                ax.legend()
            ax.grid(True, alpha=0.3)
        # 隐藏多余的子图
        for i in range(num_plots, len(axes2)):
            axes2[i].set_visible(False)
        plt.suptitle("Gray level profiles of normals with large deviations", fontsize=14)
        plt.tight_layout()
        plt.show()
    else:
        print("所有检测点的偏移都在阈值内，没有大幅度偏离点。")

if __name__ == "__main__":
    main()