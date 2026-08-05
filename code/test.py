import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


def get_ellipse_points_and_normals(center, axes, angle_deg, num_points=40):
    """
    根据椭圆参数生成离散点及对应的单位法向量
    """
    cx, cy = center
    a, b = axes # 注意：这里的 a 和 b 是半轴长
    angle_rad = np.deg2rad(angle_deg)
    
    t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    
    # 椭圆参数方程
    x = cx + a * np.cos(t) * np.cos(angle_rad) - b * np.sin(t) * np.sin(angle_rad)
    y = cy + a * np.cos(t) * np.sin(angle_rad) + b * np.sin(t) * np.cos(angle_rad)
    
    # 计算切向量 (dx/dt, dy/dt)
    dx = -a * np.sin(t) * np.cos(angle_rad) - b * np.cos(t) * np.sin(angle_rad)
    dy = -a * np.sin(t) * np.sin(angle_rad) + b * np.cos(t) * np.cos(angle_rad)
    
    # 法向量 (-dy, dx) 并归一化
    norm = np.hypot(dx, dy)
    nx = -dy / norm
    ny = dx / norm
    
    return np.column_stack((x, y)), np.column_stack((nx, ny))


def search_along_normals(img_gray, pts, normals, distances, template_kernel, valid_kernel):
    """
    同时沿着多个法线方向提取一维像素，并使用模板进行卷积寻找边缘
    """
    h, w = img_gray.shape
    
    # 同时生成所有法线上的采样点坐标
    sample_x = pts[:, 0, None] + distances[None, :] * normals[:, 0, None]
    sample_y = pts[:, 1, None] + distances[None, :] * normals[:, 1, None]
    
    # 标记超出图像边界的采样点
    valid_mask = (
        (sample_x >= 0)
        & (sample_x < w-1)
        & (sample_y >= 0)
        & (sample_y < h-1)
    ).astype(np.uint8)
    
    # 同时获取所有法线上的灰度值
    pixel_values = cv2.remap(
        img_gray,
        sample_x,
        sample_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    # 同时对所有法线执行一维卷积
    response = cv2.filter2D(
        pixel_values,
        cv2.CV_32F,
        template_kernel,
        borderType=cv2.BORDER_CONSTANT
    )
    
    # 计算每个卷积位置中有效像素的数量
    valid_count = cv2.filter2D(
        valid_mask,
        cv2.CV_32F,
        valid_kernel,
        borderType=cv2.BORDER_CONSTANT
    )
    
    template_length = template_kernel.shape[1]
    
    # 只有模板覆盖范围内所有像素都有效，该卷积结果才有效
    valid_response = valid_count >= template_length - 0.5
    
    # 寻找绝对响应最大的索引
    abs_response = np.abs(response)
    abs_response[~valid_response] = -np.inf
    
    # 判断每条法线是否存在有效卷积位置
    line_valid = np.any(valid_response, axis=1)
    
    max_idx = np.argmax(abs_response, axis=1)
    row_idx = np.arange(len(pts))
    max_response = abs_response[row_idx, max_idx]
    
    # 映射回图像坐标
    best_distance = distances[max_idx]
    best_x = pts[:, 0] + best_distance * normals[:, 0]
    best_y = pts[:, 1] + best_distance * normals[:, 1]
    
    best_points = np.column_stack((best_x, best_y))
    
    # 删除没有有效卷积区域的法线
    best_points = best_points[line_valid]
    max_response = max_response[line_valid]
    
    return best_points, max_response


def main():
    # 1. 加载真实图像
    image_path = 'images/2.png'  # 替换为你实际的图片文件名
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"无法读取图像: {image_path}，请检查路径。")
        return

    # 不要在这里 cv2.ellipse 画图！会破坏原始像素，导致算法搜到你画的线
    # 将图像转灰度并做适度的高斯平滑（平滑水下的高频噪声）
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (7, 7), 0)
    
    # 2. 设定先验/预测椭圆 (基于你的 IMU/上一帧数据)
    prior_center = (790, 500)
    # 你提供的长短轴 716.6 和 750.4 是全长，生成函数需要半轴长 (a, b)
    prior_axes = (750 / 2, 800 / 2) 
    prior_angle = 12

    
    # 3. 算法参数
    NUM_SAMPLES = 100           # 采样点数量
    SEARCH_LENGTH = 100       # 法向搜索距离 (原图分辨率较高，建议增大搜索范围以防先验误差)
    TEMPLATE = np.array(
        [-1, -2, -4, -2, -1, 0, 1, 2, 4, 2, 1],
        dtype=np.float32
    ) # 改进了阶跃模板的平滑度，提高抗悬浮物干扰能力
    RESPONSE_THRESH = 0       # 卷积响应阈值
    
    # 固定数组只计算一次
    DISTANCES = np.arange(
        -SEARCH_LENGTH,
        SEARCH_LENGTH + 1,
        dtype=np.float32
    )
    
    # cv2.filter2D执行的是相关运算，因此提前翻转模板
    TEMPLATE_KERNEL = TEMPLATE[::-1].reshape(1, -1).copy()
    
    # 用于判断卷积窗口中的像素是否全部有效
    VALID_KERNEL = np.ones(
        (1, len(TEMPLATE)),
        dtype=np.float32
    )

    start_time = time.perf_counter()
    
    # 4. 获取先验椭圆上的点和法线
    pts, normals = get_ellipse_points_and_normals(prior_center, prior_axes, prior_angle, NUM_SAMPLES)
    
    pts = pts.astype(np.float32)
    normals = normals.astype(np.float32)
    
    # 5. 同时执行所有法线的一维搜索
    detected_points, scores = search_along_normals(
        img_gray,
        pts,
        normals,
        DISTANCES,
        TEMPLATE_KERNEL,
        VALID_KERNEL
    )
    
    detected_points = detected_points[
        scores > RESPONSE_THRESH
    ]
            
    detected_points = np.array(detected_points, dtype=np.float32)
    
    end_time = time.perf_counter()
    print(f"耗时: {end_time - start_time:.4f} 秒")
    
    # 6. 拟合最终的椭圆
    if len(detected_points) >= 5:
        # 使用 OpenCV 拟合，返回 ((xc, yc), (a_full, b_full), theta)
        final_ellipse = cv2.fitEllipse(detected_points)
    else:
        final_ellipse = None
        
    # --- 可视化 ---
    plt.figure(figsize=(12, 9))
    
    # 画先验椭圆 (红色虚线) - 我们在可视化阶段画，不影响原图
    prior_ell_pts, _ = get_ellipse_points_and_normals(prior_center, prior_axes, prior_angle, 100)
    plt.plot(prior_ell_pts[:, 0], prior_ell_pts[:, 1], 'r--', label='Prior (Predicted) Ellipse')
    
    # 画搜索法线
    for i, (pt, normal) in enumerate(zip(pts, normals)):
        p1 = pt - normal * SEARCH_LENGTH
        p2 = pt + normal * SEARCH_LENGTH
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'y-', alpha=0.3)
        if i == 0:
            plt.plot([], [], 'y-', alpha=0.3, label='1D Search Ray')
            
    # 画实际检测到的边缘点
    if len(detected_points) > 0:
        plt.scatter(detected_points[:, 0], detected_points[:, 1], c='cyan', s=15, zorder=5, label='Detected Edge Points')
        
    # 画拟合结果 (绿色实线)
    if final_ellipse is not None:
        cv2.ellipse(img, final_ellipse, (0, 255, 0), 3)
        plt.plot([], [], 'g-', linewidth=2, label='Fitted Ellipse')

    # 显示结果
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Template-Constrained 1D Search on Underwater Pipe")
    plt.legend(loc='upper right')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()