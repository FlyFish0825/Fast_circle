import cv2
import numpy as np
import  os   
import cv2
import numpy as np

# ========== 全局变量 ==========
points = []              # 鼠标点击点
img = None               # 原始图像
display_img = None       # 显示用图像
virtual_ellipse = None   # 手动拟合的虚拟椭圆
extracted_ellipse = None # 自动提取的真实椭圆
band_width = 25          # 初始搜索带宽度（像素）

# ========== 椭圆拟合 ==========
def fit_ellipse_from_points(pts):
    if len(pts) < 5:
        return None
    pts = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    try:
        return cv2.fitEllipse(pts)
    except:
        return None

# ========== 搜索带生成 ==========
def create_elliptical_search_band(shape, center, axes, angle, band_width):
    (cx, cy) = center
    (a, b) = axes
    mask = np.zeros(shape[:2], dtype=np.uint8)
    outer_a = int(a/2 + band_width)
    outer_b = int(b/2 + band_width)
    cv2.ellipse(mask, (int(cx), int(cy)), (outer_a, outer_b), angle, 0, 360, 255, -1)
    inner_a = max(0, int(a/2 - band_width))
    inner_b = max(0, int(b/2 - band_width))
    cv2.ellipse(mask, (int(cx), int(cy)), (inner_a, inner_b), angle, 0, 360, 0, -1)
    return mask



def enhance_weak_edges(gray):
    # CLAHE 增强局部对比度
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
    enhanced = clahe.apply(gray)
    # 高斯差（DOG）锐化边缘：模糊后相减
    blur1 = cv2.GaussianBlur(enhanced, (3,3), 0)
    blur2 = cv2.GaussianBlur(enhanced, (9,9), 0)
    dog = cv2.subtract(blur1, blur2)
    # 将 DOG 结果线性拉伸到 0-255
    dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return dog


# ========== 搜索带内椭圆提取 ==========
def extract_ellipse_in_band(gray_img, virtual_ellipse, band_width=25,
                            canny_low=30, canny_high=100, min_area=100):
    (cx, cy), (a, b), angle = virtual_ellipse
    enhanced = enhance_weak_edges(gray_img)
    mask = create_elliptical_search_band(gray_img.shape, (cx, cy), (a, b), angle, band_width)
    edges_full = cv2.Canny(enhanced, canny_low, canny_high)
    edges_local = cv2.bitwise_and(edges_full, edges_full, mask=mask)
    
    roi_x = max(0, int(cx - a/2 - band_width))
    roi_y = max(0, int(cy - b/2 - band_width))
    roi_w = min(gray_img.shape[1] - roi_x, int(a + 2*band_width))
    roi_h = min(gray_img.shape[0] - roi_y, int(b + 2*band_width))
    edges_roi = edges_local[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    contours, _ = cv2.findContours(edges_roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    best_ellipse = None
    best_score = float('inf')
    for cnt in contours:
        if len(cnt) < 5 or cv2.contourArea(cnt) < min_area:
            continue
        cnt_global = cnt + np.array([roi_x, roi_y])
        try:
            ell = cv2.fitEllipse(cnt_global)
        except:
            continue
        (cx_e, cy_e), (a_e, b_e), ang_e = ell
        d_center = np.sqrt((cx_e-cx)**2 + (cy_e-cy)**2)
        d_a = abs(a_e - a) / max(a, 1)
        d_b = abs(b_e - b) / max(b, 1)
        d_ang = min(abs(ang_e - angle), 180 - abs(ang_e - angle)) / 180.0
        score = d_center + d_a + d_b + d_ang
        if score < best_score:
            best_score = score
            best_ellipse = ell
    return best_ellipse, edges_local

# ========== 可视化函数 ==========
def redraw():
    global display_img, points, virtual_ellipse, extracted_ellipse, band_width
    display_img = img.copy()
    
    # 1. 绘制用户点击的点
    for p in points:
        cv2.circle(display_img, p, 3, (0, 255, 0), -1)
    
    # 2. 绘制虚拟椭圆（绿色实线）
    if virtual_ellipse is not None:
        (cx, cy), (a, b), angle = virtual_ellipse
        # 搜索带半透明绘制：先生成一个单独的带颜色图层，然后混合
        overlay = display_img.copy()
        # 绘制外椭圆填充（颜色用浅蓝色半透明）
        outer_a = int(a/2 + band_width)
        outer_b = int(b/2 + band_width)
        cv2.ellipse(overlay, (int(cx), int(cy)), (outer_a, outer_b), angle, 0, 360, (255, 200, 100), -1)
        # 绘制内椭圆填充（颜色与背景相同，相当于扣除内圆）
        inner_a = max(0, int(a/2 - band_width))
        inner_b = max(0, int(b/2 - band_width))
        cv2.ellipse(overlay, (int(cx), int(cy)), (inner_a, inner_b), angle, 0, 360, (0,0,0), -1)
        # 混合：alpha = 0.3
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, display_img, 1 - alpha, 0, display_img)
        # 重绘虚拟椭圆轮廓（绿色，线宽2）
        cv2.ellipse(display_img, virtual_ellipse, (0, 255, 0), 2)
        cv2.putText(display_img, f"Band={band_width}px", (int(cx)-40, int(cy)-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    
    # 3. 绘制提取的真实椭圆（红色粗线）
    if extracted_ellipse is not None:
        cv2.ellipse(display_img, extracted_ellipse, (0, 0, 255), 2)
    
    # 4. 操作提示
    cv2.putText(display_img, "Left:add pt | Enter:virtual | Space:extract | c:clear | q:quit",
                (10, display_img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

# ========== 鼠标回调 ==========
def mouse_callback(event, x, y, flags, param):
    global points, virtual_ellipse, extracted_ellipse
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        virtual_ellipse = None
        extracted_ellipse = None
        redraw()

# ========== 滑动条回调 ==========
def on_band_change(val):
    global band_width
    band_width = max(3, val)  # 至少3像素
    redraw()

# ========== 主程序 ==========
def main(image_path):
    global img, display_img, points, virtual_ellipse, extracted_ellipse, band_width
    img = cv2.imread(image_path)
    if img is None:
        print("图片加载失败")
        return
    display_img = img.copy()

    cv2.namedWindow('Ellipse Extraction', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Ellipse Extraction', 900, 700)
    cv2.setMouseCallback('Ellipse Extraction', mouse_callback)
    # 创建滑动条
    cv2.createTrackbar('Band Width', 'Ellipse Extraction', band_width, 100, on_band_change)

    print("操作说明：")
    print("  鼠标左键 - 添加轮廓点")
    print("  回车(Enter) - 用所选点拟合虚拟椭圆")
    print("  滑动条 - 调节搜索带宽度(3-100像素)")
    print("  空格键 - 在虚拟椭圆搜索带内提取真实椭圆")
    print("  c键 - 清除所有点和椭圆")
    print("  q键 - 退出")

    while True:
        cv2.imshow('Ellipse Extraction', display_img)
        key = cv2.waitKey(100) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            points.clear()
            virtual_ellipse = None
            extracted_ellipse = None
            redraw()
            print("已清除")
        elif key == 13:  # 回车
            if len(points) >= 5:
                ell = fit_ellipse_from_points(points)
                if ell:
                    virtual_ellipse = ell
                    extracted_ellipse = None
                    (cx, cy), (a, b), angle = ell
                    print(f"虚拟椭圆: 中心({cx:.1f},{cy:.1f}), 长轴{a:.1f}, 短轴{b:.1f}, 角度{angle:.1f}")
                    redraw()
                else:
                    print("拟合失败")
            else:
                print("至少需要5个点")
        elif key == ord(' '):  # 空格提取
            if virtual_ellipse is None:
                print("请先用回车生成虚拟椭圆")
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            ell, edges = extract_ellipse_in_band(enhanced, virtual_ellipse,
                                                 band_width=band_width,
                                                 canny_low=30, canny_high=100,
                                                 min_area=200)
            if ell:
                extracted_ellipse = ell
                (cx, cy), (a, b), angle = ell
                print(f"提取椭圆: 中心({cx:.1f},{cy:.1f}), 长轴{a:.1f}, 短轴{b:.1f}, 角度{angle:.1f}")
                cv2.imshow('Edges in band', edges)
            else:
                extracted_ellipse = None
                print("未找到椭圆")
            redraw()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main("images/1.png")   # 修改为你的图片路径
if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    work_space = os.path.dirname(project_dir)
    image_path = os.path.join(work_space, "images/1.png")
    main(image_path)