import cv2
import numpy as np


def region_of_interest(img, vertices):
    """设置感兴趣区域"""
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def draw_lines(img, lines, color=(0, 0, 255), thickness=8):
    """在图像上绘制直线"""
    if lines is None:
        return
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def average_slope_lines(lines, img_shape):
    """根据斜率区分左右车道线并进行平均拟合"""
    left_lines = []
    right_lines = []
    
    if lines is None:
        return None, None
    
    height, width = img_shape[0], img_shape[1]
    
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        
        # 计算斜率
        if x2 - x1 == 0:
            continue
        slope = (y2 - y1) / (x2 - x1)
        
        # 过滤
        if abs(slope) < 0.4 or abs(slope) > 2.0:
            continue
        
        # 计算线段中点
        mid_x = (x1 + x2) / 2
            
        # 根据斜率和位置区分左右车道线
        if slope < 0 and mid_x < width * 0.5: 
            left_lines.append([x1, y1, x2, y2, slope])
        elif slope > 0 and mid_x > width * 0.5:  
            right_lines.append([x1, y1, x2, y2, slope])
    
    # 拟合左车道线
    left_lane = None
    if len(left_lines) > 0:
        left_lane = fit_line(left_lines, img_shape)
        print(f"检测到 {len(left_lines)} 条左车道线段")
    
    # 拟合右车道线
    right_lane = None
    if len(right_lines) > 0:
        right_lane = fit_line(right_lines, img_shape)
        print(f"检测到 {len(right_lines)} 条右车道线段")
    
    return left_lane, right_lane


def fit_line(lines, img_shape):
    """对检测到的直线进行拟合"""
    xs = []
    ys = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0:4]
        xs.extend([x1, x2])
        ys.extend([y1, y2])
    
    # 多项式拟合
    poly = np.polyfit(ys, xs, deg=1)
    
    y1 = img_shape[0]
    y2 = int(img_shape[0] * 0.6)
    
    x1 = int(poly[0] * y1 + poly[1])
    x2 = int(poly[0] * y2 + poly[1])
    
    return [x1, y1, x2, y2]


def lane_detection(image_path):
    """车道线检测主函数"""
    img = cv2.imread(image_path)
    if img is None:
        print("error")
        return
    
    height, width = img.shape[:2]
    
    # 预处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny边缘检测
    edges = cv2.Canny(blur, 40, 120)
    
    # 定义ROI梯形区域
    vertices = np.array([[
        (int(width * 0.1), height),
        (int(width * 0.42), int(height * 0.62)),
        (int(width * 0.58), int(height * 0.62)),
        (int(width * 0.9), height)
    ]], dtype=np.int32)
    
    roi_edges = region_of_interest(edges, vertices)
    
    # 霍夫变换检测直线
    lines = cv2.HoughLinesP(roi_edges, rho=1, theta=np.pi/180, 
                            threshold=25, minLineLength=50, maxLineGap=150)
    
    # 区分左右车道线并拟合
    left_lane, right_lane = average_slope_lines(lines, img.shape)
    
    result = img.copy()
    
    # 绘制车道线
    lane_lines = []
    if left_lane is not None:
        lane_lines.append(left_lane)
    if right_lane is not None:
        lane_lines.append(right_lane)
    
    draw_lines(result, lane_lines)
    
    return result


if __name__ == '__main__':
    image_path = '/Users/yechao.zhang/Desktop/code/machine-vision/img/img2.jpg'
    
    result = lane_detection(image_path)
    
    if result is not None:
        cv2.imshow('Lane Detection', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

