import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

output_dir = 'res1'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

img = cv2.imread('img/img1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 手动实现卷积操作
def manual_convolution(image, kernel):
    """手动实现卷积操作"""
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    # 边界填充
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    output = np.zeros_like(image, dtype=np.float64)
    
    # 执行卷积
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)
    
    return output

# 手动实现 Sobel 边缘检测
def sobel_edge_detection(image):
    """使用Sobel算子进行边缘检测"""
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float64)
    
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=np.float64)
    
    grad_x = manual_convolution(image, sobel_x)
    grad_y = manual_convolution(image, sobel_y)
    
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    gradient_magnitude = np.clip(gradient_magnitude, 0, 255).astype(np.uint8)
    
    return gradient_magnitude

print("正在进行Sobel边缘检测...")
sobel_result = sobel_edge_detection(gray)
cv2.imwrite(os.path.join(output_dir, 'img1_sobel.jpg'), sobel_result)

# 给定卷积核滤波
custom_kernel = np.array([[1, 0, -1],
                          [2, 0, -2],
                          [1, 0, -1]], dtype=np.float64)

print("正在使用给定卷积核进行滤波...")
custom_result = manual_convolution(gray, custom_kernel)
custom_result = np.abs(custom_result)
custom_result = (custom_result / custom_result.max() * 255).astype(np.uint8)
cv2.imwrite(os.path.join(output_dir, 'img1_custom_kernel.jpg'), custom_result)

# 手动实现颜色直方图
def calculate_histogram(image, bins=256):
    """手动计算颜色直方图"""
    hist = np.zeros(bins, dtype=np.int32)
    
    for pixel_value in image.flatten():
        hist[pixel_value] += 1
    
    return hist

def plot_color_histogram(image):
    """绘制彩色图像的RGB直方图"""
    colors = ('b', 'g', 'r')
    channel_names = ('Blue', 'Green', 'Red')
    
    plt.figure(figsize=(12, 4))
    
    for i, (color, name) in enumerate(zip(colors, channel_names)):
        channel = image[:, :, i]
        hist = calculate_histogram(channel)
        
        plt.subplot(1, 3, i+1)
        plt.plot(hist, color=color)
        plt.title(f'{name} Channel Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.xlim([0, 256])
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'img1_histogram.png'), dpi=150)
    plt.close()

print("正在计算颜色直方图...")
plot_color_histogram(img)

# 手动实现纹理特征提取（GLCM灰度共生矩阵）
def compute_glcm(image, distance=1, angle=0, levels=256):
    """
    计算灰度共生矩阵
    distance: 像素对之间的距离
    angle: 方向（0度）
    levels: 灰度级数
    """
    h, w = image.shape
    
    if levels < 256:
        image = (image / 256 * levels).astype(np.int32)
    
    glcm = np.zeros((levels, levels), dtype=np.float64)
    
    # 根据角度确定偏移量
    if angle == 0: 
        dx, dy = distance, 0
    elif angle == 45:
        dx, dy = distance, distance
    elif angle == 90:  
        dx, dy = 0, distance
    else: 
        dx, dy = -distance, distance
    
    # 计算共生矩阵
    for i in range(h):
        for j in range(w):
            if 0 <= i + dy < h and 0 <= j + dx < w:
                ref_pixel = image[i, j]
                neighbor_pixel = image[i + dy, j + dx]
                glcm[ref_pixel, neighbor_pixel] += 1
    
    # 归一化
    glcm = glcm / (glcm.sum() + 1e-10)
    
    return glcm

def extract_glcm_features(glcm):
    """从GLCM中提取统计特征"""
    levels = glcm.shape[0]
    i, j = np.meshgrid(range(levels), range(levels), indexing='ij')
    
    # 对比度
    contrast = np.sum(glcm * (i - j) ** 2)
    
    # 能量
    energy = np.sum(glcm ** 2)
    
    # 熵
    entropy = -np.sum(glcm * np.log(glcm + 1e-10))
    
    # 同质性
    homogeneity = np.sum(glcm / (1 + (i - j) ** 2))
    
    # 相关性
    mean_i = np.sum(i * glcm)
    mean_j = np.sum(j * glcm)
    std_i = np.sqrt(np.sum(glcm * (i - mean_i) ** 2))
    std_j = np.sqrt(np.sum(glcm * (j - mean_j) ** 2))
    correlation = np.sum(glcm * (i - mean_i) * (j - mean_j)) / (std_i * std_j + 1e-10)
    
    return {
        'contrast': contrast,
        'energy': energy,
        'entropy': entropy,
        'homogeneity': homogeneity,
        'correlation': correlation
    }

print("正在提取纹理特征...")

# 压缩灰度级以提高计算效率
levels = 32
texture_features = {}

# 计算4个方向的GLCM特征
angles = [0, 45, 90, 135]
for angle in angles:
    glcm = compute_glcm(gray, distance=1, angle=angle, levels=levels)
    features = extract_glcm_features(glcm)
    for key, value in features.items():
        if key not in texture_features:
            texture_features[key] = []
        texture_features[key].append(value)

# 对4个方向的特征取平均
averaged_features = {key: np.mean(values) for key, values in texture_features.items()}

# 构建特征向量
feature_vector = np.array([
    averaged_features['contrast'],
    averaged_features['energy'],
    averaged_features['entropy'],
    averaged_features['homogeneity'],
    averaged_features['correlation']
])

# 保存纹理特征
np.save(os.path.join(output_dir, 'img1_texture_features.npy'), feature_vector)
print(f"纹理特征已保存到 {output_dir}/img1_texture_features.npy")
print(f"纹理特征: Contrast={averaged_features['contrast']:.4f}, "
      f"Energy={averaged_features['energy']:.4f}, "
      f"Entropy={averaged_features['entropy']:.4f}, "
      f"Homogeneity={averaged_features['homogeneity']:.4f}, "
      f"Correlation={averaged_features['correlation']:.4f}")

# 创建结果可视化 
plt.figure(figsize=(15, 10))

# 原始图像
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Sobel边缘检测结果
plt.subplot(2, 3, 2)
plt.imshow(sobel_result, cmap='gray')
plt.title('Sobel Edge Detection')
plt.axis('off')

# 给定卷积核滤波结果
plt.subplot(2, 3, 3)
plt.imshow(custom_result, cmap='gray')
plt.title('Custom Kernel Filtering')
plt.axis('off')

# RGB直方
plt.subplot(2, 3, 4)
for i, color in enumerate(['b', 'g', 'r']):
    hist = calculate_histogram(img[:, :, i])
    plt.plot(hist, color=color, alpha=0.7)
plt.title('RGB Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.xlim([0, 256])
plt.grid(alpha=0.3)

# 纹理特征可视化
plt.subplot(2, 3, 5)
feature_names = ['Contrast', 'Energy', 'Entropy', 'Homogeneity', 'Correlation']
feature_values = [averaged_features[key] for key in ['contrast', 'energy', 'entropy', 'homogeneity', 'correlation']]
bars = plt.bar(range(len(feature_names)), feature_values, color='steelblue')
plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
plt.title('Texture Features (GLCM)')
plt.ylabel('Value')
plt.grid(axis='y', alpha=0.3)

# 灰度图
plt.subplot(2, 3, 6)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'result_visualization.png'), dpi=150, bbox_inches='tight')
plt.close()


