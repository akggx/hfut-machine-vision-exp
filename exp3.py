import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# 模型定义

class CNN(nn.Module):
    """简单的CNN模型"""
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# 模型训练

def train_model():
    """训练MNIST模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据加载
    print("加载MNIST数据集...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, 
                                   download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, 
                                  download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # 创建模型
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # 训练
    epochs = 5
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch: {epoch+1}/{epochs} | Batch: {batch_idx+1}/{len(train_loader)} | '
                      f'Loss: {train_loss/(batch_idx+1):.4f} | Acc: {100.*correct/total:.2f}%')
    
    # 评估
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_acc = 100. * correct / total
    print(f"测试集准确率: {test_acc:.2f}%")
    
    return model

def save_model(model, model_path='mnist_model.pth'):
    """保存模型"""
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到: {model_path}")

def load_model(model_path='mnist_model.pth'):
    """加载模型"""
    if os.path.exists(model_path):
        print(f"加载已有模型: {model_path}")
        model = CNN()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    return None

# 图像预处理和数字分割

def preprocess_image(image_path):
    """预处理输入图像"""
    img = cv2.imread(image_path)
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 自适应二值化
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31, C=10
    )
    
    # 形态学处理
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return img, binary

def extract_digits(binary):
    """提取和分割数字"""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digit_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # 过滤
        if area > 100 and h > 20 and w > 5:
            digit_regions.append((x, y, w, h, binary[y:y+h, x:x+w]))
    
    digit_regions.sort(key=lambda r: r[0])
    
    return digit_regions

def preprocess_digit(digit_img):
    """
    将单个数字图像预处理为MNIST格式 (28x28)
    保持宽高比并居中对齐
    """
    h, w = digit_img.shape
    
    scale = 20.0 / max(h, w)
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))
    
    resized = cv2.resize(digit_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_offset = (28 - new_h) // 2
    x_offset = (28 - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas

# 识别和可视化
def recognize_digits(model, digit_regions, conf_threshold=0.3):
    """
    识别所有数字
    使用MNIST标准归一化参数
    """
    model.eval()
    all_predictions = [] 
    
    with torch.no_grad():
        for x, y, w, h, digit_img in digit_regions:
            processed = preprocess_digit(digit_img)
            
            tensor = torch.from_numpy(processed).float().unsqueeze(0).unsqueeze(0) / 255.0
            tensor = (tensor - 0.1307) / 0.3081 
            
            # 预测
            output = model(tensor)
            prob = torch.softmax(output, dim=1)
            digit_label = output.argmax(1).item()
            confidence = prob[0, digit_label].item()
            
            all_predictions.append((digit_label, confidence))
    
    # 过滤低置信度结果
    predictions = [(d, c) for d, c in all_predictions if c >= conf_threshold]
    
    return predictions, all_predictions

def visualize_results(img, predictions):
    """在原图中间下方显示识别结果"""
    result_img = img.copy()
    h, w = img.shape[:2]
    
    # 构建识别结果文本
    recognized_number = ''.join([str(d) for d, _ in predictions])
    text = f"Result: {recognized_number}"
    
    # 计算文本位置
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # 文本位置
    text_x = (w - text_size[0]) // 2
    text_y = h - 50  
    
    # 绘制文本背景
    padding = 10
    cv2.rectangle(result_img, 
                  (text_x - padding, text_y - text_size[1] - padding),
                  (text_x + text_size[0] + padding, text_y + padding),
                  (255, 255, 255), -1)
    
    # 绘制文本
    cv2.putText(result_img, text, (text_x, text_y), font, 
                font_scale, (0, 0, 255), thickness)
    
    return result_img

def draw_bounding_boxes(img, digit_regions, predictions):
    """在原图上绘制边界框和预测结果"""
    result = img.copy()
    for i, (x, y, w, h, _) in enumerate(digit_regions[:len(predictions)]):
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if i < len(predictions):
            label = f"{predictions[i][0]}"
            cv2.putText(result, label, (x+w+5, y+h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return result

# 主函数

def main():
    model_path = 'mnist_model.pth'
    image_path = 'img/img3.jpg'
    conf_threshold = 0.3  
    
    # 加载或训练模型
    model = load_model(model_path)
    if model is None:
        model = train_model()
        save_model(model, model_path)
    
    # 读取和预处理图像
    print("处理输入图像...")
    img, binary = preprocess_image(image_path)
    
    # 保存二值化结果
    cv2.imwrite('debug_binary.png', binary)
    
    # 提取数字
    digit_regions = extract_digits(binary)
    print(f"检测到 {len(digit_regions)} 个候选区域")
    
    # 识别数字
    predictions, all_predictions = recognize_digits(model, digit_regions, conf_threshold)
    print(f"识别出 {len(predictions)} 个有效数字 (置信度 >= {conf_threshold:.0%})")
    
    # 输出识别结果
    recognized_number = ''.join([str(d) for d, _ in predictions])
    print(f"识别结果: {recognized_number}")
    print("\n各位数字详情:")
    for i, (digit, conf) in enumerate(predictions):
        print(f"  第{i+1}位: {digit} (置信度: {conf:.2%})")
    
    # 如果有被过滤的低置信度结果
    if len(all_predictions) > len(predictions):
        print(f"\n低置信度结果 (< {conf_threshold:.0%}):")
        for i, (digit, conf) in enumerate(all_predictions):
            if conf < conf_threshold:
                print(f"  候选区域{i+1}: {digit} (置信度: {conf:.2%})")
    
    # 可视化
    result_img = visualize_results(img, predictions)
    
    # 显示边界框
    valid_regions = digit_regions[:len(predictions)]
    debug_img = draw_bounding_boxes(img, valid_regions, predictions)
    
    # 显示最终结果
    cv2.imshow('Recognition Result', result_img)
    cv2.imshow('Debug - Bounding Boxes', debug_img)
    cv2.imshow('Debug - Binary', binary)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

