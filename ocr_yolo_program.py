import cv2
import torch
import numpy as np
import sys
import os
import tempfile
import json
from alibabacloud_ocr_api20210707.client import Client as ocr_api20210707Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_darabonba_stream.client import Client as StreamClient
from alibabacloud_ocr_api20210707 import models as ocr_api_20210707_models
from alibabacloud_tea_util import models as util_models
from ultralytics import YOLO


class OCRProcessor:
    def __init__(self):
        # 初始化YOLO模型
        self.yolo_model = self.load_yolo_model(r'C:\Users\Lenovo\pythonProject\aliOCR\program\runs\detect\train\weights\best.pt')

        # 初始化阿里云OCR客户端
        self.ocr_client = self.create_ocr_client()

    @staticmethod
    def load_yolo_model(model_path: str):
        """加载本地YOLOv8模型"""
        model = YOLO(model_path)
        model.eval()
        return model

    @staticmethod
    def create_ocr_client() -> ocr_api20210707Client:
        """创建阿里云OCR客户端"""
        config = open_api_models.Config(
            access_key_id="LTAI5tHWgaNmttQzWU15wjhj",
            access_key_secret="cMlumsF20GFT1HT9G1HI9vjxjaBzTW"
        )
        config.endpoint = 'ocr-api.cn-hangzhou.aliyuncs.com'
        return ocr_api20210707Client(config)

    def detect_boxes(self, image: np.ndarray, confidence_threshold=0.0) -> list:
        """YOLO目标检测，返回检测框坐标"""
        results = self.yolo_model(image)
        boxes = []
        for result in results:
            for det in result.boxes.xyxy.cpu().numpy():
                if len(det) == 6:
                    x1, y1, x2, y2, conf, cls = det
                elif len(det) == 4:
                    x1, y1, x2, y2 = det
                    conf = 1.0  # 如果没有置信度信息，可以设置为默认值
                    cls = 0  # 如果没有类别信息，可以设置为默认值
                else:
                    raise ValueError(f"Unexpected detection format: {det}")
                if conf > confidence_threshold:
                    boxes.append([int(x1), int(y1), int(x2), int(y2)])
        return boxes

    def recognize_text(self, image: np.ndarray) -> str:
        """调用阿里云OCR识别图像文字"""
        # 使用临时文件
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_path = temp_file.name
            cv2.imwrite(temp_path, image)

        try:
            body_stream = StreamClient.read_from_file_path(temp_path)
            request = ocr_api_20210707_models.RecognizeHandwritingRequest(body=body_stream)
            runtime = util_models.RuntimeOptions()
            response = self.ocr_client.recognize_handwriting_with_options(request, runtime)
            return response.body.data
        except Exception as error:
            print(f"OCR识别错误: {error.message}")
            return ""

    def process_image(self, image_path: str) -> list:
        """完整处理流程：YOLO检测 → 裁剪 → 预处理 → OCR识别"""
        # 读取原始图像（保持彩色用于后续裁剪）
        image = cv2.imread(image_path)
        if image is None:
            print("错误：图像加载失败")
            return []

        # YOLO检测
        boxes = self.detect_boxes(image)
        print(f"检测到 {len(boxes)} 个目标区域")

        # 创建一个列表来存储每个框的信息（包括位置和识别结果）
        box_results = []

        # 遍历每个检测框
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            # 从原始彩色图像裁剪区域（保持最大信息量）
            cropped = image[y1:y2, x1:x2]

            # 在裁剪后执行改进的预处理流程
            # 1. 转换为灰度图
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

            # 2. 自适应直方图均衡化（优化参数）
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))  # 降低对比度增强幅度
            enhanced = clahe.apply(gray)

            # 3. 非局部均值去噪（优化参数保留细节）
            denoised = cv2.fastNlMeansDenoising(enhanced, h=10, templateWindowSize=5, searchWindowSize=15)

            # 4. 锐化处理（新增步骤）
            kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharpened = cv2.filter2D(denoised, -1, kernel_sharpen)

            # 5. 自适应阈值二值化（调整参数）
            binary = cv2.adaptiveThreshold(sharpened, 255,
                                           cv2.ADAPTIVE_THRESH_MEAN_C,  # 改用均值法
                                           cv2.THRESH_BINARY, 21, 5)  # 增大块大小

            # 6. 形态学操作（优化核尺寸）
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 椭圆核更符合文字形状
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # 闭运算填充小孔

            # 7. 分辨率提升（可选步骤）
            cleaned = cv2.resize(cleaned, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

            # OCR识别预处理后的图像
            text = self.recognize_text(cleaned)
            content = ""
            if text:
                try:
                    data = json.loads(text)
                    content = data.get("content", "")
                except json.JSONDecodeError as e:
                    print(f"解析 JSON 错误: {e}")

            # 存储框的中心坐标和识别内容
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            box_results.append({
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'center_x': center_x,
                'center_y': center_y,
                'content': content
            })

        # 按从上到下、从左到右排序
        # 首先按垂直位置（y坐标）排序，然后按水平位置（x坐标）排序
        box_results.sort(key=lambda box: (box['center_y'], box['center_x']))

        # 提取排序后的内容
        recognized_texts = [box['content'] for box in box_results]

        return recognized_texts


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("请指定图像路径: python script.py <image_path>")
        sys.exit(1)

    processor = OCRProcessor()
    recognized_texts = processor.process_image(sys.argv[1])
    print("最终识别结果:", recognized_texts)

    # 预期结果
    expected_results = ['50', '3', '357911', '1', '1361015', 'default42']

    # 检查每个识别结果是否包含对应预期结果的所有字符
    for i in range(min(len(recognized_texts), len(expected_results))):
        recognized_text = recognized_texts[i] if i < len(recognized_texts) else ""
        expected_result = expected_results[i]


        # 改进的字符串清理逻辑
        def clean_text(text):
            # 移除所有非字母数字字符（保留数字和字母）
            cleaned = ''.join(c for c in text if c.isalnum())
            return cleaned.lower()


        cleaned_recognized = clean_text(recognized_text)
        cleaned_expected = clean_text(expected_result)

        # 检查预期结果是否为识别结果的子串
        if cleaned_expected in cleaned_recognized:
            print(f"识别结果 '{recognized_text.strip()}' 包含预期结果 '{expected_result}'：正确")
        else:
            print(f"识别结果 '{recognized_text.strip()}' 不包含预期结果 '{expected_result}'：错误")