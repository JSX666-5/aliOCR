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
        self.yolo_model = self.load_yolo_model(r"C:\Users\Lenovo\pythonProject\aliOCR\xuehao\runs\detect\train\weights\best.pt")

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

    def detect_boxes(self, image: np.ndarray, confidence_threshold=0.5) -> list:
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
        """完整处理流程：YOLO检测 → 裁剪 → OCR识别"""
        image = cv2.imread(image_path)

        if image is None:
            print("错误：图像加载失败")
            return []

        # 转换为灰度图（单通道）
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 中值滤波去噪
        gray = cv2.medianBlur(gray, 3)

        # 二值化处理
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # 将单通道二值图像转换为3通道RGB图像
        binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        # YOLO检测
        boxes = self.detect_boxes(binary_rgb)
        print(f"检测到 {len(boxes)} 个目标区域")

        # 存储识别结果
        recognized_texts = []

        # 遍历每个检测框
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            # 裁剪目标区域
            cropped = gray[y1:y2, x1:x2]

            # OCR识别
            text = self.recognize_text(cropped)
            if text:
                try:
                    # 解析返回的 JSON 数据
                    data = json.loads(text)
                    # 提取 content 字段的内容
                    content = data.get("content", "")
                    recognized_texts.append(content)

                except json.JSONDecodeError as e:
                    print(f"解析 JSON 错误: {e}")
                    recognized_texts.append("")  # 如果解析失败，添加空字符串

        return recognized_texts


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("请指定图像路径: python script.py <image_path>")
        sys.exit(1)

    processor = OCRProcessor()
    recognized_texts = processor.process_image(sys.argv[1])
    print("最终识别结果:", recognized_texts)