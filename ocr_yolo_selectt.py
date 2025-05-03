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
import matplotlib.pyplot as plt
import time


class OCRProcessor:
    def __init__(self):
        # 初始化YOLO模型
        self.yolo_model = self.load_yolo_model(r'D:\aliOCR\practice\runs\detect\train\weights\best.pt')

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
            # access_key_id="LTAI5tHWgaNmttQzWU15wjhj",
            access_key_id="LTAI5tRGS9kT7xxKhbsJkqNu",
            # access_key_secret="cMlumsF20GFT1HT9G1HI9vjxjaBzTW"
            access_key_secret="XtstpTk3uWTIliwhCWFkmDEygwTqLm"
        )
        config.endpoint = 'ocr-api.cn-hangzhou.aliyuncs.com'
        return ocr_api20210707Client(config)

    def detect_boxes(self, image: np.ndarray, confidence_threshold=0.5) -> list:
        """YOLO目标检测，返回检测框坐标"""
        results = self.yolo_model(image, iou=0.45, conf=0.5)
        boxes = []
        for result in results:
            for det in result.boxes:
                # 获取坐标、标签ID和置信度
                x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
                label_id = int(det.cls)
                label_name = self.yolo_model.names[label_id]  # 获取标签名
                conf = float(det.conf)
                if conf > confidence_threshold:
                    boxes.append({
                        "box": [x1, y1, x2, y2],
                        "label_id": label_id,
                        "label_name": label_name,
                        "confidence": conf
                    })
        return boxes

    def recognize_text(self, image: np.ndarray) -> str:
        """调用阿里云OCR识别图像文字"""
        # 使用临时文件
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_path = temp_file.name
            cv2.imwrite(temp_path, image)

        try:
            body_stream = StreamClient.read_from_file_path(temp_path)
            request = ocr_api_20210707_models.RecognizeAllTextRequest(body=body_stream, type='Advanced')
            runtime = util_models.RuntimeOptions()
            response = self.ocr_client.recognize_all_text_with_options(request, runtime)
            return response.body.data.content
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

        sorted_detections = sorted(boxes, key=lambda x: x["label_name"])

        # 创建一个列表来存储每个框的信息（包括位置和识别结果）
        box_results = []

        # 遍历每个检测框
        for i, det in enumerate(sorted_detections):
            x1, y1, x2, y2 = det["box"]
            label_name = det["label_name"]
            print(f"\n处理第 {i + 1} 个区域...")

            # 从原始彩色图像裁剪区域（保持最大信息量）
            cropped = image[y1:y2, x1:x2]

            # 2. 判断是否为3通道（RGB）或4通道（RGBA）
            if cropped.shape[2] == 4:  # 带透明通道的PNG
                b, g, r, a = cv2.split(cropped)
            else:  # 普通RGB/BGR
                b, g, r = cv2.split(cropped)
                a = None

            # 3. 创建黑色像素的掩膜（纯黑色条件：R=G=B=0）
            black_mask = (b <= 90) & (g <= 90) & (r <= 60)

            # 4. 生成结果图像（非黑色区域设为白色）
            result = np.ones_like(cropped) * 255  # 全白背景
            result[black_mask] = cropped[black_mask]  # 保留黑色像素

            # 2. 转为灰度图并二值化
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

            # 3. 查找轮廓（只检测外部轮廓）
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 4. 遍历轮廓，找到矩形框
            for cnt in contours:
                # 计算轮廓的矩形边界
                x, y, w, h = cv2.boundingRect(cnt)
                print(f"区域 {i + 1} 的轮廓尺寸：w={w}, h={h}")  # 输出所有轮廓尺寸

                # 判断是否为矩形框（长宽比例和面积筛选）
                if w > 100 and h > 70:  # 近似正方形
                    # 用白色覆盖矩形框（线宽约为3像素）
                    cv2.rectangle(result, (x, y), (x + w, y + h), (255, 255, 255), 20)
                    print(f"区域 {i + 1} 已覆盖")
                else:
                    print(f"区域 {i + 1} 未覆盖")

            # OCR识别预处理后的图像
            text = self.recognize_text(result)
            content = ""
            if text:
                try:
                    data = text
                    content = data
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
                'label_name': int(label_name),
                'center_x': center_x,
                'center_y': center_y,
                'content': content
            })

        sorted_boxes = sorted(box_results, key=lambda box: box['label_name'])

        # 提取排序后的内容
        recognized_texts = [str(box['label_name'])+box['content'] for box in sorted_boxes]

        return recognized_texts


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("请指定图像路径: python script.py <image_path>")
        sys.exit(1)

    processor = OCRProcessor()
    recognized_texts = processor.process_image(sys.argv[1])
    print("最终识别结果:", recognized_texts)
