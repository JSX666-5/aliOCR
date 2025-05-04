import cv2
import numpy as np
import os
from ultralytics import YOLO
import time


class DataProcessor:
    def __init__(self):
        self.yolo_model = self.load_yolo_model(
            r'C:\Users\26455\PycharmProjects\aliOCR\practice\runs\detect\train\weights\best.pt')

    @staticmethod
    def load_yolo_model(model_path: str):
        model = YOLO(model_path)
        model.eval()
        return model

    def detect_boxes(self, image: np.ndarray, confidence_threshold=0.5) -> list:
        results = self.yolo_model(image, iou=0.45, conf=0.5)
        boxes = []
        for result in results:
            for det in result.boxes:
                x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
                label_id = int(det.cls)
                label_name = self.yolo_model.names[label_id]
                conf = float(det.conf)
                if conf > confidence_threshold:
                    boxes.append({
                        "box": [x1, y1, x2, y2],
                        "label_id": label_id,
                        "label_name": label_name,
                        "confidence": conf
                    })
        return boxes

    def process_image(self, image_path: str, output_folder: str):  # 修改参数为输出文件夹
        image = cv2.imread(image_path)
        if image is None:
            print(f"错误：图像加载失败 {image_path}")
            return

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        boxes = self.detect_boxes(image)
        print(f"检测到 {len(boxes)} 个目标区域")

        for idx, det in enumerate(boxes):
            x1, y1, x2, y2 = det["box"]
            cropped = image[y1:y2, x1:x2].copy()  # 创建副本避免影响原图

            # 预处理流程
            if cropped.shape[2] == 4:
                b, g, r, a = cv2.split(cropped)
            else:
                b, g, r = cv2.split(cropped)
                a = None

            # 创建黑白掩膜
            black_mask = (b <= 90) & (g <= 90) & (r <= 60)
            processed_region = np.ones_like(cropped) * 255
            processed_region[black_mask] = cropped[black_mask]

            # 生成唯一文件名
            output_path = os.path.join(
                output_folder,
                f"{base_name}_region{idx + 1}_{det['label_name']}.png"  # 包含原文件名、区域序号和标签
            )

            # 保存处理后的区域
            cv2.imwrite(output_path, processed_region)
            print(f"已保存区域 {idx + 1} 到 {output_path}")

    def process_folder(self, input_folder: str, output_folder: str):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"创建输出文件夹: {output_folder}")

        supported_exts = ['.jpg', '.jpeg', '.png', '.bmp']
        processed_count = 0

        for filename in os.listdir(input_folder):
            if any(filename.lower().endswith(ext) for ext in supported_exts):
                input_path = os.path.join(input_folder, filename)

                start_time = time.time()
                self.process_image(input_path, output_folder)  # 传入输出文件夹
                print(f"处理 {filename} 耗时: {time.time() - start_time:.2f}s")
                processed_count += 1

        print(f"\n处理完成！共处理 {processed_count} 张图像")


# 使用示例
if __name__ == "__main__":
    processor = DataProcessor()
    input_dir = r"C:\Users\26455\PycharmProjects\aliOCR\raw_data"
    output_dir = r"C:\Users\26455\PycharmProjects\aliOCR\processed"  # 建议使用新文件夹
    processor.process_folder(input_dir, output_dir)