import cv2
import numpy as np


def keep_black_pixels(image_path, output_path):
    # 1. 读取图像（保留Alpha通道，如果有）
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        print("错误：图像加载失败！")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 中值滤波去噪
    gray = cv2.medianBlur(gray, 3)

    # 二值化处理
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 将单通道二值图像转换为3通道RGB图像
    binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # 5. 保存结果
    cv2.imwrite(output_path, binary_rgb)
    print(f"处理完成，结果已保存至：{output_path}")


# 使用示例
keep_black_pixels("IMG_8364.JPG", "output3.png")