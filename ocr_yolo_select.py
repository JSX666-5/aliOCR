import cv2
import numpy as np


def keep_black_pixels(image_path, output_path):
    # 1. 读取图像（保留Alpha通道，如果有）
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        print("错误：图像加载失败！")
        return

    # 2. 判断是否为3通道（RGB）或4通道（RGBA）
    if img.shape[2] == 4:  # 带透明通道的PNG
        b, g, r, a = cv2.split(img)
    else:  # 普通RGB/BGR
        b, g, r = cv2.split(img)
        a = None

    # 3. 创建黑色像素的掩膜（纯黑色条件：R=G=B=0）
    black_mask = (b <= 90) & (g <= 90) & (r <= 60)

    # 4. 生成结果图像（非黑色区域设为白色）
    result = np.ones_like(img) * 255  # 全白背景
    result[black_mask] = img[black_mask]  # 保留黑色像素

    # 5. 保存结果
    cv2.imwrite(output_path, result)
    print(f"处理完成，结果已保存至：{output_path}")


# 使用示例
keep_black_pixels("IMG_8363.JPG", "output2.png")