import cv2
import numpy as np
import os


def deskew_image(image_path, output_path=None):
    """
    矫正倾斜的图片
    :param image_path: 输入图片路径
    :param output_path: 输出图片路径(可选)
    :return: 矫正后的图片
    """
    # 检查文件是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"无法找到图片文件: {image_path}")

    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片，请检查文件格式是否支持: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化处理
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # 计算包含所有白色像素的最小旋转矩形
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    # 调整角度
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # 旋转图像
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # 显示结果
    print(f"[INFO] 矫正角度: {angle:.3f}°")

    # 保存结果
    if output_path:
        cv2.imwrite(output_path, rotated)

    return rotated


def main():
    try:
        # 输入图片路径 - 修改为你的实际路径
        input_image = r"D:\aliOCR\IMG_8342.JPG"  # 使用原始字符串避免转义问题
        output_image = r"D:\aliOCR\IMG_8342_deskewed.JPG"  # 输出图片路径

        # 检查输入文件是否存在
        if not os.path.exists(input_image):
            print(f"错误: 输入文件不存在 - {input_image}")
            return

        # 执行矫正
        deskewed = deskew_image(input_image, output_image)

        # 显示结果
        cv2.imshow("Original", cv2.imread(input_image))
        cv2.imshow("Deskewed", deskewed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(f"图片矫正完成，结果已保存到: {output_image}")
    except Exception as e:
        print(f"发生错误: {str(e)}")


if __name__ == "__main__":
    main()