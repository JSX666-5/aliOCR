import cv2
import os

# 输入图片路径（修改为你的图片名）
input_image = 'IMG_8377.JPG'

# 输出目录和路径
output_dir = 'processed'
output_image = os.path.join(output_dir, 'gray_' + os.path.basename(input_image))

# 检查并创建processed目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"创建目录: {output_dir}")

try:
    # 读取图片
    image = cv2.imread(input_image)
    if image is None:
        raise FileNotFoundError(f"无法加载图片，请检查路径: {input_image}")

    # 转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 保存结果
    cv2.imwrite(output_image, gray_image)
    print(f"灰度图片已保存到: {output_image}")

    # 显示对比（可选）
    cv2.imshow('Original', image)
    cv2.imshow('Gray Image', gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print(f"处理出错: {str(e)}")