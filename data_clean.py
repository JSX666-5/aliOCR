import os
import cv2
import sys


def image_cleaner(folder_path):
    # 支持的图片格式
    supported_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

    # 获取所有图片文件并按修改时间排序
    image_files = sorted(
        [f for f in os.listdir(folder_path)
         if os.path.splitext(f)[1].lower() in supported_exts],
        key=lambda x: os.path.getmtime(os.path.join(folder_path, x)))

    # 获取当前最大序号（兼容已存在的编号文件）
    max_num = 0
    for f in os.listdir(folder_path):
        if '_' in f and f.split('_')[0].isalpha() and len(f.split('_')[0]) == 1:
            try:
                num = int(f.split('_')[1].split('.')[0])
                max_num = max(max_num, num)
            except:
                pass

    current_num = max_num + 1  # 起始序号

    # 遍历处理每个文件
    for filename in image_files:
        file_path = os.path.join(folder_path, filename)
        base, ext = os.path.splitext(filename)

        # 显示图片
        img = cv2.imread(file_path)
        if img is None:
            print(f"无法读取文件: {filename}")
            continue

        cv2.imshow('Review - Press Q to quit', img)
        key = cv2.waitKey(0)

        # 退出机制
        if key == ord('q') or key == 27:  # Q或ESC退出
            cv2.destroyAllWindows()
            sys.exit("用户终止操作")

        # 获取有效字母输入
        while True:
            cv2.destroyAllWindows()
            user_input = input(f"当前文件: {filename}\n"
                               "输入分类字母（a-z/A-Z）或 q 退出: ").lower()

            if user_input == 'q':
                sys.exit("用户终止操作")
            if len(user_input) == 1 and user_input.isalpha():
                break
            print("无效输入，请输入单个字母")

        # 生成新文件名
        new_name = f"{user_input.upper()}_{current_num}{ext}"
        new_path = os.path.join(folder_path, new_name)

        # 处理文件名冲突
        while os.path.exists(new_path):
            current_num += 1
            new_name = f"{user_input.upper()}_{current_num}{ext}"
            new_path = os.path.join(folder_path, new_name)

        # 执行重命名
        os.rename(file_path, new_path)
        print(f"已重命名: {filename} -> {new_name}")
        current_num += 1  # 递增序号

    cv2.destroyAllWindows()
    print("\n所有图片处理完成！")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法: python clean.py [图片文件夹路径]")
        sys.exit(1)

    folder = sys.argv[1]
    if not os.path.isdir(folder):
        print(f"错误：'{folder}' 不是有效目录")
        sys.exit(1)

    image_cleaner(folder)