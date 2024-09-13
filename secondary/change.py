import cv2
import os

# 输入文件夹路径和输出文件夹路径
input_folder = r"C:\Users\marsh\Desktop\0807num_data"
output_folder = r"C:\Users\marsh\Desktop\0807111"

# 确保输出文件夹存在，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    # 只处理图像文件
    if filename.endswith((".png", ".jpg", ".jpeg", ".bmp")):
        # 构建完整的文件路径
        img_path = os.path.join(input_folder, filename)

        # 加载图像
        img = cv2.imread(img_path)

        # 检查图像是否加载成功
        if img is None:
            print(f"Failed to load {filename}")
            continue

        # 将图像转换为灰度图像（黑白）
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 构建输出文件的路径
        output_path = os.path.join(output_folder, filename)

        # 打印保存路径进行调试
        print(f"Saving to: {output_path}")

        # 保存黑白图像
        cv2.imwrite(output_path, gray_img)
        print(f"Processed and saved {filename}")

print("Batch processing complete.")
