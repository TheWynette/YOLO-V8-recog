import os
from PIL import Image, ImageEnhance
import random

def augment_image(image_path, output_dir, angles=[15, 30, 45, 60, 75, 90]):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image = Image.open(image_path)
    base_name = os.path.basename(image_path).split('.')[0]
    
    for angle in angles:
        # 旋转图像
        rotated_image = image.rotate(angle)
        rotated_image.save(os.path.join(output_dir, f"{base_name}_rot_{angle}.jpg"))
        
        # 其他增强操作（例如调整亮度、对比度、颜色和锐度）
        enhancer = ImageEnhance.Brightness(rotated_image)
        bright_image = enhancer.enhance(random.uniform(0.8, 1.2))
        bright_image.save(os.path.join(output_dir, f"{base_name}_rot_{angle}_bright.jpg"))
        
        enhancer = ImageEnhance.Contrast(rotated_image)
        contrast_image = enhancer.enhance(random.uniform(0.8, 1.2))
        contrast_image.save(os.path.join(output_dir, f"{base_name}_rot_{angle}_contrast.jpg"))
        
        enhancer = ImageEnhance.Color(rotated_image)
        color_image = enhancer.enhance(random.uniform(0.8, 1.2))
        color_image.save(os.path.join(output_dir, f"{base_name}_rot_{angle}_color.jpg"))
        
        enhancer = ImageEnhance.Sharpness(rotated_image)
        sharp_image = enhancer.enhance(random.uniform(0.8, 1.2))
        sharp_image.save(os.path.join(output_dir, f"{base_name}_rot_{angle}_sharp.jpg"))

def augment_dataset(input_dir, output_dir, angles=[15, 30, 45, 60, 75, 90]):
    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            augment_image(image_path, output_dir, angles)

# 使用示例
input_directory = '/path/to/your/dataset/images/train'
output_directory = '/path/to/your/dataset/images/train_augmented'

augment_dataset(input_directory, output_directory)
