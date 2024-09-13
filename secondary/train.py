from ultralytics import YOLO


# 加载YOLOv8模型，选择一个预训练的模型作为起点
model = YOLO(
    r"C:\Users\marsh\Desktop\recognition\yolo\weight_data\vision1_only_number.pt"
)

# 配置训练参数
model.train(
    data=r"C:\Users\marsh\Desktop\recognition\yolo\0807yolo_num\data.yaml",  # 数据集配置文件路径
    batch=30,  # 批次大小
    epochs=180,  # 训练轮数
    imgsz=640,  # 输入图片尺寸
)
