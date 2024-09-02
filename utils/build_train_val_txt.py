import os
import random
from pathlib import Path

# 设置包含训练图像的目录
image_dir = Path('/data/ykx/RoadScene/ir')

# 设置train.txt和val.txt文件的保存路径
train_file = Path('/data/ykx/RoadScene/meta/train.txt')
val_file = Path('/data/ykx/RoadScene/meta/val.txt')

# 确保输出文件的目录存在
train_file.parent.mkdir(parents=True, exist_ok=True)
val_file.parent.mkdir(parents=True, exist_ok=True)

# 支持的图像文件格式
supported_formats = {'.jpg', '.jpeg', '.png', '.bmp'}

# 获取所有支持格式的图像文件
image_files = [f.name for f in image_dir.glob('*') if f.suffix.lower() in supported_formats]

# 打乱图像文件列表
random.shuffle(image_files)

# 分割数据集为训练集和验证集（例如，80%训练，20%验证）
split_ratio = 0.8  # 训练集所占比例
split_index = int(len(image_files) * split_ratio)
train_samples = image_files[:split_index]
val_samples = image_files[split_index:]

# 将训练集文件名写入train.txt
with train_file.open('w') as f:
    for image_file in train_samples:
        f.write(image_file + '\n')

# 将验证集文件名写入val.txt
with val_file.open('w') as f:
    for image_file in val_samples:
        f.write(image_file + '\n')

print(f'train.txt文件已生成，包含{len(train_samples)}个图像.')
print(f'val.txt文件已生成，包含{len(val_samples)}个图像.')