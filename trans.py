# import os
# from PIL import Image
# from tqdm import tqdm  # 进度条工具，可选
#
#
# def resize_images(input_folder, output_folder, target_size=(128, 128)):
#     """
#     将输入文件夹中的所有图像缩放到目标尺寸并保存到输出文件夹
#
#     参数:
#         input_folder (str): 输入图像文件夹路径（包含256x256图像）
#         output_folder (str): 输出文件夹路径（存放128x128图像）
#         target_size (tuple): 目标尺寸，默认为(128, 128)
#     """
#     # 确保输出文件夹存在
#     os.makedirs(output_folder, exist_ok=True)
#
#     # 获取输入文件夹中的所有图像文件
#     valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
#     image_files = [f for f in os.listdir(input_folder)
#                    if f.lower().endswith(valid_extensions)]
#
#     print(f"找到 {len(image_files)} 张图像需要处理...")
#
#     # 使用进度条（需要安装tqdm库：pip install tqdm）
#     for filename in tqdm(image_files, desc="处理进度"):
#         try:
#             # 打开图像
#             img_path = os.path.join(input_folder, filename)
#             img = Image.open(img_path)
#
#             # 检查原始尺寸是否为256x256（可选）
#             if img.size != (256, 256):
#                 print(f"\n警告: {filename} 的尺寸是 {img.size}，不是256x256")
#
#             # 高质量缩放（使用LANCZOS抗锯齿）
#             img_resized = img.resize(target_size, Image.LANCZOS)
#
#             # 保存图像（保持原始格式）
#             output_path = os.path.join(output_folder, filename)
#             img_resized.save(output_path)
#
#         except Exception as e:
#             print(f"\n处理 {filename} 时出错: {str(e)}")
#
#     print(f"\n所有图像已处理完成！结果保存在: {output_folder}")
#
#
# if __name__ == "__main__":
#     # 配置路径（请修改为你的实际路径）
#     folder_a = "D:\PycharmProjects\Image-Super-Resolution-via-Iterative-Refinement-master\dataset\celeba_hq_256\hr_256"  # 原始图像文件夹（256x256）
#     folder_b = "D:\PycharmProjects\Image-Super-Resolution-via-Iterative-Refinement-master\dataset\celeba_hq_256\lr_16"  # 输出文件夹（128x128）
#
#     # 执行转换
#     resize_images(folder_a, folder_b, target_size=(128, 128))

import torch
import torch
print(torch.cuda.is_available())  # 输出 False 表示 CUDA 不可用
print(torch.__version__)          # 查看版本