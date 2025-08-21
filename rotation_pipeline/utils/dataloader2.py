import os
import shutil
import json
from PIL import Image

def process_pairs(dataset_pairs_file, input_base_path, output_base_path):
    """
    处理 dataset_pairs.json 中的每个配对，将图片和掩码文件复制到新的文件夹，并修改文件名。
    """
    # 读取 dataset_pairs.json 文件
    with open(dataset_pairs_file, 'r') as f:
        dataset_pairs = json.load(f)

    # 设置输出目录结构
    output_source_image_dir = os.path.join(output_base_path, 'source_image')
    output_target_image_dir = os.path.join(output_base_path, 'target_image')
    output_source_mask_dir = os.path.join(output_base_path, 'source_mask')
    output_target_mask_dir = os.path.join(output_base_path, 'target_mask')

    # 如果目标文件夹不存在，创建它们
    os.makedirs(output_source_image_dir, exist_ok=True)
    os.makedirs(output_target_image_dir, exist_ok=True)
    os.makedirs(output_source_mask_dir, exist_ok=True)
    os.makedirs(output_target_mask_dir, exist_ok=True)

    # 遍历 dataset_pairs.json 中的每个元素
    for pair in dataset_pairs:
        camera_name = pair['camera']
        source_image = pair['source_image']
        target_image = pair['target_image']
        
        # 构建文件路径
        source_image_path = os.path.join(input_base_path, 'images', source_image)
        target_image_path = os.path.join(input_base_path, 'images', target_image)
        
        # 提取对应的 mask 文件路径
        source_mask_path = os.path.join(input_base_path, 'test_masks_cryptomatte2', source_image)
        target_mask_path = os.path.join(input_base_path, 'test_masks_cryptomatte2', target_image)

        # 检查文件是否存在
        if not os.path.exists(source_image_path) or not os.path.exists(target_image_path):
            print(f"❌ 错误：源图像或目标图像不存在: {source_image_path}, {target_image_path}")
            continue
        if not os.path.exists(source_mask_path) or not os.path.exists(target_mask_path):
            print(f"❌ 错误：源掩码或目标掩码不存在: {source_mask_path}, {target_mask_path}")
            continue

        # 生成新的文件名
        new_file_name = f"{source_image.replace('.png', '')}_{target_image.replace('.png', '')}.png"

        # 定义新的文件路径
        new_source_image_path = os.path.join(output_source_image_dir, new_file_name)
        new_target_image_path = os.path.join(output_target_image_dir, new_file_name)
        new_source_mask_path = os.path.join(output_source_mask_dir, new_file_name)
        new_target_mask_path = os.path.join(output_target_mask_dir, new_file_name)

        # 复制源图像、目标图像、源掩码和目标掩码到目标文件夹，并修改文件名
        try:
            shutil.copy(source_image_path, new_source_image_path)
            shutil.copy(target_image_path, new_target_image_path)
            shutil.copy(source_mask_path, new_source_mask_path)
            shutil.copy(target_mask_path, new_target_mask_path)

            # 打印处理成功信息
            print(f"✅ 已处理配对: {source_image} 和 {target_image}")

        except Exception as e:
            print(f"❌ 处理配对 {source_image} 和 {target_image} 时出错: {e}")


def process_all_folders(base_input_path, output_base_path):
    """
    处理所有的子文件夹，遍历 base_input_path 下的所有目标子文件夹。
    """
    # 获取 base_input_path 下的所有子文件夹
    # subfolders = [
    #     "output_indoor5_sculpture1", "output_indoor8_sculpture1", "output_indoor9_sculpture1", 
    #     "output_outdoor1_sculpture1", "output_outdoor5_sculpture1", "output_outdoor6_sculpture1", 
    #     "output_outdoor8_sculpture1", "output_outdoor9_sculpture1"
    # ]

    subfolders = [
        "output_indoor8_sculpture1", "output_indoor9_sculpture1", 
        "output_outdoor1_sculpture1", "output_outdoor5_sculpture1", "output_outdoor6_sculpture1", 
        "output_outdoor8_sculpture1", "output_outdoor9_sculpture1"
    ]


    # 遍历每个子文件夹
    for subfolder in subfolders:
        print(f"🔄 处理子文件夹: {subfolder}")

        # 构建每个子文件夹的 dataset_pairs.json 路径
        dataset_pairs_file = os.path.join(base_input_path, "sculpture1", subfolder, 'dataset_pairs.json')

        # 处理每个子文件夹中的数据对
        if os.path.exists(dataset_pairs_file):
            process_pairs(dataset_pairs_file, os.path.join(base_input_path, "sculpture1", subfolder), output_base_path)
        else:
            print(f"❌ 未找到 {dataset_pairs_file} 文件，跳过该文件夹。")


if __name__ == "__main__":
    # 定义输入的基本路径
    base_input_path = "/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs"
    
    # 定义输出的基本路径
    output_base_path = "/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/blender/train_data/rotation"
    
    # 执行批量处理函数
    process_all_folders(base_input_path, output_base_path)
