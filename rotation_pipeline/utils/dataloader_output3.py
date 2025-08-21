import os

def count_images_in_folder(folder_path):
    """
    统计给定文件夹中 .png 文件的数量
    """
    if not os.path.exists(folder_path):
        print(f"❌ 文件夹 {folder_path} 不存在")
        return 0

    # 获取该文件夹下所有文件，筛选出 .png 文件
    png_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
    return len(png_files)

def print_image_counts(output_base_path):
    """
    打印出四个目标文件夹中图片的数量
    """
    # 定义四个目标文件夹路径
    source_image_folder = os.path.join(output_base_path, 'source_image')
    target_image_folder = os.path.join(output_base_path, 'target_image')
    source_mask_folder = os.path.join(output_base_path, 'source_mask')
    target_mask_folder = os.path.join(output_base_path, 'target_mask')

    # 统计每个文件夹中的 .png 文件数量
    source_image_count = count_images_in_folder(source_image_folder)
    target_image_count = count_images_in_folder(target_image_folder)
    source_mask_count = count_images_in_folder(source_mask_folder)
    target_mask_count = count_images_in_folder(target_mask_folder)

    # 打印结果
    print(f"📁 source_image 文件夹中的图片数量: {source_image_count}")
    print(f"📁 target_image 文件夹中的图片数量: {target_image_count}")
    print(f"📁 source_mask 文件夹中的图片数量: {source_mask_count}")
    print(f"📁 target_mask 文件夹中的图片数量: {target_mask_count}")

if __name__ == "__main__":
    # 设置输出目录的路径
    output_base_path = "/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/blender/train_data/rotation"
    
    # 打印四个文件夹中的图片数量
    print_image_counts(output_base_path)
