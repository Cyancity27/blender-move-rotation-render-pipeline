import os

def find_mask_folders_with_image_counts():
    current_dir = os.getcwd()
    result = {}

    for root, dirs, files in os.walk(current_dir):
        for dir_name in dirs:
            if dir_name.lower() in ('test_masks_cryptomatte2','test_masks_cryptomatte'):
                mask_path = os.path.join(root, dir_name)
                file_list = os.listdir(mask_path)
                # 只统计图片文件
                image_count = sum(
                    1 for f in file_list
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                )
                if image_count > 0:
                    # 以绝对路径为key
                    abs_path = os.path.abspath(mask_path)
                    result[abs_path] = image_count

    return result

if __name__ == '__main__':
    mask_counts = find_mask_folders_with_image_counts()
    print("所有非空mask/masks子文件夹及其图片数量：")
    for path, count in mask_counts.items():
        print(f"{path}: {count}")
