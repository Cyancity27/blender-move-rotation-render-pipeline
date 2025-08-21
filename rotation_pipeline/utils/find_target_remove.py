import os
import numpy as np
from PIL import Image
import logging


def is_special_mask(bin_mask, area_threshold=0.005):
    total_pixels = bin_mask.size
    white_pixels = np.sum(bin_mask)

    if white_pixels == 0:
        return True
    is_small_area = (white_pixels / total_pixels) < area_threshold
    if is_small_area:
        touches_top = np.any(bin_mask[0, :])
        touches_bottom = np.any(bin_mask[-1, :])
        touches_left = np.any(bin_mask[:, 0])
        touches_right = np.any(bin_mask[:, -1])
        if touches_top or touches_bottom or touches_left or touches_right:
            return True
    return False

def binarize_and_save_mask(img_path):
    """
    读取mask图片，二值化处理后覆盖保存（黑0，白255）
    """
    try:
        with Image.open(img_path) as img:
            img_array = np.array(img)
            if img_array.ndim == 3:
                gray_mask = img_array[..., 0]
            else:
                gray_mask = img_array
            # 强制二值化
            bin_mask = (gray_mask > 180).astype(np.uint8) * 255
            # 保存覆盖
            out_img = Image.fromarray(bin_mask.astype(np.uint8), mode='L')
            out_img.save(img_path)
            logging.info(f"  [二值化并覆盖保存] {img_path}")
            return bin_mask // 255  # 返回0/1掩码用于后续筛选
    except Exception as e:
        logging.error(f"  [失败] 二值化保存mask时出错: {img_path} -> {e}")
        return None


def find_and_delete_unwanted_images(mask_folder, area_threshold=0.005):
    """
    批量二值化并覆盖mask图片，然后筛选和删除“特殊”掩码及其同名数据图片。
    """
    selected_names = []
    if not os.path.isdir(mask_folder):
        logging.error(f"错误：文件夹不存在，跳过处理: {mask_folder}")
        return

    for file in os.listdir(mask_folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(mask_folder, file)
            # 1. 覆盖式二值化，并获得0/1掩码
            bin_mask = binarize_and_save_mask(file_path)
            if bin_mask is None:
                continue  # 二值化失败跳过
            # 2. 特殊掩码筛选
            if is_special_mask(bin_mask, area_threshold):
                selected_names.append(file)

    folder_name_display = os.path.basename(os.path.dirname(mask_folder)) + "/" + os.path.basename(mask_folder)
    logging.info(f"\n--- 开始处理文件夹: {folder_name_display} ---")
    
    if not selected_names:
        logging.info("在此文件夹中未找到符合条件的目标图片。")
        logging.info("-" * 50)
        return

    logging.info(f"找到 {len(selected_names)} 张符合条件的图片，准备开始删除操作。")
    data_folder = os.path.dirname(mask_folder)

    for name in sorted(selected_names):
        mask_file_path = os.path.join(mask_folder, name)
        images_folder = os.path.join(os.path.dirname(mask_folder), 'images')
        data_file_path = os.path.join(images_folder, name)

        try:
            if os.path.exists(mask_file_path):
                os.remove(mask_file_path)
                logging.info(f"  [成功] 已删除 Mask 文件: {mask_file_path}")
            else:
                logging.warning(f"  [警告] Mask 文件不存在，无法删除: {mask_file_path}")
        except OSError as e:
            logging.error(f"  [失败] 删除 Mask 文件时出错 {mask_file_path}: {e}")

        try:
            if os.path.exists(data_file_path):
                os.remove(data_file_path)
                logging.info(f"  [成功] 已删除 Data 文件: {data_file_path}")
            else:
                logging.warning(f"  [警告] Data 文件不存在，无法删除: {data_file_path}")
        except OSError as e:
            logging.error(f"  [失败] 删除 Data 文件时出错 {data_file_path}: {e}")
    
    logging.info(f"--- 文件夹处理完毕: {folder_name_display} ---")
    logging.info("-" * 50)

if __name__ == '__main__':
    log_directory = '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/logs/sculpture6/binary/'
    os.makedirs(log_directory, exist_ok=True)
    log_file = os.path.join(log_directory, 'selection_and_deletion_results.log')

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


    # folders = [
    # '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_indoor1_sculpture6/test_masks_cryptomatte2',
    # '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_indoor2_sculpture6/test_masks_cryptomatte2',
    # '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_indoor3_sculpture6/test_masks_cryptomatte2',
    # '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_indoor4_sculpture6/test_masks_cryptomatte2',
    # '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_indoor5_sculpture6/test_masks_cryptomatte2',
    # '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_indoor7_sculpture6/test_masks_cryptomatte2',
    # '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_indoor8_sculpture6/test_masks_cryptomatte2',
    # '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_indoor9_sculpture6/test_masks_cryptomatte2',
    # '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_indoor10_sculpture6/test_masks_cryptomatte2',
    # '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor1_sculpture6/test_masks_cryptomatte2',
    # '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor2_sculpture6/test_masks_cryptomatte2',
    # '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor3_sculpture6/test_masks_cryptomatte2',
    # '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor5_sculpture6/test_masks_cryptomatte2',
    # '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor6_sculpture6/test_masks_cryptomatte2',
    # '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor7_sculpture6/test_masks_cryptomatte2',
    # '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor8_sculpture6/test_masks_cryptomatte2',
    # '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor9_sculpture6/test_masks_cryptomatte2',
    # '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor11_sculpture6/test_masks_cryptomatte2',
    # '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor12_sculpture6/test_masks_cryptomatte2',
    # '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor13_sculpture6/test_masks_cryptomatte2',
    # '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor14_sculpture6/test_masks_cryptomatte2',
    # '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor15_sculpture6/test_masks_cryptomatte2',
    # '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor16_sculpture6/test_masks_cryptomatte2',
    # '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor17_sculpture6/test_masks_cryptomatte2'
    # ]

    # folders = [
    #     '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor5_sculpture6/test_masks_cryptomatte2',
    #     '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor6_sculpture6/test_masks_cryptomatte2',
    #     '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor7_sculpture6/test_masks_cryptomatte2',
    #     '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor8_sculpture6/test_masks_cryptomatte2',
    #     '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor9_sculpture6/test_masks_cryptomatte2',
    #     '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor11_sculpture6/test_masks_cryptomatte2',
    #     '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor12_sculpture6/test_masks_cryptomatte2',
    #     '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor13_sculpture6/test_masks_cryptomatte2',
    #     '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor14_sculpture6/test_masks_cryptomatte2',
    #     '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor15_sculpture6/test_masks_cryptomatte2',
    #     '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor16_sculpture6/test_masks_cryptomatte2',
    #     '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor17_sculpture6/test_masks_cryptomatte2'
    # ]
    
    # folders = [
    #     '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_indoor5_sculpture6/test_masks_cryptomatte2',
    #     '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_indoor8_sculpture6/test_masks_cryptomatte2',
    #     '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_indoor9_sculpture6/test_masks_cryptomatte2',
    #     '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor1_sculpture6/test_masks_cryptomatte2',
    #     '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor5_sculpture6/test_masks_cryptomatte2',
    #     '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor6_sculpture6/test_masks_cryptomatte2',
    #     '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor8_sculpture6/test_masks_cryptomatte2',
    #     '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor9_sculpture6/test_masks_cryptomatte2'
    # ]

    # folders = [
    #     '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_indoor5_sculpture6/test_masks_cryptomatte2',
    #     '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_indoor9_sculpture6/test_masks_cryptomatte2',
    #     '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor3_sculpture6/test_masks_cryptomatte2',
    #     '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor5_sculpture6/test_masks_cryptomatte2',
    #     '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor6_sculpture6/test_masks_cryptomatte2',
    #     '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor8_sculpture6/test_masks_cryptomatte2',
    #     '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor12_sculpture6/test_masks_cryptomatte2',
    #     '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor16_sculpture6/test_masks_cryptomatte2'
    # ]

    folders = [
        '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_indoor1_sculpture6/test_masks_cryptomatte2',
        '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_indoor4_sculpture6/test_masks_cryptomatte2',
        '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_indoor5_sculpture6/test_masks_cryptomatte2',
        '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_indoor7_sculpture6/test_masks_cryptomatte2',
        '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_indoor8_sculpture6/test_masks_cryptomatte2',
        '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_indoor9_sculpture6/test_masks_cryptomatte2',
        '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor1_sculpture6/test_masks_cryptomatte2',
        '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/outputs/sculpture6/output_outdoor5_sculpture6/test_masks_cryptomatte2',
    ]

    area_threshold = 0.005
    for folder in folders:
        find_and_delete_unwanted_images(folder, area_threshold)

    final_message = f"处理完成。详细操作日志已保存到: {os.path.abspath(log_file)}"
    print(final_message)
    logging.info("\n" + "="*20 + " 所有任务已执行完毕 " + "="*20)
