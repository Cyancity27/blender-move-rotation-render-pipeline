import os

root_dir = '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/outputs'

for current_dir, dirs, files in os.walk(root_dir):
    # 判断当前目录下是否同时存在两个目标子文件夹
    if 'images' in dirs and 'test_masks_cryptomatte2' in dirs:
        images_dir = os.path.join(current_dir, 'images')
        masks_dir = os.path.join(current_dir, 'test_masks_cryptomatte2')

        images_png_count = len([f for f in os.listdir(images_dir) if f.lower().endswith('.png')])
        masks_png_count = len([f for f in os.listdir(masks_dir) if f.lower().endswith('.png')])

        print(f'文件夹: {current_dir}')
        print(f'  images文件夹png数量: {images_png_count}')
        print(f'  test_masks_cryptomatte2文件夹png数量: {masks_png_count}')
        print('-' * 40)
