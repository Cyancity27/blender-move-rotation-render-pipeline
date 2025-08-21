import os

root_dir = '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/outputs/sculpture4/output_outdoor7_sculpture4/test_masks_cryptomatte2'  # 请替换为你的目标路径

for current_dir, dirs, files in os.walk(root_dir):
    for file in files:
        if file.lower().endswith('.png'):
            print(file)
