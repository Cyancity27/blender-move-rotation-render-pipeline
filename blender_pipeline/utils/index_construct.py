# import os
# import json
# from itertools import combinations
# from PIL import Image
# import numpy as np

# folders = [
#     '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/outputs/sculpture1/output_4_sculpture1/mask',
#     '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/outputs/sculpture2/output_4_sculpture2/mask',
#     '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/outputs/sculpture3/output_4_sculpture3/mask',
#     '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/outputs/sculpture5/output_4_sculpture5/mask',
#     '/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/outputs/sculpture6/output_4_sculpture6/mask',
# ]

# # --- 配置您的路径 ---
# IMAGE_DIR = "/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/outputs/sculpture6/output_4_sculpture6/images"
# MASK_DIR = "/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/outputs/sculpture6/output_4_sculpture6/mask"
# OUTPUT_META_FILE = "/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/outputs/sculpture6/output_4_sculpture6/dataset_pairs.json"
# # --------------------


# # 1. 按相机分组
# camera_groups = {}
# for filename in os.listdir(IMAGE_DIR):
#     if not filename.endswith('.png'):
#         continue
#     parts = filename.split('_')
#     # 假定相机名是固定的格式，例如 Camera_X_Y
#     cam_name = f"{parts[2]}_{parts[3]}_{parts[4]}"
    
#     if cam_name not in camera_groups:
#         camera_groups[cam_name] = []
        
#     mask_path = os.path.join(MASK_DIR, filename)
#     if os.path.exists(mask_path):
#         camera_groups[cam_name].append(filename)

# print(f"找到 {len(camera_groups)} 个有效相机视角。")

# # 2. 创建所有可能的 (源, 目标) 配对
# all_pairs = []
# for cam_name, files in camera_groups.items():
#     # 对文件名进行排序，确保帧的顺序是正确的
#     files.sort() 
    
#     # 从一个相机的多帧中任取2帧作为一对
#     for src_file, tgt_file in combinations(files, 2):
#         all_pairs.append({
#             "camera": cam_name,
#             "source_image": src_file,
#             "target_image": tgt_file
#         })

# print(f"构建完成！总共生成了 {len(all_pairs)} 个训练对。")

# # 3. 保存配对列表到JSON文件
# with open(OUTPUT_META_FILE, 'w') as f:
#     json.dump(all_pairs, f, indent=4)

# print(f"配对信息已保存到: {OUTPUT_META_FILE}")



import os
import json
from itertools import combinations

from PIL import Image
import numpy as np

# --- 配置你的数据集组 ---
DATASET_GROUPS = [
    "/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/outputs/sculpture1/output_4_sculpture1",
    "/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/outputs/sculpture2/output_4_sculpture2",
    "/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/outputs/sculpture3/output_4_sculpture3",
    "/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/outputs/sculpture5/output_4_sculpture5",
    "/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/outputs/sculpture6/output_4_sculpture6"
]

for group_dir in DATASET_GROUPS:
    IMAGE_DIR = os.path.join(group_dir, "images")
    MASK_DIR = os.path.join(group_dir, "mask")
    OUTPUT_META_FILE = os.path.join(group_dir, "dataset_pairs.json")

    if not os.path.isdir(IMAGE_DIR) or not os.path.isdir(MASK_DIR):
        print(f"跳过无效目录: {group_dir}")
        continue

    # 1. 按相机分组
    camera_groups = {}
    for filename in os.listdir(IMAGE_DIR):
        if not filename.endswith('.png'):
            continue
        parts = filename.split('_')
        if len(parts) < 5:
            print(f"警告：文件名格式异常，跳过 {filename}")
            continue
        cam_name = f"{parts[2]}_{parts[3]}_{parts[4]}"

        if cam_name not in camera_groups:
            camera_groups[cam_name] = []

        mask_path = os.path.join(MASK_DIR, filename)
        if os.path.exists(mask_path):
            camera_groups[cam_name].append(filename)

    print(f"[{group_dir}] 找到 {len(camera_groups)} 个有效相机视角。")

    # 2. 创建所有可能的 (源, 目标) 配对
    all_pairs = []
    for cam_name, files in camera_groups.items():
        files.sort() 
        for src_file, tgt_file in combinations(files, 2):
            all_pairs.append({
                "camera": cam_name,
                "source_image": src_file,
                "target_image": tgt_file
            })

    print(f"[{group_dir}] 构建完成！总共生成了 {len(all_pairs)} 个训练对。")

    # 3. 保存配对列表到JSON文件
    with open(OUTPUT_META_FILE, 'w') as f:
        json.dump(all_pairs, f, indent=4)

    print(f"[{group_dir}] 配对信息已保存到: {OUTPUT_META_FILE}")

print("全部数据集处理完毕。")
