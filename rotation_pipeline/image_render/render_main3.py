# 脚本1: 主图渲染脚本 (render_main.py) - 断点续传优化版（适配旋转+平移版Mask + 按相机/场景步长）
# 功能：根据【已筛选好】的Mask文件列表，智能跳过已渲染的主图，仅渲染缺失的部分，并追加生成CSV日志。
# 适配点：
#   1) 支持 rot_init 与 rotcwDDD_stepNN 命名；
#   2) 与 Mask 完全一致的旋转规则（对象局部Z顺时针为负）；
#   3) 「步长 move_step」改为按【场景片段 × 相机名】精确解析，确保与 Mask 一致；
#   4) 每帧贴地，确保主图与 Mask 像素级对齐。
# 前置要求：必须先运行 render_mask.py，确保 test_masks_cryptomatte2 目录存在。

import time
import bpy
import os
import math
import re
from mathutils import Vector, Euler
from PIL import Image
import tempfile
import numpy as np
import csv
import argparse
import sys


def get_command_line_args():
    argv = sys.argv
    if "--" not in argv:
        return None
    args_to_parse = argv[argv.index("--") + 1:]
    parser = argparse.ArgumentParser(description="通过命令行参数控制的Blender主图渲染脚本。")
    parser.add_argument("--gpu-index", dest="target_gpu_index", type=int, required=True, help="要使用的GPU设备索引 (必需)")
    parser.add_argument("--object", dest="object_blend_path", type=str, required=True, help="要渲染的物体.blend文件路径 (必需)")
    parser.add_argument("--scene", dest="scene_blend_path", type=str, required=True, help="渲染所用的场景.blend文件路径 (必需)")
    parser.add_argument("--output", dest="output_path", type=str, required=True, help="渲染输出的【完整】目标目录 (必需)")
    try:
        return parser.parse_args(args=args_to_parse)
    except SystemExit:
        return None


# --- 执行参数解析并配置脚本变量 ---
cli_args = get_command_line_args()
if cli_args is None:
    print("❌ 错误：未检测到有效的命令行参数或解析失败。")
    print("   示例: blender -b -P render_main.py -- --scene <path> --object <path> --output <path> --gpu-index 0")
    bpy.ops.wm.quit_blender()

target_gpu_index = cli_args.target_gpu_index
object_blend_path = cli_args.object_blend_path
scene_blend_path = cli_args.scene_blend_path
output_path = cli_args.output_path

os.makedirs(output_path, exist_ok=True)

# ========== 关键改动：按【场景 × 相机】解析步长（与 Mask 保持一致） ==========
def _cam_key(cam_name: str) -> str:
    """从 'Camera_1_0' 提取 '1_0' 作为索引键。"""
    parts = cam_name.split("_")
    if len(parts) >= 3:
        return f"{parts[1]}_{parts[2]}"
    return cam_name

def resolve_per_cam_move_step(scene_path: str, cam_name: str):
    """
    返回给定场景路径 + 相机名的步长向量 [s, s, 0.0]。
    映射与 Mask 脚本保持一致；若未命中具体表项，采用相同兜底逻辑。
    """
    scene_lower = scene_path.lower()

    table = {
        "indoor1": {"1_0": 0.13,  "1_6": 0.55},
        "indoor4": {"1_0": 0.04,  "1_6": 0.17},
        "indoor5": {"1_0": 0.055, "1_6": 0.19},
        "indoor7": {"1_0": 0.1,  "1_6": 1.5},
        "indoor8": {"1_0": 0.055, "1_6": 0.19},
        "indoor9": {"1_0": 0.03,  "1_6": 0.19},
        "outdoor1": {"1_0": 0.3,  "1_6": 0.9},
        "outdoor5": {"1_0": 0.7,  "1_6": 1.0},
    }
    # 优先匹配具体 indoorX/outdoorX
    matched_key = None
    for key in table.keys():
        if key in scene_lower:
            matched_key = key
            break

    cam_suffix = _cam_key(cam_name)
    if matched_key is not None:
        row = table[matched_key]
        if cam_suffix not in row:
            raise RuntimeError(f"场景 '{matched_key}' 的步长表不包含相机键 '{cam_suffix}'（相机名: {cam_name}）。")
        s = row[cam_suffix]
        return [s, s, 0.0]

    # 兜底策略（与 Mask 一致）
    if "indoor" in scene_lower:
        # indoor 通用：1_0 -> 0.17；1_6 -> 0.71
        fallback = {"1_0": 0.17, "1_6": 0.71}
        s = fallback.get(cam_suffix, 0.17)
        return [s, s, 0.0]
    elif "outdoor3" in scene_lower:
        return [0.10, 0.10, 0.0]
    elif ("outdoor6" in scene_lower) or ("outdoor12" in scene_lower):
        return [0.23, 0.23, 0.0]
    else:
        # 默认
        return [0.70, 0.70, 0.0]

# 其余保持原逻辑
num_frames = 10  # 未实际使用，但保留原变量以兼容旧代码


### === 工具函数 (保持不变/小幅增强) === ###
def direction_to_euler(direction_vec):
    target = Vector(direction_vec)
    rot_quat = target.to_track_quat('-Z', 'Y')
    return rot_quat.to_euler()

def get_scene_bounds():
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and not obj.hide_get():
            bbox_world = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
            for v in bbox_world:
                min_x, min_y, min_z = min(min_x, v.x), min(min_y, v.y), min(min_z, v.z)
                max_x, max_y, max_z = max(max_x, v.x), max(max_y, v.y), max(max_z, v.z)
    return Vector((min_x, min_y, min_z)), Vector((max_x, max_y, max_z))

def auto_adjust_object_size(obj, scene_bounds_min, scene_bounds_max):
    size = scene_bounds_max - scene_bounds_min
    avg_dim = (size.x + size.y + size.z) / 3
    target_size = avg_dim * 0.005
    current_size = max(obj.dimensions.x, obj.dimensions.y, obj.dimensions.z)
    if current_size == 0: return
    scale_factor = target_size / current_size
    obj.scale *= scale_factor
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.select_set(False)

def adjust_lighting_to_target_brightness(target_brightness=0.4, preview_resolution=(128, 72), camera_list=None):
    scene = bpy.context.scene
    if camera_list: scene.camera = camera_list[0]
    old_x, old_y = scene.render.resolution_x, scene.render.resolution_y
    old_samples = scene.cycles.samples
    scene.render.resolution_x, scene.render.resolution_y = preview_resolution
    scene.cycles.samples = 32
    scene.render.image_settings.file_format = 'PNG'
    temp_file = os.path.join(tempfile.gettempdir(), "preview.png")
    scene.render.filepath = temp_file
    bpy.ops.render.render(write_still=True)
    time.sleep(1.5)
    img = Image.open(temp_file).convert('RGB')
    avg_brightness = np.asarray(img).astype(np.float32).mean() / 255.0
    if avg_brightness > 0:
        scale = target_brightness / avg_brightness
        for light in bpy.data.lights: light.energy *= scale
        if scene.world and scene.world.node_tree:
            for node in scene.world.node_tree.nodes:
                if node.type == 'BACKGROUND': node.inputs[1].default_value *= scale
    scene.render.resolution_x, scene.render.resolution_y = old_x, old_y
    scene.cycles.samples = old_samples

def find_ground_object_auto():
    candidates = [obj for obj in bpy.data.objects if obj.type == 'MESH' and not obj.hide_get()]
    for obj in candidates:
        if obj.name.lower() == "ground": return obj
    lowest, ground_obj = float('inf'), None
    for obj in candidates:
        z = min(v.z for v in [obj.matrix_world @ Vector(c) for c in obj.bound_box])
        if z < lowest: lowest, ground_obj = z, obj
    return ground_obj

def fix_ground_orientation(ground_obj):
    if ground_obj.scale.z < 0: ground_obj.scale.z = abs(ground_obj.scale.z)
    if abs(math.degrees(ground_obj.rotation_euler.x)) in [90, 180, 270]: ground_obj.rotation_euler.x = 0.0

def adjust_object_to_ground_with_ray(obj, ground_obj):
    scene, depsgraph = bpy.context.scene, bpy.context.evaluated_depsgraph_get()
    bbox_world = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_z = min(v.z for v in bbox_world)
    center_x = sum(v.x for v in bbox_world) / 8.0
    center_y = sum(v.y for v in bbox_world) / 8.0
    ray_origin = Vector((center_x, center_y, min_z + 0.05))
    hit, location, _, _, hit_obj, _ = scene.ray_cast(depsgraph, ray_origin, Vector((0, 0, -1)), distance=2.0)
    if hit and (ground_obj is None or hit_obj == ground_obj):
        obj.location.z += location.z - min_z

def create_cameras_rings(camera_list, bounds_min, bounds_max, target_obj, rings_config):
    bbox_world = [target_obj.matrix_world @ Vector(corner) for corner in target_obj.bound_box]
    center = sum(bbox_world, Vector((0, 0, 0))) / 8.0
    scene_size = bounds_max - bounds_min
    max_span, scene_height = max(scene_size.x, scene_size.y), scene_size.z
    for ring_idx, (radius_ratio, height_ratio, cameras_count) in enumerate(rings_config):
        radius, height = max_span * radius_ratio, scene_height * height_ratio
        for i in range(cameras_count):
            angle = 2 * math.pi * i / cameras_count
            cam_loc = Vector((center.x + radius * math.cos(angle), center.y + radius * math.sin(angle), center.z + height))
            cam_dir = center - cam_loc
            cam_data = bpy.data.cameras.new(name=f"Camera_{ring_idx}_{i}")
            cam_obj = bpy.data.objects.new(name=f"Camera_{ring_idx}_{i}", object_data=cam_data)
            cam_obj.location = cam_loc
            cam_obj.rotation_euler = direction_to_euler(cam_dir)
            bpy.context.collection.objects.link(cam_obj)
            camera_list.append(cam_obj)

def euler_with_local_z(base_euler: Euler, cw_deg: int) -> Euler:
    """基于对象初始姿态，绕对象局部Z顺时针旋转指定角度（负角度），返回绝对欧拉角。"""
    rad = math.radians(-cw_deg)  # 顺时针为负
    return Euler((base_euler.x, base_euler.y, base_euler.z + rad), base_euler.order)


### === 渲染设置 (保持不变) === ###
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'CUDA'
prefs.get_devices()
cuda_devices = [d for d in prefs.devices if d.type == 'CUDA']
if target_gpu_index >= len(cuda_devices): raise IndexError("指定 GPU index 超出范围")
for i, d in enumerate(cuda_devices): d.use = (i == target_gpu_index)
scene.cycles.device = 'GPU'
scene.cycles.samples = 1024
scene.cycles.use_denoising = False
scene.cycles.denoiser = 'OPENIMAGEDENOISE'
scene.cycles.max_bounces = 32
scene.cycles.transparent_max_bounces = 32
scene.cycles.transmission_max_bounces = 12
scene.view_settings.view_transform = 'Filmic'
scene.view_settings.look = 'None'
scene.view_settings.gamma = 0.5


### === 加载与准备 (保持不变) === ###
bpy.ops.wm.open_mainfile(filepath=scene_blend_path)
scene = bpy.context.scene
scene.render.resolution_x = 1024
scene.render.resolution_y = 1024
scene.render.resolution_percentage = 100

for light in bpy.data.lights: light.energy *= 0.3
if scene.world and scene.world.node_tree:
    for node in scene.world.node_tree.nodes:
        if node.type == 'BACKGROUND': node.inputs[1].default_value *= 0.3

with bpy.data.libraries.load(object_blend_path) as (data_from, data_to):
    data_to.objects = data_from.objects

imported_mesh_objs, empty_name = [], None
for o in data_to.objects:
    if o:
        bpy.context.collection.objects.link(o)
        if o.type == 'MESH': imported_mesh_objs.append(o)
        elif o.type == 'EMPTY' and empty_name is None: empty_name = o.name

if not imported_mesh_objs: raise RuntimeError("未找到任何 mesh 类型主物体")
for obj in imported_mesh_objs: obj.select_set(True)
bpy.context.view_layer.objects.active = imported_mesh_objs[0]
bpy.ops.object.join()
main_obj = bpy.context.view_layer.objects.active
if empty_name: main_obj.name = empty_name

ground_obj = find_ground_object_auto()
if ground_obj: fix_ground_orientation(ground_obj)
bounds_min, bounds_max = get_scene_bounds()
auto_adjust_object_size(main_obj, bounds_min, bounds_max)
if ground_obj: adjust_object_to_ground_with_ray(main_obj, ground_obj)
start_location = list(main_obj.location)
start_rotation = main_obj.rotation_euler.copy()  # 记录初始姿态，供绝对角计算

camera_rings_config = [(0.01, 0.005, 10), (0.015, 0.008, 10), (0.02, 0.012, 10)]
camera_list = []
create_cameras_rings(camera_list, bounds_min, bounds_max, main_obj, camera_rings_config)

# 自适应亮度（保留原逻辑）
adjust_lighting_to_target_brightness(0.4, camera_list=camera_list)


# ==============================================================================
# ### === 渲染任务生成：【依据旋转+平移版Mask命名】 === ###
# ==============================================================================
output_image_dir = os.path.join(output_path, 'images')
output_mask_dir = os.path.join(output_path, 'test_masks_cryptomatte2')
os.makedirs(output_image_dir, exist_ok=True)

print("\n" + "="*50)
print("⚙️ 开始根据Mask和已有的主图生成增量渲染任务...")

# 1) 检查Mask目录
if not os.path.isdir(output_mask_dir):
    print(f"❌ 致命错误: Mask目录 '{output_mask_dir}' 不存在。请先运行 render_mask.py。")
    bpy.ops.wm.quit_blender()

# 2) 全部 Mask 文件
all_mask_filenames = sorted([f for f in os.listdir(output_mask_dir) if f.lower().endswith('.png')])
if not all_mask_filenames:
    print("⚠️ 警告: Mask目录为空，没有需要渲染的主图。脚本将正常退出。")
    bpy.ops.wm.quit_blender()

# 3) 已有主图
existing_image_filenames = set(f for f in os.listdir(output_image_dir) if f.lower().endswith('.png'))

# 4) 差集：需要渲染的文件
tasks_to_render_filenames = [m for m in all_mask_filenames if m not in existing_image_filenames]
if not tasks_to_render_filenames:
    print("✅ 所有在Mask目录中的文件均已渲染成主图。无需操作。")
    bpy.ops.wm.quit_blender()

total_skipped = len(all_mask_filenames) - len(tasks_to_render_filenames)
print(f"✅ 检查完成。共找到 {len(all_mask_filenames)} 个Mask，其中 {total_skipped} 个主图已存在。")
print(f"   将为剩余的 {len(tasks_to_render_filenames)} 个文件渲染主图。")

# 5) 相机索引表
camera_map = {cam.name: cam for cam in camera_list}

# 为避免重复计算步长，这里缓存「相机名 -> 步长向量」
per_cam_move_step_map = {}

# 6) 解析 rot_init / rotcwDDD_stepNN
render_tasks = []
pattern_init = re.compile(r'_(Camera_\d+_\d+)_rot_init\.png$')
pattern_step = re.compile(r'_(Camera_\d+_\d+)_rotcw(\d{3})_step(\d{2})\.png$')

for img_name in tasks_to_render_filenames:
    try:
        cam = None
        pos = None
        rot_euler = None

        m_init = pattern_init.search(img_name)
        m_step = pattern_step.search(img_name)

        if m_init:
            cam_name = m_init.group(1)
            cam = camera_map.get(cam_name)
            if not cam:
                print(f"   -> ⚠️ 相机不存在（跳过）: {cam_name} 对应文件 '{img_name}'")
                continue

            # 记录并打印该相机的专属步长（即使 init 不用）
            if cam_name not in per_cam_move_step_map:
                per_cam_move_step_map[cam_name] = resolve_per_cam_move_step(scene_blend_path, cam_name)
                print(f"📐 相机 {cam_name} 的步长 move_step = {per_cam_move_step_map[cam_name]} (scene='{scene_blend_path}')")

            # init：初始位置与姿态
            pos = tuple(start_location)
            rot_euler = Euler((start_rotation.x, start_rotation.y, start_rotation.z), start_rotation.order)

        elif m_step:
            cam_name, deg_str, step_str = m_step.groups()
            cam = camera_map.get(cam_name)
            if not cam:
                print(f"   -> ⚠️ 相机不存在（跳过）: {cam_name} 对应文件 '{img_name}'")
                continue

            # 获取该相机的专属步长
            if cam_name not in per_cam_move_step_map:
                per_cam_move_step_map[cam_name] = resolve_per_cam_move_step(scene_blend_path, cam_name)
                print(f"📐 相机 {cam_name} 的步长 move_step = {per_cam_move_step_map[cam_name]} (scene='{scene_blend_path}')")

            step_vec = per_cam_move_step_map[cam_name]
            angle_deg = int(deg_str)          # 000..330
            step_idx = int(step_str)          # 01..03（与Mask一致：从1开始）

            # 位置：从初始化点出发，沿「该相机专属 step_vec」前进 step_idx 步
            pos = (
                start_location[0] + step_vec[0] * step_idx,
                start_location[1] + step_vec[1] * step_idx,
                start_location[2] + step_vec[2] * step_idx
            )
            # 姿态：对象局部Z顺时针 angle_deg（绝对角）
            rot_euler = euler_with_local_z(start_rotation, angle_deg)

        else:
            # 与旋转+平移命名不匹配的，跳过（兼容旧文件）
            print(f"   -> ℹ️ 跳过非旋转+平移命名: {img_name}")
            continue

        render_tasks.append((pos, rot_euler, cam, img_name))

    except Exception as e:
        print(f"   -> ❌ 解析文件名 '{img_name}' 时异常，已跳过：{e}")

if not render_tasks:
    print("❌ 致命错误: 成功解析了0个渲染任务。请检查Mask文件名格式或是否所有任务已完成。")
    bpy.ops.wm.quit_blender()

print(f"✅ 成功构建了 {len(render_tasks)} 个渲染任务。")
print("="*50 + "\n")


# ==============================================================================
# ### === 主图渲染与CSV日志记录（基于新的 render_tasks） === ###
# ==============================================================================
csv_path = os.path.join(output_path, 'render_info.csv')
print(f"✍️ 将向 {csv_path} 写入渲染信息 (追加模式)...")

write_header = not os.path.exists(csv_path)
with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    if write_header:
        writer.writerow([
            'image_name', 'scene_size', 'light_name', 'light_pos', 'light_strength', 'light_direction',
            'camera_pos', 'camera_euler', 'camera_focal_length', 'camera_sensor_width', 'camera_principal_point',
            'object_name', 'object_size', 'object_pos', 'object_rotation_deg'
        ])

    for i, (pos, rot_euler, cam, img_name) in enumerate(render_tasks):
        print(f"🎬 渲染主图 ({i+1}/{len(render_tasks)}): {img_name}")

        # 设置对象位姿并贴地（确保与Mask对齐）
        main_obj.location = Vector(pos)
        main_obj.rotation_euler = rot_euler
        if ground_obj:
            adjust_object_to_ground_with_ray(main_obj, ground_obj)

        # 设置相机与输出
        scene.camera = cam
        scene.render.filepath = os.path.join(output_image_dir, img_name)
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGB'
        bpy.ops.render.render(write_still=True)

        # --- CSV信息收集 ---
        scene_size = bounds_max - bounds_min
        scene_size_str = f"({scene_size.x:.4f},{scene_size.y:.4f},{scene_size.z:.4f})"

        light_infos = []
        for light_obj in [obj for obj in bpy.data.objects if obj.type == 'LIGHT']:
            light = light_obj.data
            direction_vec = light_obj.matrix_world.to_quaternion() @ Vector((0, 0, -1))
            light_infos.append({
                'name': light_obj.name,
                'pos': tuple(round(x, 4) for x in light_obj.location),
                'strength': getattr(light, 'energy', None),
                'direction': tuple(round(x, 4) for x in direction_vec) if light_obj.type in ['SPOT', 'AREA', 'SUN'] else 'N/A'
            })

        cam_pos = tuple(round(x, 4) for x in cam.location)
        cam_euler = tuple(round(x, 4) for x in cam.rotation_euler)
        principal_x = scene.render.resolution_x * (0.5 - getattr(cam.data, 'shift_x', 0.0))
        principal_y = scene.render.resolution_y * (0.5 + getattr(cam.data, 'shift_y', 0.0))

        obj_size = tuple(round(x, 4) for x in main_obj.dimensions)
        obj_pos = tuple(round(x, 4) for x in main_obj.location)
        obj_rot_deg = tuple(round(math.degrees(x), 2) for x in main_obj.rotation_euler)

        if not light_infos:
            writer.writerow([
                img_name, scene_size_str, 'N/A', 'N/A', 'N/A', 'N/A',
                cam_pos, cam_euler, getattr(cam.data, 'lens', None), getattr(cam.data, 'sensor_width', None),
                (round(principal_x, 2), round(principal_y, 2)), main_obj.name, obj_size, obj_pos, obj_rot_deg
            ])
        else:
            for light_info in light_infos:
                writer.writerow([
                    img_name, scene_size_str, light_info['name'], light_info['pos'], light_info['strength'], light_info['direction'],
                    cam_pos, cam_euler, getattr(cam.data, 'lens', None), getattr(cam.data, 'sensor_width', None),
                    (round(principal_x, 2), round(principal_y, 2)), main_obj.name, obj_size, obj_pos, obj_rot_deg
                ])

print(f"\n✅ 主图渲染完成！")
print(f"📁 图像保存在: {output_image_dir}")
print(f"📄 渲染信息保存在: {csv_path}")
