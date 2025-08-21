# 脚本2: Mask渲染脚本 (render_mask.py) - Cryptomatte专业版 (智能跳过 + 精准断点续传 v3 + 旋转平移采样)
# 功能：使用Cryptomatte生成Mask，智能跳过无效序列，并在中断后能从上次位置继续渲染。
# 本版新增：三层相机中随机选两个不同环的两台相机；对这两台相机执行“对象自旋（顺时针30°×12）+ 每角度前进3步（世界系move_step）+ 始终贴地”的数据采样。
# 版本：在不改变原有逻辑框架的前提下重构任务蓝图，仅渲染旋转+平移的两台相机（取消原 forward/reverse 任务）。

import bpy
import os
import math
from mathutils import Vector, Euler
import argparse
import sys
import numpy as np
from PIL import Image
import json
import random
import math as _math

def get_command_line_args():
    """解析从命令行传递给Blender脚本的参数。"""
    argv = sys.argv
    if "--" not in argv: return None
    args_to_parse = argv[argv.index("--") + 1:]
    parser = argparse.ArgumentParser(description="通过命令行参数控制的Blender精确Mask渲染脚本。")
    parser.add_argument("--gpu-index", dest="target_gpu_index", type=int, required=True, help="GPU索引")
    parser.add_argument("--object", dest="object_blend_path", type=str, required=True, help="物体.blend文件路径")
    parser.add_argument("--scene", dest="scene_blend_path", type=str, required=True, help="场景.blend文件路径")
    parser.add_argument("--output", dest="output_path", type=str, required=True, help="输出目录")
    try:
        return parser.parse_args(args=args_to_parse)
    except SystemExit:
        return None

# --- 参数解析 ---
cli_args = get_command_line_args()
if cli_args is None:
    print("❌ 错误：参数解析失败。")
    bpy.ops.wm.quit_blender()

target_gpu_index = cli_args.target_gpu_index
object_blend_path = cli_args.object_blend_path
scene_blend_path = cli_args.scene_blend_path
output_path = cli_args.output_path

os.makedirs(output_path, exist_ok=True)

# 原版参数（保留）
num_frames = 10
# 根据 --scene 路径自动设置 move_step
scene_lower = scene_blend_path.lower()
if "indoor" in scene_lower:
    move_step = [0.04, 0.04, 0]
elif "outdoor6" in scene_lower:
    move_step = [0.65, 0.65, 0]
elif "outdoor5" in scene_lower:
    move_step = [0.4, 0.4, 0]
else:
    move_step = [0.16, 0.16, 0]
print(f"📐 根据场景路径选择的步长 move_step = {move_step}  (scene='{scene_blend_path}')")

# 新增常量：旋转+平移采样配置
ROT_STEPS = 12                  # 12个方向（360/30）
ROT_DEG_STEP = 30               # 顺时针每步30°
MOVE_STEPS_PER_ANGLE = 3        # 每个方向前进3步
ANGLE_LABEL_STYLE = "cw"        # 文件名中标记顺时针

### === 工具函数 (与原版一致/轻微增强) === ###
def direction_to_euler(direction_vec):
    return Vector(direction_vec).to_track_quat('-Z', 'Y').to_euler()

def get_scene_bounds():
    min_v = Vector((float('inf'),) * 3)
    max_v = Vector((float('-inf'),) * 3)
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and not obj.hide_get():
            for corner in obj.bound_box:
                v_world = obj.matrix_world @ Vector(corner)
                min_v.x, min_v.y, min_v.z = min(min_v.x, v_world.x), min(min_v.y, v_world.y), min(min_v.z, v_world.z)
                max_v.x, max_v.y, max_v.z = max(max_v.x, v_world.x), max(max_v.y, v_world.y), max(max_v.z, v_world.z)
    return min_v, max_v

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

def find_ground_object_auto():
    candidates = [o for o in bpy.data.objects if o.type == 'MESH' and not o.hide_get()]
    for obj in candidates:
        if "ground" in obj.name.lower(): return obj
    lowest_z, ground_obj = float('inf'), None
    for obj in candidates:
        z = min( (obj.matrix_world @ Vector(c)).z for c in obj.bound_box )
        if z < lowest_z: lowest_z, ground_obj = z, obj
    return ground_obj

def fix_ground_orientation(ground_obj):
    if ground_obj.scale.z < 0: ground_obj.scale.z = abs(ground_obj.scale.z)
    if abs(math.degrees(ground_obj.rotation_euler.x)) % 180 == 90: ground_obj.rotation_euler.x = 0.0

def adjust_object_to_ground_with_ray(obj, ground_obj):
    """将对象最低点贴到地面网格上。"""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bbox_world = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    min_z = min(v.z for v in bbox_world)
    center_xy = sum((Vector((v.x, v.y, 0)) for v in bbox_world), Vector()) / 8.0
    ray_origin = Vector((center_xy.x, center_xy.y, min_z + 0.05))
    hit, loc, *_ = bpy.context.scene.ray_cast(depsgraph, ray_origin, Vector((0, 0, -1)))
    if hit:
        obj.location.z += loc.z - min_z

def create_cameras_rings(camera_list, bounds_min, bounds_max, target_obj, rings_config):
    """按环参数创建相机，并在相机对象上记录 ring_idx 以便后续分层随机。"""
    center = sum((target_obj.matrix_world @ Vector(c) for c in target_obj.bound_box), Vector()) / 8.0
    scene_size = bounds_max - bounds_min
    max_span, scene_height = max(scene_size.x, scene_size.y), scene_size.z
    for ring_idx, (r_ratio, h_ratio, count) in enumerate(rings_config):
        radius, height = max_span * r_ratio, scene_height * h_ratio
        for i in range(count):
            angle = 2 * math.pi * i / count
            loc = Vector((center.x + radius * math.cos(angle), center.y + radius * math.sin(angle), center.z + height))
            cam_data = bpy.data.cameras.new(name=f"Camera_{ring_idx}_{i}")
            cam_obj = bpy.data.objects.new(name=cam_data.name, object_data=cam_data)
            cam_obj.location = loc
            cam_obj.rotation_euler = direction_to_euler(center - loc)
            # 记录环索引（不改变其它逻辑）
            cam_obj["ring_idx"] = ring_idx
            bpy.context.collection.objects.link(cam_obj)
            camera_list.append(cam_obj)

### === 核心改进：检查函数 (与原版一致) === ###
def is_special_mask(bin_mask, area_threshold=0.005):
    """判断一个二值化掩码是否为“特殊”情况（全黑，或面积过小且触碰边缘）。"""
    total_pixels = bin_mask.size
    white_pixels = np.sum(bin_mask)
    if white_pixels == 0: return True
    is_small_area = (white_pixels / total_pixels) < area_threshold
    if is_small_area:
        touches_top = np.any(bin_mask[0, :])
        touches_bottom = np.any(bin_mask[-1, :])
        touches_left = np.any(bin_mask[:, 0])
        touches_right = np.any(bin_mask[:, -1])
        if touches_top or touches_bottom or touches_left or touches_right: return True
    return False

def check_rendered_mask(image_path, area_threshold=0.005):
    """读取渲染后的mask图片，进行检查，返回True表示是特殊/无效mask。"""
    try:
        with Image.open(image_path) as img:
            img_array = np.array(img.convert('L'))
            bin_mask = (img_array > 180).astype(np.uint8)
            if is_special_mask(bin_mask, area_threshold): return True
        return False
    except Exception as e:
        print(f"⚠️  警告：检查图片时发生错误 {image_path}: {e}")
        return False

### === 新增功能：断点续传核心函数 (原版，未改) === ###
def get_resume_tasks(full_task_sequences, output_dir):
    """
    检查已渲染的文件，确定断点，并返回一个裁剪过的、需要继续执行的任务字典。
    能正确处理由“智能跳过”导致的不连续文件序列。
    """
    print("\n--- 正在检查已存在文件以实现断点续传 ---")
    
    # 1. 遍历总任务蓝图，找到真正“最后”一个存在的文件
    last_found_info = {
        "key": None,
        "frame_index": -1,
        "img_name": None
    }
    
    sequence_keys = list(full_task_sequences.keys())
    for key in sequence_keys:
        tasks = full_task_sequences[key]
        for i, (pos, rot_euler, img_name) in enumerate(tasks):
            filepath = os.path.join(output_dir, img_name)
            if os.path.exists(filepath):
                last_found_info["key"] = key
                last_found_info["frame_index"] = i
                last_found_info["img_name"] = img_name

    # 2. 根据找到的最后一个文件，确定起点
    start_key = None
    start_frame_index = 0

    if last_found_info["key"] is None:
        print("--- 检查完成。没有发现已渲染的文件，将从头开始。 ---")
        return full_task_sequences, sum(len(v) for v in full_task_sequences.values())

    print(f"  🔍 找到顺序上最后一张已存在图片: {last_found_info['img_name']}")
    last_filepath = os.path.join(output_dir, last_found_info['img_name'])
    
    # 检验这张图片
    if check_rendered_mask(last_filepath):
        print(f"  ❗️ 最后一张图片检验为无效，视为序列中断。")
        try:
            os.remove(last_filepath)
            print(f"  🗑️  已删除无效文件: {last_filepath}")
        except OSError as e:
            print(f"  ⚠️  警告：删除文件失败 {last_filepath}: {e}")
        
        # 从下一个序列开始
        current_key_index = sequence_keys.index(last_found_info["key"])
        next_key_index = current_key_index + 1
        if next_key_index < len(sequence_keys):
            start_key = sequence_keys[next_key_index]
            start_frame_index = 0
        else:
            start_key = None 
    else:
        print(f"  ✅ 最后一张图片检验通过。")
        start_key = last_found_info["key"]
        start_frame_index = last_found_info["frame_index"] + 1

    # 3. 构建待办任务列表
    tasks_to_run = {}
    total_remaining = 0
    
    if start_key is None:
        print("--- 检查完成。所有任务均已渲染且有效。 ---")
        return {}, 0

    start_key_index = sequence_keys.index(start_key)

    original_tasks = full_task_sequences[start_key]
    if start_frame_index < len(original_tasks):
        tasks_to_run[start_key] = original_tasks[start_frame_index:]
        total_remaining += len(tasks_to_run[start_key])

    for i in range(start_key_index + 1, len(sequence_keys)):
        key = sequence_keys[i]
        tasks_to_run[key] = full_task_sequences[key]
        total_remaining += len(tasks_to_run[key])

    total_tasks_count = sum(len(v) for v in full_task_sequences.values())
    total_skipped = total_tasks_count - total_remaining

    if total_skipped > 0:
        # 取待办首帧名用于提示
        first_key = list(tasks_to_run.keys())[0]
        resume_img_name = tasks_to_run[first_key][0][2]
        print(f"--- 检查完成。将跳过 {total_skipped} 个任务，从 {resume_img_name} 开始渲染。---")
    
    return tasks_to_run, total_remaining

### === 合成器与场景加载等设置 (与原版一致) === ###
def setup_compositor_for_cryptomatte_mask(scene, object_to_mask):
    scene.use_nodes = True
    tree = scene.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)
    render_layers = tree.nodes.new(type='CompositorNodeRLayers')
    composite_node = tree.nodes.new(type='CompositorNodeComposite')
    cryptomatte_node = tree.nodes.new(type='CompositorNodeCryptomatteV2')
    cryptomatte_node.matte_id = object_to_mask.name
    tree.links.new(render_layers.outputs['CryptoObject00'], cryptomatte_node.inputs['Image'])
    tree.links.new(cryptomatte_node.outputs['Matte'], composite_node.inputs['Image'])

def cleanup_compositor(scene):
    scene.use_nodes = False

# --- 打开场景并设置 Cycles/GPU ---
bpy.ops.wm.open_mainfile(filepath=scene_blend_path)
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'CUDA'
prefs.get_devices()
for i, d in enumerate(prefs.devices):
    if d.type == 'CUDA' and i == target_gpu_index: d.use = True
    elif d.type == 'CUDA': d.use = False
scene.cycles.device = 'GPU'
scene.cycles.max_bounces = 32
scene.cycles.transparent_max_bounces = 32
scene.cycles.transmission_max_bounces = 12
scene.cycles.diffuse_bounces = 0
scene.cycles.glossy_bounces = 0
scene.cycles.volume_bounces = 0
scene.cycles.samples = 128
scene.cycles.use_denoising = False
view_layer = scene.view_layers[0]
view_layer.use_pass_cryptomatte_object = True
scene.render.resolution_x, scene.render.resolution_y = 1024, 1024

# --- 加载对象 ---
with bpy.data.libraries.load(object_blend_path) as (data_from, data_to):
    data_to.objects = data_from.objects
imported_mesh_objs, empty_name = [], None
for o in data_to.objects:
    if o:
        bpy.context.collection.objects.link(o)
        if o.type == 'MESH': imported_mesh_objs.append(o)
        elif o.type == 'EMPTY': empty_name = o.name
if not imported_mesh_objs: raise RuntimeError("未找到任何 MESH 物体")
bpy.context.view_layer.objects.active = imported_mesh_objs[0]
for obj in imported_mesh_objs: obj.select_set(True)
bpy.ops.object.join()
main_obj = bpy.context.active_object
if empty_name: main_obj.name = empty_name

# --- 地面定位与对象尺度 ---
ground_obj = find_ground_object_auto()
if ground_obj: fix_ground_orientation(ground_obj)
bounds_min, bounds_max = get_scene_bounds()
auto_adjust_object_size(main_obj, bounds_min, bounds_max)
if ground_obj: adjust_object_to_ground_with_ray(main_obj, ground_obj)
start_location = list(main_obj.location)
start_rotation = main_obj.rotation_euler.copy()  # 记录初始姿态（用于绝对角设置）

# # --- 创建三环相机 ---
# camera_rings_config = [(0.01, 0.005, 10), (0.015, 0.008, 10), (0.02, 0.012, 10)]
# camera_list = []
# create_cameras_rings(camera_list, bounds_min, bounds_max, main_obj, camera_rings_config)

# # --- 基础命名与输出目录 ---
# scene_name = os.path.splitext(os.path.basename(scene_blend_path))[0]
# object_name = os.path.splitext(os.path.basename(object_blend_path))[0]
# output_mask_dir = os.path.join(output_path, 'test_masks_cryptomatte2')
# os.makedirs(output_mask_dir, exist_ok=True)

# # --- 选择两台相机（两个不同环），带持久化计划文件 ---
# plan_path = os.path.join(output_mask_dir, "rotation_plan.json")

# def pick_two_cameras_from_two_distinct_rings(cams):
#     rings = {}
#     for cam in cams:
#         ring_idx = int(cam.get("ring_idx", -1))
#         if ring_idx not in rings: rings[ring_idx] = []
#         rings[ring_idx].append(cam)
#     ring_indices = sorted([idx for idx in rings.keys() if idx >= 0])
#     if len(ring_indices) < 2:
#         raise RuntimeError("可用的相机环少于两个，无法按要求选择。")
#     selected_rings = random.sample(ring_indices, 2)
#     chosen = []
#     for r in selected_rings:
#         chosen_cam = random.choice(rings[r])
#         chosen.append({"ring_idx": r, "camera_name": chosen_cam.name})
#     return chosen

# if os.path.exists(plan_path):
#     try:
#         with open(plan_path, "r", encoding="utf-8") as f:
#             plan = json.load(f)
#         print(f"📝 发现现有 rotation_plan.json，将复用相机选择：{plan}")
#     except Exception as e:
#         print(f"⚠️  读取 rotation_plan.json 失败，将重新选择相机：{e}")
#         plan = pick_two_cameras_from_two_distinct_rings(camera_list)
#         with open(plan_path, "w", encoding="utf-8") as f:
#             json.dump(plan, f, ensure_ascii=False, indent=2)
# else:
#     plan = pick_two_cameras_from_two_distinct_rings(camera_list)
#     with open(plan_path, "w", encoding="utf-8") as f:
#         json.dump(plan, f, ensure_ascii=False, indent=2)
#     print(f"📝 已写入 rotation_plan.json：{plan}")

# --- 创建三环相机 ---
camera_rings_config = [(0.01, 0.005, 10), (0.015, 0.008, 10), (0.02, 0.012, 10)]
camera_list = []
create_cameras_rings(camera_list, bounds_min, bounds_max, main_obj, camera_rings_config)

# --- 基础命名与输出目录 ---
scene_name = os.path.splitext(os.path.basename(scene_blend_path))[0]
object_name = os.path.splitext(os.path.basename(object_blend_path))[0]
output_mask_dir = os.path.join(output_path, 'test_masks_cryptomatte2')
os.makedirs(output_mask_dir, exist_ok=True)

# --- 固定选择两台相机（Camera_1_9 与 Camera_0_8），并写入持久化计划文件 ---
plan_path = os.path.join(output_mask_dir, "rotation_plan.json")

def _get_cam_or_raise(name: str):
    cam = bpy.data.objects.get(name)
    if cam is None:
        raise RuntimeError(f"指定的相机不存在：{name}。请确认 camera_rings_config 或创建相机的命名规则未被修改。")
    return cam

# 固定相机名称
fixed_cam_names = ["Camera_1_9", "Camera_0_5"]
# fixed_cam_names = ["Camera_0_5"]
# 取回相机对象并确定 ring_idx（优先读取创建时写入的自定义属性；如无则从名称中解析）
selected_cameras = []
plan = []
for cam_name in fixed_cam_names:
    cam = _get_cam_or_raise(cam_name)
    if "ring_idx" in cam:
        ring_idx = int(cam["ring_idx"])
    else:
        # 兜底：从名称解析 "Camera_{ring}_{i}"
        try:
            _, ring_str, _ = cam_name.split("_", 2)
            ring_idx = int(ring_str)
        except Exception:
            raise RuntimeError(f"无法从相机名解析 ring_idx：{cam_name}，且对象未包含 'ring_idx' 属性。")
    selected_cameras.append(cam)
    plan.append({"ring_idx": ring_idx, "camera_name": cam_name})

# 校验两台相机是否来自不同环
if len(plan) != 2 or plan[0]["ring_idx"] == plan[1]["ring_idx"]:
    raise RuntimeError(f"固定相机需来自不同环，但得到的环索引为：{plan[0]['ring_idx']} 与 {plan[1]['ring_idx']}。")

# 将固定选择写入（覆盖） rotation_plan.json，便于留档与后续复核
with open(plan_path, "w", encoding="utf-8") as f:
    json.dump(plan, f, ensure_ascii=False, indent=2)
print(f"📝 已固定选择相机并写入 rotation_plan.json：{plan}")

# 将计划中的相机对象取回（校验存在性）
selected_cameras = []
for item in plan:
    cam_name = item["camera_name"]
    cam = bpy.data.objects.get(cam_name)
    if cam is None:
        raise RuntimeError(f"计划文件中的相机不存在：{cam_name}")
    selected_cameras.append(cam)

### === 生成总任务蓝图（仅旋转+平移两台相机） === ###
render_sequences_blueprint = {}  # key -> list of (position, rotation_euler, img_name)

def euler_with_local_z(base_euler, cw_deg):
    """基于对象初始姿态，绕对象局部Z顺时针旋转指定角度（负角度）。返回新的Euler(绝对角)。"""
    # 顺时针为负角度
    rad = _math.radians(-cw_deg)
    # 将旋转应用为相对局部Z：在Blender中，直接修改rotation_euler.z为绝对角即可（局部Z与Euler Z对齐）
    new_euler = Euler((base_euler.x, base_euler.y, base_euler.z + rad), base_euler.order)
    return new_euler

def add_init_frame_for_camera(cam):
    key = (cam, 'rot_init')
    render_sequences_blueprint[key] = []
    img_name = f"{scene_name}_{object_name}_{cam.name}_rot_init.png"
    render_sequences_blueprint[key].append((
        Vector(start_location),  # 位置
        Euler((start_rotation.x, start_rotation.y, start_rotation.z), start_rotation.order),  # 姿态
        img_name
    ))

def add_rotmove_sequences_for_camera(cam):
    # 12个方向：0,30,...,330（顺时针）
    for k in range(ROT_STEPS):
        cw_deg = k * ROT_DEG_STEP  # 0..330
        angle_label = f"{ANGLE_LABEL_STYLE}{cw_deg:03d}"  # e.g., 'cw030'
        key = (cam, f'rotmove_{angle_label}')
        render_sequences_blueprint[key] = []

        # 每个角度的3步，均从初始化点出发，并沿世界系 move_step 前进
        for step_idx in range(1, MOVE_STEPS_PER_ANGLE + 1):
            pos = Vector((
                start_location[0] + move_step[0] * step_idx,
                start_location[1] + move_step[1] * step_idx,
                start_location[2] + move_step[2] * step_idx
            ))
            rot = euler_with_local_z(start_rotation, cw_deg)
            img_name = f"{scene_name}_{object_name}_{cam.name}_rot{angle_label}_step{step_idx:02d}.png"
            render_sequences_blueprint[key].append((pos, rot, img_name))

# 仅为两台被选相机生成任务：每台相机 1个init + 12个角度×3步
for cam in selected_cameras:
    add_init_frame_for_camera(cam)
    add_rotmove_sequences_for_camera(cam)

### === 调用断点续传函数，获取待办任务列表 === ###
tasks_to_run, tasks_to_run_count = get_resume_tasks(render_sequences_blueprint, output_mask_dir)

# 如果待办列表为空，说明所有任务都已完成，直接退出
if not tasks_to_run:
    print("\n✅ 所有渲染任务均已完成！")
    bpy.ops.wm.quit_blender()

### === 开始渲染 === ###
print(f"\n--- 开始使用Cryptomatte渲染精确Mask (共 {tasks_to_run_count} 个任务) ---")

setup_compositor_for_cryptomatte_mask(scene, main_obj)

# 关闭灯光与背景，确保纯净Mask
original_light_energies = {light.name: light.energy for light in bpy.data.lights}
for light in bpy.data.lights: light.energy = 0.0
original_bg_strength = 0
if scene.world and scene.world.node_tree:
    bg_node = scene.world.node_tree.nodes.get('Background')
    if bg_node:
        original_bg_strength = bg_node.inputs['Strength'].default_value
        bg_node.inputs['Strength'].default_value = 0.0

rendered_count = 0

def set_obj_pose_and_ground(pos, rot_euler, ground):
    """设置位置与姿态并贴地。"""
    main_obj.location = pos
    main_obj.rotation_euler = rot_euler
    if ground is not None:
        adjust_object_to_ground_with_ray(main_obj, ground)

# 遍历每一个待办序列（保持插入顺序）
for key, tasks in tasks_to_run.items():
    cam, seq_tag = key
    print(f"\n--- 正在处理新序列: Camera '{cam.name}', Tag '{seq_tag}' ---")
    
    # 对于 rotmove_* 序列，确保每个角度都从初始化点出发（这里在每个任务帧内都直接用pos设定，不需额外reset）
    for i, (pos, rot_euler, img_name) in enumerate(tasks):
        rendered_count += 1

        # 姿态与位置，并贴地
        set_obj_pose_and_ground(pos, rot_euler, ground_obj)

        # 设定相机
        scene.camera = cam
        
        filepath = os.path.join(output_mask_dir, img_name)
        scene.render.filepath = filepath
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'BW'
        
        print(f"🎬 渲染精确Mask ({rendered_count}/{tasks_to_run_count}): {img_name}")
        bpy.ops.render.render(write_still=True)
        
        # --- 渲染后立即检查 ---
        if check_rendered_mask(filepath):
            print(f"  ❗️ 检测到无效Mask: {img_name}。物体可能已移出视野或过小。")
            print(f"  ⏭️  正在中止当前序列，跳过剩余的 {len(tasks) - (i + 1)} 帧。")
            try:
                os.remove(filepath)
                print(f"  🗑️  已删除无效文件: {filepath}")
            except OSError as e:
                print(f"  ⚠️  警告：删除文件失败 {filepath}: {e}")
            break

# 渲染后，恢复设置并清理
cleanup_compositor(scene)
for light_name, energy in original_light_energies.items():
    if light_name in bpy.data.lights:
        bpy.data.lights[light_name].energy = energy
if scene.world and scene.world.node_tree:
    bg_node = scene.world.node_tree.nodes.get('Background')
    if bg_node:
        bg_node.inputs['Strength'].default_value = original_bg_strength

print(f"\n✅ 精确Mask渲染完成！")
print(f"📁 Mask图像保存在: {output_mask_dir}")
print(f"📝 相机选择计划文件: {plan_path}")
