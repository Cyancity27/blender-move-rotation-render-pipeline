# 脚本2: Mask渲染脚本 (render_mask.py) - Cryptomatte专业版 (智能跳过优化版)
# 功能：使用Cryptomatte通道生成精确的二值Mask，并能智能检测并跳过物体移出视野后的无效渲染序列。

import bpy
import os
import math
from mathutils import Vector
import argparse
import sys
import numpy as np
from PIL import Image

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
num_frames = 10
move_step = [0.045, 0.045, 0]

### === 工具函数 (与主图脚本保持一致) === ###
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
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bbox_world = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    min_z = min(v.z for v in bbox_world)
    center_xy = sum((Vector((v.x, v.y, 0)) for v in bbox_world), Vector()) / 8.0
    ray_origin = Vector((center_xy.x, center_xy.y, min_z + 0.01))
    hit, loc, *_ = bpy.context.scene.ray_cast(depsgraph, ray_origin, Vector((0, 0, -1)))
    if hit: obj.location.z += loc.z - min_z

def create_cameras_rings(camera_list, bounds_min, bounds_max, target_obj, rings_config):
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
            bpy.context.collection.objects.link(cam_obj)
            camera_list.append(cam_obj)

### === 核心改进：检查函数 (从您的逻辑中提取) === ###
def is_special_mask(bin_mask, area_threshold=0.005):
    """判断一个二值化掩码是否为“特殊”情况（全黑，或面积过小且触碰边缘）。"""
    total_pixels = bin_mask.size
    white_pixels = np.sum(bin_mask)

    # 条件1：图像是否完全为黑
    if white_pixels == 0:
        return True
    
    # 条件2：白色区域面积是否过小
    is_small_area = (white_pixels / total_pixels) < area_threshold
    if is_small_area:
        # 条件3：如果面积过小，是否还触碰了图像的任意一边
        touches_top = np.any(bin_mask[0, :])
        touches_bottom = np.any(bin_mask[-1, :])
        touches_left = np.any(bin_mask[:, 0])
        touches_right = np.any(bin_mask[:, -1])
        if touches_top or touches_bottom or touches_left or touches_right:
            return True
            
    return False

def check_rendered_mask(image_path, area_threshold=0.005):
    """读取渲染后的mask图片，进行检查，返回True表示是特殊/无效mask。"""
    try:
        with Image.open(image_path) as img:
            # 转换为灰度并二值化为0/1数组
            img_array = np.array(img.convert('L'))
            bin_mask = (img_array > 180).astype(np.uint8) # 大于128的为1，否则为0
            
            if is_special_mask(bin_mask, area_threshold):
                return True # 是特殊mask
        return False # 是正常mask
    except Exception as e:
        print(f"⚠️  警告：检查图片时发生错误 {image_path}: {e}")
        return False # 检查失败时，默认其为正常，避免误删

### === 合成器与场景加载等设置 (保持不变) === ###
def setup_compositor_for_cryptomatte_mask(scene, object_to_mask):
    scene.use_nodes = True
    tree = scene.node_tree
    for node in tree.nodes: tree.nodes.remove(node)
    render_layers = tree.nodes.new(type='CompositorNodeRLayers')
    composite_node = tree.nodes.new(type='CompositorNodeComposite')
    cryptomatte_node = tree.nodes.new(type='CompositorNodeCryptomatteV2')
    cryptomatte_node.matte_id = object_to_mask.name
    tree.links.new(render_layers.outputs['CryptoObject00'], cryptomatte_node.inputs['Image'])
    tree.links.new(cryptomatte_node.outputs['Matte'], composite_node.inputs['Image'])

def cleanup_compositor(scene):
    scene.use_nodes = False

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

ground_obj = find_ground_object_auto()
if ground_obj: fix_ground_orientation(ground_obj)
bounds_min, bounds_max = get_scene_bounds()
auto_adjust_object_size(main_obj, bounds_min, bounds_max)
if ground_obj: adjust_object_to_ground_with_ray(main_obj, ground_obj)
start_location = list(main_obj.location)

camera_rings_config = [(0.01, 0.005, 10), (0.015, 0.008, 10), (0.02, 0.012, 10)]
camera_list = []
create_cameras_rings(camera_list, bounds_min, bounds_max, main_obj, camera_rings_config)

### === 核心改进：重构渲染任务为序列 === ###
scene_name = os.path.splitext(os.path.basename(scene_blend_path))[0]
object_name = os.path.splitext(os.path.basename(object_blend_path))[0]
output_mask_dir = os.path.join(output_path, 'test_masks_cryptomatte2')
os.makedirs(output_mask_dir, exist_ok=True)

render_sequences = {} # 使用字典来组织任务序列
total_tasks = 0

# 生成前向移动任务
for cam in camera_list:
    key = (cam, 'forward')
    render_sequences[key] = []
    for frame in range(num_frames):
        pos = (start_location[0] + move_step[0] * frame, start_location[1] + move_step[1] * frame, start_location[2] + move_step[2] * frame)
        img_name = f"{scene_name}_{object_name}_{cam.name}_forward_{frame+1:02d}.png"
        render_sequences[key].append((pos, img_name))
        total_tasks += 1

# 生成反向移动任务
for cam in camera_list:
    key = (cam, 'reverse')
    render_sequences[key] = []
    for frame in range(num_frames):
        pos = (start_location[0] - move_step[0] * frame, start_location[1] - move_step[1] * frame, start_location[2] - move_step[2] * frame)
        img_name = f"{scene_name}_{object_name}_{cam.name}_reverse_{frame+1:02d}.png"
        render_sequences[key].append((pos, img_name))
        total_tasks += 1

### === 核心改进：带有提前退出的渲染循环 === ###
print("--- 开始使用Cryptomatte渲染精确Mask (智能跳过模式) ---")

setup_compositor_for_cryptomatte_mask(scene, main_obj)

original_light_energies = {light.name: light.energy for light in bpy.data.lights}
for light in bpy.data.lights: light.energy = 0.0
original_bg_strength = 0
if scene.world and scene.world.node_tree:
    bg_node = scene.world.node_tree.nodes.get('Background')
    if bg_node:
        original_bg_strength = bg_node.inputs['Strength'].default_value
        bg_node.inputs['Strength'].default_value = 0.0

rendered_count = 0
# 遍历每一个序列 (每个相机 + 每个方向 构成一个序列)
for (cam, direction), tasks in render_sequences.items():
    print(f"\n--- 正在处理新序列: Camera '{cam.name}', Direction '{direction}' ---")
    
    # 遍历序列中的每一帧
    for i, (pos, img_name) in enumerate(tasks):
        rendered_count += 1
        main_obj.location = pos
        scene.camera = cam
        
        filepath = os.path.join(output_mask_dir, img_name)
        scene.render.filepath = filepath
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'BW'
        
        print(f"🎬 渲染精确Mask ({rendered_count}/{total_tasks}): {img_name}")
        bpy.ops.render.render(write_still=True)
        
        # --- 渲染后立即检查 ---
        if check_rendered_mask(filepath):
            print(f"  ❗️ 检测到无效Mask: {img_name}。物体可能已移出视野。")
            print(f"  ⏭️  正在中止当前序列，跳过剩余的 {len(tasks) - (i + 1)} 帧。")
            
            # 删除这张触发条件的无效图片
            try:
                os.remove(filepath)
                print(f"  🗑️  已删除无效文件: {filepath}")
            except OSError as e:
                print(f"  ⚠️  警告：删除文件失败 {filepath}: {e}")
            
            # 跳出当前序列的循环
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
