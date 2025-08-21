import bpy
import os
import math
from mathutils import Vector
from PIL import Image
import tempfile
import numpy as np
import argparse
import sys
import time
import json

def get_command_line_args():
    """
    解析从命令行传递给Blender脚本的参数。
    Blender在后台模式下运行时，会使用 '--' 将自身参数与Python脚本的参数分开。
    """
    argv = sys.argv
    if "--" not in argv:
        return None  # 在GUI模式下运行，无参数可解析
    
    # 获取 '--' 之后的所有参数进行解析
    args_to_parse = argv[argv.index("--") + 1:]

    parser = argparse.ArgumentParser(description="通过命令行参数控制的Blender背景渲染脚本。")
    
    # 定义需要从命令行接收的参数
    parser.add_argument("--gpu-index",
                        dest="target_gpu_index",
                        type=int,
                        required=True,
                        help="要使用的GPU设备索引 (必需)")
                        
    parser.add_argument("--object",
                        dest="object_blend_path",
                        type=str,
                        required=True,
                        help="用于确定相机和光照的物体.blend文件路径 (必需)")
                        
    parser.add_argument("--scene",
                        dest="scene_blend_path",
                        type=str,
                        required=True,
                        help="渲染所用的场景.blend文件路径 (必需)")
                        
    parser.add_argument("--output",
                        dest="output_path",
                        type=str,
                        required=True,
                        help="渲染输出的【完整】目标目录 (必需)")

    try:
        args = parser.parse_args(args=args_to_parse)
        return args
    except SystemExit:
        # 捕获argparse的退出行为，允许Blender正常关闭
        return None

# --- 执行参数解析并配置脚本变量 ---

# 1. 解析命令行参数
cli_args = get_command_line_args()

# 2. 检查参数有效性
if cli_args is None:
    print("❌ 错误：未检测到有效的命令行参数或解析失败。")
    print("   请在后台模式下运行，并提供所有必需的参数。")
    print("   示例: blender -b -P script.py -- --scene <path> --object <path> --output <path> --gpu-index 0")
    bpy.ops.wm.quit_blender()

# 3. 将解析到的参数赋值给脚本变量
target_gpu_index = cli_args.target_gpu_index
object_blend_path = cli_args.object_blend_path
scene_blend_path = cli_args.scene_blend_path
output_path = cli_args.output_path

# 确保输出目录存在
os.makedirs(output_path, exist_ok=True)

### === 读取 rotation_plan.json 并解析需要渲染的相机 === ###
def load_rotation_plan_required_cameras(base_output_dir):
    """
    从 base_output_dir/test_masks_cryptomatte2/rotation_plan.json 读取需要渲染的相机列表。
    JSON 期望为数组，每个元素形如：
    {
        "ring_idx": 1,
        "camera_name": "Camera_1_9"
    }
    返回：set(camera_name字符串)
    """
    plan_path = os.path.join(base_output_dir, "test_masks_cryptomatte2", "rotation_plan.json")
    print(f"🔎 尝试读取相机计划：{plan_path}")
    if not os.path.exists(plan_path):
        print("❌ 未找到 rotation_plan.json，已停止。请确保文件存在。")
        bpy.ops.wm.quit_blender()
        return set()

    try:
        with open(plan_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 读取 rotation_plan.json 失败：{e}")
        bpy.ops.wm.quit_blender()
        return set()

    if not isinstance(data, list) or len(data) == 0:
        print("❌ rotation_plan.json 内容为空或格式不是数组，已停止。")
        bpy.ops.wm.quit_blender()
        return set()

    required_names = set()
    for idx, entry in enumerate(data):
        cname = None
        if isinstance(entry, dict):
            # 优先使用 camera_name；若没有，也可从 ring_idx + 下标推断（但这里严格按 camera_name）
            cname = entry.get("camera_name", None)
        if not cname:
            print(f"⚠️ 第 {idx} 条记录缺少 'camera_name' 字段，已跳过：{entry}")
            continue
        required_names.add(str(cname))

    if len(required_names) == 0:
        print("❌ rotation_plan.json 未解析到任何有效的 camera_name，已停止。")
        bpy.ops.wm.quit_blender()
        return set()

    print(f"✅ 读取到 {len(required_names)} 个需要渲染的相机：{sorted(required_names)}")
    return required_names

required_camera_names = load_rotation_plan_required_cameras(output_path)

### === 工具函数 === ###
def direction_to_euler(direction_vec):
    target = Vector(direction_vec)
    rot_quat = target.to_track_quat('-Z', 'Y')
    return rot_quat.to_euler()

# ===== 1.py功能集成 =====
def get_scene_bounds():
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and not obj.hide_get():
            bbox_world = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
            for v in bbox_world:
                min_x = min(min_x, v.x)
                min_y = min(min_y, v.y)
                min_z = min(min_z, v.z)
                max_x = max(max_x, v.x)
                max_y = max(max_y, v.y)
                max_z = max(max_z, v.z)
    return Vector((min_x, min_y, min_z)), Vector((max_x, max_y, max_z))

def auto_adjust_object_size(obj, scene_bounds_min, scene_bounds_max):
    size = scene_bounds_max - scene_bounds_min
    avg_dim = (size.x + size.y + size.z) / 3
    target_size = avg_dim * 0.005
    current_size = max(obj.dimensions.x, obj.dimensions.y, obj.dimensions.z)
    if current_size == 0:
        print("❌ 物体尺寸为0，无法缩放。")
        return
    scale_factor = target_size / current_size
    obj.scale *= scale_factor
    print(f"📏 物体缩放因子: {scale_factor:.4f} (目标尺寸: {target_size:.2f}, 当前尺寸: {current_size:.2f})")
    # 应用缩放
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.select_set(False)

def adjust_lighting_to_target_brightness(target_brightness=0.4, preview_resolution=(128, 72), camera_list=None):
    # scene = bpy.context.scene
    # if not camera_list:
    #     print("⚠️ 自动调光警告：没有提供相机列表，跳过。")
    #     return

    # scene.camera = camera_list[0]
    # old_x, old_y = scene.render.resolution_x, scene.render.resolution_y
    # old_samples = scene.cycles.samples
    # scene.render.resolution_x, scene.render.resolution_y = preview_resolution
    # scene.cycles.samples = 32
    # scene.render.image_settings.file_format = 'PNG'
    # temp_file = os.path.join(tempfile.gettempdir(), f"preview_{os.getpid()}.png") # 使用进程ID确保临时文件唯一
    # scene.render.filepath = temp_file

    # print("💡 正在执行自动调光预览渲染...")
    # bpy.ops.render.render(write_still=True)

    # time.sleep(0.5) 

    # try:
    #     if not os.path.exists(temp_file) or os.path.getsize(temp_file) == 0:
    #         print(f"❌ 自动调光失败：预览文件 '{temp_file}' 未能成功创建或为空。")
    #         print("   请优先检查Blender版本与.blend文件版本的兼容性！")
    #         return

    #     with Image.open(temp_file) as img:
    #         avg_brightness = np.asarray(img.convert('RGB')).astype(np.float32).mean() / 255.0
        
    #     if avg_brightness > 0:
    #         scale = target_brightness / avg_brightness
    #         print(f"💡 自动调光: 当前平均亮度 {avg_brightness:.3f}, 目标亮度 {target_brightness:.3f}, 缩放因子 {scale:.3f}")
    #         for light in bpy.data.lights:
    #             light.energy *= scale
    #         if scene.world and scene.world.node_tree:
    #             for node in scene.world.node_tree.nodes:
    #                 if node.type == 'BACKGROUND':
    #                     node.inputs[1].default_value *= scale
    # except Exception as e:
    #     print(f"❌ 自动调光失败：读取预览图时出错: {e}")
    # finally:
    #     # 恢复原始设置并清理
    #     scene.render.resolution_x, scene.render.resolution_y = old_x, old_y
    #     scene.cycles.samples = old_samples
    #     if os.path.exists(temp_file):
    #         try:
    #             os.remove(temp_file)
    #         except OSError as e:
    #             print(f"⚠️ 无法删除临时文件: {e})")

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
        if obj.name.lower() == "ground":
            return obj
    lowest = float('inf')
    ground_obj = None
    for obj in candidates:
        bbox = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        z = min(v.z for v in bbox)
        if z < lowest:
            lowest = z
            ground_obj = obj
    return ground_obj

def fix_ground_orientation(ground_obj):
    if ground_obj.scale.z < 0:
        ground_obj.scale.z = abs(ground_obj.scale.z)
        print("⚠️ 修复地面负缩放")
    rx = math.degrees(ground_obj.rotation_euler.x)
    if abs(rx) in [90, 180, 270]:
        ground_obj.rotation_euler.x = 0.0
        print("⚠️ 修复地面旋转角度")

def adjust_object_to_ground_with_ray(obj, ground_obj):
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bbox_world = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_z = min(v.z for v in bbox_world)
    center_x = sum(v.x for v in bbox_world) / 8.0
    center_y = sum(v.y for v in bbox_world) / 8.0
    ray_origin = Vector((center_x, center_y, min_z + 0.01))
    ray_dir = Vector((0, 0, -1))
    hit, location, _, _, hit_obj, _ = scene.ray_cast(depsgraph, ray_origin, ray_dir, distance=2.0)
    if hit and hit_obj == ground_obj:
        delta_z = location.z - min_z
        obj.location.z += delta_z
        print(f"✅ 精确贴合地面，提升高度: {delta_z:.4f}")
    else:
        print("⚠️ 未命中地面")
# ===== 1.py功能集成结束 =====

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


# ### === 渲染设置 === ###
# scene = bpy.context.scene
# scene.render.engine = 'CYCLES'

# # GPU 设置
# prefs = bpy.context.preferences.addons['cycles'].preferences
# prefs.compute_device_type = 'CUDA'
# prefs.get_devices()

# cuda_devices = [d for d in prefs.devices if d.type == 'CUDA']
# if target_gpu_index >= len(cuda_devices):
#     raise IndexError("指定 GPU index 超出范围")
# for i, d in enumerate(cuda_devices):
#     d.use = (i == target_gpu_index)
#     print(f"{'✅ 启用' if d.use else '🚫 禁用'} GPU: {d.name}")

# # 高显存使用配置
# scene.cycles.device = 'GPU'
# scene.cycles.use_adaptive_sampling = True
# scene.cycles.adaptive_threshold = 0.01
# scene.cycles.samples = 512
# scene.cycles.use_denoising = True
# scene.cycles.denoiser = 'OPENIMAGEDENOISE'
# scene.cycles.use_bvh_spatial_split = True
# scene.cycles.debug_use_spatial_splits = False
# scene.cycles.debug_use_qbvh = False
# scene.cycles.use_progressive_refine = False
# scene.cycles.tile_size = 512

# # 曝光和色彩管理，防止过曝
# scene.view_settings.view_transform = 'Filmic'
# scene.view_settings.look = 'None'
# scene.view_settings.gamma = 0.5

# ### === 加载场景文件 === ###
# bpy.ops.wm.open_mainfile(filepath=scene_blend_path)
# scene = bpy.context.scene

# # 在这里设置分辨率，确保覆盖场景文件的设置
# scene.render.resolution_x = 1024
# scene.render.resolution_y = 1024
# scene.render.resolution_percentage = 100

# 降低所有灯光强度
for light in bpy.data.lights:
    light.energy *= 0.3

# 降低环境光（如果使用了背景贴图）
if scene.world and scene.world.node_tree:
    for node in scene.world.node_tree.nodes:
        if node.type == 'BACKGROUND':
            node.inputs[1].default_value *= 0.3

### === 导入物体（用于场景调整） === ###
with bpy.data.libraries.load(object_blend_path) as (data_from, data_to):
    data_to.objects = data_from.objects

# 导入后，自动选择所有 mesh 类型对象为主物体候选
imported_mesh_objs = []
for o in data_to.objects:
    if o:
        bpy.context.collection.objects.link(o)
        if o.type == 'MESH':
            imported_mesh_objs.append(o)
        print(f"导入对象: {o.name}, 类型: {o.type}")

if not imported_mesh_objs:
    raise RuntimeError("未找到任何 mesh 类型主物体")

# 合并所有 mesh 为一个主物体
for obj in imported_mesh_objs:
    obj.select_set(True)
bpy.context.view_layer.objects.active = imported_mesh_objs[0]
bpy.ops.object.join()
main_obj = bpy.context.view_layer.objects.active
print(f"合并后主物体: {main_obj.name}, 类型: {main_obj.type}")

# === 1.py功能：地面检测、修正、物体缩放、贴地 === #
ground_obj = find_ground_object_auto()
if ground_obj:
    fix_ground_orientation(ground_obj)
bounds_min, bounds_max = get_scene_bounds()
auto_adjust_object_size(main_obj, bounds_min, bounds_max)
if ground_obj:
    adjust_object_to_ground_with_ray(main_obj, ground_obj)

# ====== 三圈相机参数 ======
# 相机圈配置：每圈包含 (半径比例, 高度比例, 相机数量)
camera_rings_config = [
    (0.01, 0.005, 10),  
    (0.015, 0.008, 10),    
    (0.02, 0.012, 10),   
]

def create_cameras_rings(camera_list, bounds_min, bounds_max, target_obj, rings_config):
    """
    创建多圈相机
    """
    bbox_world = [target_obj.matrix_world @ Vector(corner) for corner in target_obj.bound_box]
    center = sum(bbox_world, Vector((0, 0, 0))) / 8.0
    scene_size = bounds_max - bounds_min
    max_span = max(scene_size.x, scene_size.y)
    scene_height = scene_size.z
    
    for ring_idx, (radius_ratio, height_ratio, cameras_count) in enumerate(rings_config):
        radius = max_span * radius_ratio
        height = scene_height * height_ratio
        
        print(f"🎥 创建第{ring_idx + 1}圈相机: 半径={radius:.2f}, 高度={height:.2f}, 数量={cameras_count}")
        
        for i in range(cameras_count):
            angle = 2 * math.pi * i / cameras_count
            x = center.x + radius * math.cos(angle)
            y = center.y + radius * math.sin(angle)
            z = center.z + height
            cam_loc = Vector((x, y, z))
            cam_dir = center - cam_loc
            
            cam_data = bpy.data.cameras.new(name=f"Camera_{ring_idx}_{i}")
            cam_obj = bpy.data.objects.new(name=f"Camera_{ring_idx}_{i}", object_data=cam_data)
            cam_obj.location = cam_loc
            cam_obj.rotation_euler = direction_to_euler(cam_dir)
            bpy.context.collection.objects.link(cam_obj)
            camera_list.append(cam_obj)

### === 创建三圈相机 === #
camera_list = []
create_cameras_rings(camera_list, bounds_min, bounds_max, main_obj, camera_rings_config)

# 仅保留 rotation_plan.json 中指定的相机
name_to_cam = {cam.name: cam for cam in camera_list}
missing = [n for n in required_camera_names if n not in name_to_cam]
if missing:
    print(f"⚠️ 警告：下列相机在场景中未创建，将被跳过：{missing}")

filtered_cameras = [name_to_cam[n] for n in required_camera_names if n in name_to_cam]
if len(filtered_cameras) == 0:
    print("❌ 未找到与 rotation_plan.json 匹配的任何相机，停止。")
    bpy.ops.wm.quit_blender()

# 自动调光仅使用被选择的相机列表
adjust_lighting_to_target_brightness(0.4, camera_list=filtered_cameras)

# 现在我们不再需要物体移动，也不需要渲染遮罩或记录CSV。
# 只需要让指定的相机各拍一张照片即可。

### === 准备背景图渲染 === ###
# 创建新的输出文件夹用于存放背景图
output_bg_dir = os.path.join(output_path, 'background_images')
os.makedirs(output_bg_dir, exist_ok=True)

# 定义场景名称，用于文件名
scene_name = os.path.splitext(os.path.basename(scene_blend_path))[0]

# 【核心步骤】在渲染前，将主物体设置为在渲染中不可见
print(f"\n🙈 隐藏主物体 '{main_obj.name}' 以便渲染背景图。")
main_obj.hide_render = True

### === 开始渲染背景图（仅 JSON 指定的相机） === ###
print(f"🎬 开始为 {len(filtered_cameras)} 个指定相机视角渲染背景图...")

for cam in filtered_cameras:
    # 设置当前场景的活动相机
    scene.camera = cam
    
    # 定义输出文件名
    img_name = f"bg_{scene_name}_{cam.name}.png"
    img_path = os.path.join(output_bg_dir, img_name)
    
    print(f"  -> 正在渲染: {img_name}")
    
    # 配置并执行渲染
    scene.render.filepath = img_path
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    bpy.ops.render.render(write_still=True)

print(f"\n✅ 背景图渲染完成！")
print(f"📁 所有背景图像保存在: {output_bg_dir}")
