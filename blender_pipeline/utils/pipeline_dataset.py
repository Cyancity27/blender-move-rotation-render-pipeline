import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image, ImageDraw
import json
import os
import shutil

# ========================================================================
#  1. 辅助函数与单个数据集类
# ========================================================================

def get_bbox_from_mask(mask_tensor):
    if mask_tensor.dim() == 3:
        mask_tensor = mask_tensor.squeeze(0)
    non_zero_indices = torch.nonzero(mask_tensor, as_tuple=False)
    if non_zero_indices.numel() == 0:
        return None
    y_indices = non_zero_indices[:, 0]
    x_indices = non_zero_indices[:, 1]
    x_min, y_min = torch.min(x_indices), torch.min(y_indices)
    x_max, y_max = torch.max(x_indices), torch.max(y_indices)
    return (x_min.item(), y_min.item(), x_max.item(), y_max.item())

class ObjectMovementDataset(Dataset):
    def __init__(self, meta_file, image_dir, mask_dir, image_size=(512, 512)):
        print(f"正在从 {meta_file} 加载元数据...")
        with open(meta_file, 'r') as f:
            self.all_pairs = json.load(f)
        print(f"已加载 {len(self.all_pairs)} 个对: {meta_file}")

        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.all_pairs)

    def __getitem__(self, idx):
        pair_info = self.all_pairs[idx]
        src_img_name = pair_info['source_image']
        tgt_img_name = pair_info['target_image']

        src_img_path = os.path.join(self.image_dir, src_img_name)
        tgt_img_path = os.path.join(self.image_dir, tgt_img_name)
        src_mask_path = os.path.join(self.mask_dir, src_img_name)
        tgt_mask_path = os.path.join(self.mask_dir, tgt_img_name)

        src_image_pil = Image.open(src_img_path).convert("RGB")
        tgt_image_pil = Image.open(tgt_img_path).convert("RGB")
        src_mask_pil = Image.open(src_mask_path).convert("L")
        tgt_mask_pil = Image.open(tgt_mask_path).convert("L")
        
        I1 = self.image_transform(src_image_pil)
    
        target_size = self.image_transform.transforms[0].size
        src_image_pil_resized = transforms.Resize(target_size)(src_image_pil)
        src_image_tensor_resized = transforms.ToTensor()(src_image_pil_resized)
        src_mask_tensor = self.mask_transform(src_mask_pil)
        I2_tensor = src_image_tensor_resized * src_mask_tensor
        I2 = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(I2_tensor)

        H, W = self.mask_transform.transforms[0].size
        instruction_map = torch.zeros(2, H, W)
        
        src_bbox = get_bbox_from_mask(src_mask_tensor)
        tgt_bbox = get_bbox_from_mask(self.mask_transform(tgt_mask_pil))

        if src_bbox:
            x_min, y_min, x_max, y_max = src_bbox
            instruction_map[0, y_min:y_max+1, x_min:x_max+1] = 1.0
        if tgt_bbox:
            x_min, y_min, x_max, y_max = tgt_bbox
            instruction_map[1, y_min:y_max+1, x_min:x_max+1] = 1.0
        I3 = instruction_map

        clean_target = self.image_transform(tgt_image_pil)

        noise = torch.randn_like(clean_target)
        time_step = torch.randint(0, 1000, (1,)).item()
        noisy_target = clean_target + noise * (time_step / 1000.0)
        noisy_target = torch.clamp(noisy_target, -1.0, 1.0)

        return {
            "I1_source_image": I1, 
            "I2_object_image": I2, 
            "I3_instruction_map": I3,
            "clean_target": clean_target, 
            "noisy_target": noisy_target, 
            "time_step": time_step, 
            "source_mask": src_mask_tensor 
        }

# ========================================================================
#  2. 配置多数据集路径并合并为一个大数据集
# ========================================================================

DATASET_GROUPS = [
    "/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/outputs/sculpture1/output_4_sculpture1",
    "/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/outputs/sculpture2/output_4_sculpture2",
    "/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/outputs/sculpture3/output_4_sculpture3",
    "/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/outputs/sculpture5/output_4_sculpture5",
    "/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/outputs/sculpture6/output_4_sculpture6"
]

datasets = []
for group in DATASET_GROUPS:
    image_dir = os.path.join(group, "images")
    mask_dir = os.path.join(group, "mask")
    meta_file = os.path.join(group, "dataset_pairs.json")
    if not (os.path.exists(image_dir) and os.path.exists(mask_dir) and os.path.isfile(meta_file)):
        print(f"警告：{group} 目录缺少 images、mask 或 json，跳过。")
        continue
    dataset = ObjectMovementDataset(
        meta_file=meta_file,
        image_dir=image_dir,
        mask_dir=mask_dir
    )
    datasets.append(dataset)

# 合并所有数据集
if len(datasets) == 0:
    raise RuntimeError("没有找到可用的数据集！")
elif len(datasets) == 1:
    all_dataset = datasets[0]
else:
    all_dataset = ConcatDataset(datasets)

print(f"已整合 {len(datasets)} 个数据子集，总样本数：{len(all_dataset)}")

# ========================================================================
#  3. DataLoader和可视化测试
# ========================================================================

BATCH_SIZE = 1 
NUM_SAMPLES_TO_VISUALIZE = 100

data_loader = DataLoader(
    all_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


OUTPUT_VIS_DIR = "test_output2"
if os.path.exists(OUTPUT_VIS_DIR):
    print(f"发现旧的测试目录 '{OUTPUT_VIS_DIR}'，正在删除...")
    shutil.rmtree(OUTPUT_VIS_DIR)
os.makedirs(OUTPUT_VIS_DIR)
print(f"已创建空的测试目录: '{OUTPUT_VIS_DIR}'")

to_pil = transforms.ToPILImage()
def save_tensor_image(tensor, path, denorm=True):
    if denorm:
        tensor = tensor.clone() * 0.5 + 0.5
    tensor = torch.clamp(tensor, 0, 1)
    pil_img = to_pil(tensor)
    pil_img.save(path)

def save_object_image_with_white_bg(object_tensor, mask_tensor, path):
    object_tensor_denorm = object_tensor.clone() * 0.5 + 0.5
    object_tensor_denorm = torch.clamp(object_tensor_denorm, 0, 1)
    white_bg = torch.ones_like(object_tensor_denorm)
    mask_3_channels = mask_tensor.expand_as(object_tensor_denorm)
    final_image_tensor = object_tensor_denorm * mask_3_channels + white_bg * (1 - mask_3_channels)
    pil_img = to_pil(final_image_tensor)
    pil_img.save(path)

def save_instruction_map_outlines(instruction_tensor, path):
    src_mask = instruction_tensor[0]
    tgt_mask = instruction_tensor[1]
    src_bbox = get_bbox_from_mask(src_mask)
    tgt_bbox = get_bbox_from_mask(tgt_mask)
    h, w = instruction_tensor.shape[1], instruction_tensor.shape[2]
    canvas = Image.new('RGB', (w, h), 'white')
    draw = ImageDraw.Draw(canvas)
    if tgt_bbox:
        draw.rectangle(tgt_bbox, outline="blue", width=2)
    if src_bbox:
        draw.rectangle(src_bbox, outline="red", width=2)
    canvas.save(path)

print(f"\n🚀 开始提取并保存 {NUM_SAMPLES_TO_VISUALIZE} 个样本进行可视化...")

for i, batch in enumerate(data_loader):
    if i >= NUM_SAMPLES_TO_VISUALIZE:
        break
    sample_dir = os.path.join(OUTPUT_VIS_DIR, f"sample_{i}")
    os.makedirs(sample_dir)
    i1 = batch["I1_source_image"][0]
    i2 = batch["I2_object_image"][0]
    i3 = batch["I3_instruction_map"][0]
    noisy_target = batch["noisy_target"][0]
    source_mask = batch["source_mask"][0]
    save_tensor_image(i1, os.path.join(sample_dir, "1_source_image.png"))
    save_object_image_with_white_bg(i2, source_mask, os.path.join(sample_dir, "2_object_image.png"))
    save_instruction_map_outlines(i3, os.path.join(sample_dir, "3_instruction_map.png"))
    save_tensor_image(noisy_target, os.path.join(sample_dir, "4_noisy_target.png"))
    print(f"  -> 已保存样本 {i} 到 '{sample_dir}'")

print(f"\n 可视化完成，全都在 '{OUTPUT_VIS_DIR}' 目录下")
