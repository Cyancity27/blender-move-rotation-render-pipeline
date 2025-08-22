# OmniMoveEdit: Blender Rendering Dataset Pipeline (Translation / Rotation + Translation)

> Using Blender 4.2.4, this pipeline batch-renders high-quality object/scene assets (manually selected from BlenderKit) to create editing result datasets that adhere to **physical and visual consistency** (including lighting, shadows, etc.). It covers **translation** and **rotation + translation** tasks, while simultaneously rendering main images, background images, foreground binary masks, and comprehensive metadata records. Complementary scripts for statistics, cleaning, and data loading are also provided.

------

## ✨ Project Goals and Data Design

### 1) Translation Rendering

- **Object/Scene Scale**: 8 objects × 24 scenes (9 indoor + 15 outdoor).
- **Camera Setup**: For each object-scene pair, **3 rings of cameras**, with **10 cameras per ring**, totaling **30 cameras**.
- **Motion Trajectory**: Under each camera viewpoint, the object moves along the XY plane in the **x = y** direction:
  - **Forward 10 steps** + **Backward 10 steps** (up to 20 frames).
  - Frames outside the field of view are removed using a "viewport culling trick".
- **Pixel Resolution**: All images are **1024×1024**.
- **Synchronized Export**: Batch outputs **main images** (rendered results), **masks** (binary masks), **backgrounds** (target-free backgrounds), and **metadata** (details below).

### 2) Rotation + Translation Rendering

- **Object/Scene Scale**: 3 objects × 8 scenes.
- **Camera Setup**: For each object-scene pair, **2 cameras at different heights** (selected from the "30-camera set").
- **Motion and Rotation**:
  - Initial direction: Move forward **3 steps** along the current XY plane's **positive direction**;
  - Then, at the "initial position", **rotate 30°**, and move forward 3 steps; repeated for **12 directions (360°)**;
  - Per camera viewpoint: **3×12 + 1 = 37** frames (including initial).
  - Frames outside the field of view are similarly removed via the "viewport culling trick".
- **Pixel Resolution**: **1024×1024**.
- **Synchronized Export**: Main images / masks / backgrounds / metadata.

> **Metadata Fields Example** (excerpt):
>  `image_name, scene_size, light_name, light_pos, light_strength, light_direction, camera_pos, camera_direction, camera_focal_length, camera_sensor_width, camera_principal_point, object_name, object_size, object_pos, object_rotation`
>  Records key geometric/lighting/camera/object pose information for each rendered frame, facilitating downstream training and analysis.

------

## 🗂️ Repository Structure Overview

```
bash复制blender-move-rotation-render-pipeline/
├─ .gitattributes
├─ blender_pipeline/ # Translation Rendering Pipeline
│ ├─ render_bg/
│ │ ├─ render_bg.csv # CSV columns (task queue): scene_path,object_path,base_output_dir,gpu_index
│ │ ├─ render_bg.py # Batch rendering of background images (without target object)
│ │ └─ render_bg.sh # Blender batch processing script invocation
│ ├─ render_main/
│ │ ├─ mask_indoor.csv # Indoor scene list/annotations
│ │ ├─ mask_outdoor.csv # Outdoor scene list/annotations
│ │ ├─ render_main_test.py # Main image rendering script (translation/outdoor)
│ │ ├─ render_main_test_in.py # Main image rendering script (translation/indoor)
│ │ └─ render_main_test.sh # Main image rendering batch processing
│ ├─ render_mask/
│ │ ├─ mask_indoor.csv
│ │ ├─ mask_outdoor.csv
│ │ ├─ render_mask2_10step_in.py # Indoor: Mask rendering along x=y direction ±10 steps
│ │ ├─ render_mask2_10step_out.py # Outdoor: Mask rendering along x=y direction ±10 steps
│ │ ├─ render_indoor.sh
│ │ └─ render_outdoor.sh
│ └─ utils/
│   ├─ count_main_img.py # Main image counting/quality checks
│   ├─ find_target_remove.py # Out-of-view frame detection and removal
│   ├─ index_construct.py # Sample pair/index construction
│   └─ pipeline_dataset.py # Dataset class/loading tools
├─ data/
│ ├─ models/
│ │ └─ sculpture4.blend # Example model (binary)
│ └─ test_scene/
│   └─ indoor6.blend # Example scene (binary)
└─ rotation_pipeline/ # Rotation + Translation Rendering Pipeline
  ├─ bg_render/
  │ ├─ render_bg.py
  │ ├─ render_bg.sh
  │ ├─ render_bg_test.csv
  │ ├─ mask_test.csv
  │ ├─ mask_test2.csv
  │ └─ mask_test3.csv
  ├─ image_render/
  │ ├─ render_main.py # Rotation + translation main image rendering (for object1)
  │ ├─ render_main2.py # Rotation + translation main image rendering (for object2)
  │ ├─ render_main3.py # Rotation + translation main image rendering (for object3)
  │ ├─ render_main_test.sh
  │ ├─ mask_test.csv
  │ ├─ mask_test2.csv
  │ ├─ mask_test3.csv
  ├─ mask_render/
  │ ├─ render_mask.sh
  │ ├─ render_mask_rotation.py # Rotation + translation: Mask rendering (for object1)
  │ ├─ render_mask_rotation2.py # Rotation + translation: Mask rendering (for object2)
  │ ├─ render_mask_rotation3.py # Rotation + translation: Mask rendering (for object3)
  │ ├─ mask_test.csv
  │ ├─ mask_test2.csv
  │ └─ mask_test3.csv
  └─ utils/
    ├─ count_main_img.py
    ├─ dataloader_output.py
    ├─ find_target_remove.py
    ├─ index_construct.py
    └─ pipeline_dataset.py
```

## 🔧 Environment and Dependencies

- **Blender**: Recommended `blender-4.2.4-linux-x64`.
- **Rendering**: Cycles (GPU recommended); configure GPU devices (CUDA/OptiX/Metal, etc.).
- **Python**: Uses Blender's built-in Python (invoking `bpy`).
- **Assets**: High-quality objects/scenes manually curated from sources like BlenderKit; organized by absolute/relative paths in CSVs.
- **VRAM/Storage**: Suggest 12GB+ VRAM; allocate disk space based on scale (including PNG main images, masks, backgrounds, and metadata).

------

## 🛠️ Rendering Pipeline Details and Script Responsibilities

> Note: The following provides a detailed walkthrough of the **typical workflow** for scripts, facilitating unified usage and secondary development by users. Script implementations rely on `bpy` and our custom path conventions.

### A. Translation Rendering (`blender_pipeline/`)

#### 1) Background Rendering (`render_bg/`)

- **`render_bg.csv`**: Four columns
  - `scene_path`: Path to `.blend` scene file
  - `object_path`: Path to `.blend` object/asset file
  - `base_output_dir`: Output root directory
  - `gpu_index`: GPU index (for multi-GPU scheduling)
- **`render_bg.py`**: Reads `render_bg.csv` row by row, for each (scene, object) pair:
  - Loads `scene.blend` (link/append);
  - **Excludes target object**, renders pure background;
  - Sets rendering engine, resolution (1024×1024), samples, denoising, etc.;
  - Iterates over camera poses (preset in scene or script-generated), outputs to `base_output_dir/bg/`;
  - Records `camera_*`, `light_*`, etc., metadata.
- **`render_bg.sh`**: Invokes the script for batch tasks based on settings in `render_bg.csv`.

#### 2) Mask Rendering (`render_mask/`, Translation)

- **CSVs**: `mask_indoor.csv`, `mask_outdoor.csv` (indoor/outdoor scene lists).
- **`render_mask2_10step_in.py` / `render_mask2_10step_out.py`**:
  - Places target object into the scene;
  - Under each camera viewpoint, performs **±10 steps** translation along **x=y** direction;
  - Outputs **binary foreground** (object=1, background=0) using material/visibility isolation;
  - Culls out-of-view frames (see utils detection logic).
- **`render_indoor.sh` / `render_outdoor.sh`**: Batch entry points for indoor/outdoor (core logic is similar).

#### 3) Main Image Rendering (`render_main/`, Translation)

- **`render_main_test.py` / `render_main_test_in.py`**:
  - Same steps as masks (x=y direction ±10 steps);
  - Uses **realistic lighting**, shadows, and materials to output physically consistent main images (PNG);
  - Naming convention example: `indoor1_sculpture1_Camera_0_0_forward_01.png`;
  - Synchronously writes metadata (camera/lighting/object pose, size, etc.).
- **`render_main_test.sh`**: Batch execution entry.

#### 4) Utilities (`utils/`)

- **`find_target_remove.py`**: Detects **if target is in view** based on **bounding box / pixel ratio / frustum clipping** rules, marks or removes invalid frames (corresponding to the "culling trick").
- **`count_main_img.py`**: Counts main images, category/scene distribution, and detects missing frames.
- **`index_construct.py`**: Organizes (object, scene, camera, step) into **trainable indices/sample pairs** (e.g., current position ↔ target position).
- **`pipeline_dataset.py`**: Unified Dataset / DataLoader (for loading main images, masks, backgrounds, and metadata during training/evaluation).

------

### B. Rotation + Translation Rendering (`rotation_pipeline/`)

#### 1) Background Rendering (`bg_render/`)

- **`render_bg.py` / `render_bg.sh`**: Consistent with translation pipeline, outputs to `bg/`.
- **Various CSVs**: `render_bg_test.csv`, `mask_test.csv`, etc., for quick testing different scene combinations.

#### 2) Mask Rendering (`mask_render/`, Rotation + Translation)

- **`render_mask_rotation.py` / `render_mask_rotation2.py` / `render_mask_rotation3.py`**:
  - Outputs 1 initial frame mask at **starting pose**;
  - Enumerates **12 directions** around 360° with **30° angular steps**, then **forward 3 steps** per direction;
  - Outputs total **37 frames/camera** masks; culls out-of-view frames;
  - Supports indoor/outdoor and multiple CSV lists.
- **`render_mask.sh`**: Batch execution entry.

#### 3) Main Image Rendering (`image_render/`)

- **`render_main.py` / `render_main2.py` / `render_main3.py`**:
  - Exports main images along the same trajectory as masks; multiple versions for **comparing camera sampling/lighting settings/rotation details** or tuning;
  - Output resolution 1024×1024, naming/metadata consistent with translation section;
  - `render_main_test.sh`: Batch entry (selectable sample subsets via CSV).
- **CSVs**: `mask_outdoor.csv`, `mask_test*.csv`, etc.

#### 4) Utilities (`utils/`)

- **`dataloader_output.py`**:
  - Data loaders and sample pairing tailored for rotation + translation tasks (e.g., initial pose → post-rotation target pose).
- **`count_main_img.py`, `find_target_remove.py`, `index_construct.py`, `pipeline_dataset.py`**: Same as translation logic, adapted for 37 frames/camera indexing.

------

## 🧾 Metadata Fields Documentation

- **File Naming** (Example):

  ```
  indoor1_sculpture1_Camera_0_0_forward_01.png
  ```

  - `indoor1`: Scene ID
  - `sculpture1`: Object ID
  - `Camera_0_0`: Camera ring / index
  - `forward_01`: 1st step forward along trajectory (or back_XX)

- **Metadata Fields**:
   `image_name, scene_size, light_name, light_pos, light_strength, light_direction, camera_pos, camera_direction, camera_focal_length, camera_sensor_width, camera_principal_point, object_name, object_size, object_pos, object_rotation`

  | Field Name               | Meaning/Unit      | Description                                              |
  | ------------------------ | ----------------- | -------------------------------------------------------- |
  | `image_name`             | Filename          | Corresponds one-to-one with main/mask images             |
  | `scene_size`             | `(sx, sy, sz)`    | Scene bounding box or size in world coordinates (meters) |
  | `light_name`             | Light type name   | E.g., `Area`, `Area.001`                                 |
  | `light_pos`              | `(x, y, z)`       | Light source world coordinates                           |
  | `light_strength`         | Scalar            | Light intensity (render engine units)                    |
  | `light_direction`        | Direction/angle   | N/A for non-directional lights                           |
  | `camera_pos`             | `(x, y, z)`       | Camera world coordinates                                 |
  | `camera_direction`       | Angle vector      | In radians (unit vector)                                 |
  | `camera_focal_length`    | mm                | Focal length                                             |
  | `camera_sensor_width`    | mm                | Sensor width                                             |
  | `camera_principal_point` | Pixel coordinates | E.g., `(512, 512)`                                       |
  | `object_name`            | Name              | E.g., `Brother Sculpture`                                |
  | `object_size`            | `(sx, sy, sz)`    | Object dimensions                                        |
  | `object_pos`             | `(x, y, z)`       | Object world coordinates                                 |
  | `object_rotation`        | Radians           | Object rotation angle vector                             |

  

## 🚀 Quick Start

1. **Environment Setup**:

   - Download and install Blender 4.2.4, ensuring GPU support (Cycles + OptiX/CUDA/Metal).

     ```
     git clone https://github.com/Cyancity27/blender-move-rotation-render-pipeline.git
     ```

2. **Asset Collection**: Organize selected `.blend` scene and object files in `./data` or a custom directory.

3. **Edit CSVs**: Use `blender_pipeline/render_bg/render_bg.csv` as a template, filling in:

   - `scene_path,object_path,base_output_dir,gpu_index`

4. **Render Backgrounds** (Optional):

   ```
   bash blender_pipeline/render_bg/render_bg.sh
   ```

   Customize parameters: `BLENDER_EXEC` (Blender executable path), `PYTHON_SCRIPT` (background rendering script path), `TASKS_FILE` (CSV path for object-scene tasks), `LOG_DIR` (log output path).

5. **Render Masks** (Translation or Rotation + Translation - Critical first step; main images depend on masks):

   - To efficiently cull object-out-of-view scenarios, we **render masks first**. Based on mask image analysis, we skip remaining sequences once the object exits view, maximizing time savings. Finally, we parse rendered mask filenames and render corresponding main images from task CSVs, ensuring objects are always in-frame.

   1. Translation (Indoor/Outdoor):

      ```
      bash复制bash blender_pipeline/render_mask/render_indoor.sh
      bash blender_pipeline/render_mask/render_outdoor.sh
      ```

   2. Rotation + Translation:

      ```
      bash rotation_pipeline/mask_render/render_mask.sh
      ```

6. **Render Main Images**:

   - Translation:

     ```
     bash blender_pipeline/render_main/render_main_test.sh
     ```

   - Rotation + Translation:

     ```
     bash rotation_pipeline/image_render/render_main_test.sh
     ```

7. **Cleaning and Statistics** (Optional):

   ```
   # Out-of-view frame removal
   python blender_pipeline/utils/find_target_remove.py 
   # Main image counting/missing frame inspection
   python blender_pipeline/utils/count_main_img.py 
   # Sample indexing/pairing
   python blender_pipeline/utils/index_construct.py 
   # Dataset loading check
   python blender_pipeline/utils/pipeline_dataset.py
   ```

> 💡 **Batch Parallelization**: Use `gpu_index` for multi-machine/multi-GPU scheduling of CSV rows.

## Additional Notes

- In the bg/mask/image rendering scripts, we implement checkpoint resuming. If a rendering sequence interrupts due to mesh loss, insufficient VRAM, or other issues, restarting the script will auto-detect progress and continue from the last rendered frame, avoiding restarts from scratch.
