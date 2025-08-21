BLENDER_EXEC="/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/blender/blender-4.2.4-linux-x64/blender"
PYTHON_SCRIPT="/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/image_render/render_main3.py"
TASKS_FILE="/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/image_render/mask_test3.csv"
LOG_DIR="/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/logs/sculpture6/images/"

export ORION_ENABLE_LPC=1
# ================================================
# --- 脚本主体（通常无需修改） ---
# 0. 预检查，确保关键文件存在
if [ ! -f "$BLENDER_EXEC" ]; then
    echo "❌ 错误: Blender可执行文件未找到于 '$BLENDER_EXEC'"
    exit 1
fi
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ 错误: Python渲染脚本未找到于 '$PYTHON_SCRIPT'"
    exit 1
fi
if [ ! -f "$TASKS_FILE" ]; then
    echo "❌ 错误: 任务文件 '$TASKS_FILE' 未在当前目录找到"
    exit 1
fi

# 1. 确保日志目录存在
mkdir -p "$LOG_DIR"
# 2. 读取CSV文件并分发任务
#    - 使用 `tail -n +2` 跳过CSV文件的第一行（表头）。
#    - 设置 IFS=',' 让 `read` 命令以逗号为分隔符。
tail -n +2 "$TASKS_FILE" | while IFS=',' read -r scene_path object_path base_output_dir gpu_index

do  
    echo "开始执行"
    if [[ -z "$scene_path" || "$scene_path" == \#* ]]; then
        continue
    fi
    
    # 3. 【统一规范的日志输出】
    #    从文件路径中提取基础名，创建具有描述性的日志文件名。
    #    例如：scene9_sculpture4_gpu0.log
    scene_name=$(basename "$scene_path" .blend)
    object_name=$(basename "$object_path" .blend)
    log_file="${LOG_DIR}/${scene_name}_${object_name}.log"
    
    echo "========================================================"
    echo "🚀 正在分发渲染任务:"
    echo "   - 场景: $scene_name"
    echo "   - 物体: $object_name"
    echo "   - 分配GPU: $gpu_index"
    echo "   - 查看日志: tail -f $log_file"
    echo "========================================================"
    
    # 4. 调用Blender执行渲染
    #    - 使用 `nohup` 和 `&` 使其在后台持久运行。
    #    - 通过 `--` 将参数传递给Python脚本。
    #    - `> "$log_file" 2>&1` 将所有输出（标准和错误）重定向到规范的日志文件。
    nohup "$BLENDER_EXEC" -b -P "$PYTHON_SCRIPT" -- \
        --scene "$scene_path" \
        --object "$object_path" \
        --output "$base_output_dir" \
        --gpu-index "$gpu_index" > "$log_file" 2>&1 < /dev/null &

done
