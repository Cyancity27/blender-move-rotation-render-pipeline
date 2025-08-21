BLENDER_EXEC="/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/blender/blender-4.2.4-linux-x64/blender"
PYTHON_SCRIPT="/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/bg_render/render_bg.py"
TASKS_FILE="/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/bg_render/mask_test.csv"
LOG_DIR="/gemini/platform/public/aigc/cv_banc/zhanghy56_intern/cjr/blender_cjr/blender/rotation_pipeline/logs/sculpture1/bg/"


export ORION_ENABLE_LPC=1
# ================================================
# 0. 预检查
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

mkdir -p "$LOG_DIR"

tail -n +2 "$TASKS_FILE" | while IFS=',' read -r scene_path object_path base_output_dir gpu_index
do
    # 跳过空行或注释行
    if [[ -z "$scene_path" || "$scene_path" == \#* ]]; then
        continue
    fi
    
    scene_name=$(basename "$scene_path" .blend)
    log_file="${LOG_DIR}/${scene_name}_bg.log"
    
    echo "========================================================"
    echo "🚀 正在分发渲染背景任务:"
    echo "   - 场景: $scene_name"
    echo "   - 分配GPU: $gpu_index"
    echo "   - 查看日志: tail -f $log_file"
    echo "========================================================"
    
    # 只传递scene、output、gpu-index参数，不传object参数
    nohup "$BLENDER_EXEC" -b -P "$PYTHON_SCRIPT" -- \
        --scene "$scene_path" \
        --object "$object_path" \
        --output "$base_output_dir" \
        --gpu-index "$gpu_index" > "$log_file" 2>&1 < /dev/null &
done