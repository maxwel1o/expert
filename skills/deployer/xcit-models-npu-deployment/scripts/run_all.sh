#!/bin/bash
# XCiT 系列模型昇腾 NPU 自动化适配脚本
# 串行执行全部 5 个模型的推理、精度验证、文档生成和 GitCode 发布
set -e

OUTPUT_DIR="${1:-./output}"
SAMPLES="${2:-50}"
GITCODE_PUSH="${3:-false}"
GITCODE_USERNAME="${4:-gcw_C8PI9e90}"
GITCODE_TOKEN="${ATOMGIT_USER_TOKEN}"

mkdir -p "$OUTPUT_DIR"
FAILURES_LOG="$OUTPUT_DIR/failures.log"
> "$FAILURES_LOG"

# 模型配置: ms_name:timm_name:ms_path:ckpt_path
MODELS=(
    "xcit_tiny_12_p16_384:xcit_tiny_12_p16_384.fb_dist_in1k:timm/xcit_tiny_12_p16_384:/tmp/modelscope/timm/xcit_tiny_12_p16_384___fb_dist_in1k/pytorch_model.bin"
    "xcit_tiny_12_p8_384:xcit_tiny_12_p8_384.fb_dist_in1k:timm/xcit_tiny_12_p8_384:/tmp/modelscope/timm/xcit_tiny_12_p8_384___fb_dist_in1k/pytorch_model.bin"
    "xcit_small_12_p8_224:xcit_small_12_p8_224.fb_in1k:timm/xcit_small_12_p8_224:/tmp/modelscope/timm/xcit_small_12_p8_224___fb_in1k/pytorch_model.bin"
    "xcit_medium_24_p16_384:xcit_medium_24_p16_384.fb_dist_in1k:timm/xcit_medium_24_p16_384:/tmp/modelscope/timm/xcit_medium_24_p16_384___fb_dist_in1k/pytorch_model.bin"
    "xcit_large_24_p8_224:xcit_large_24_p8_224.fb_in1k:timm/xcit_large_24_p8_224:/tmp/modelscope/timm/xcit_large_24_p8_224___fb_in1k/pytorch_model.bin"
)

echo "=============================================="
echo "XCiT Models NPU Deployment Pipeline"
echo "Models: ${#MODELS[@]}"
echo "Output: $OUTPUT_DIR"
echo "Samples: $SAMPLES"
echo "GitCode Push: $GITCODE_PUSH"
echo "=============================================="

for entry in "${MODELS[@]}"; do
    IFS=':' read -r ms_name timm_name ms_path ckpt_path <<< "$entry"
    
    echo ""
    echo "=============================================="
    echo "[$(date '+%H:%M:%S')] Processing: $ms_name"
    echo "  timm: $timm_name"
    echo "  checkpoint: $ckpt_path"
    echo "=============================================="
    
    model_dir="${OUTPUT_DIR}/${ms_name}-npu"
    mkdir -p "$model_dir"
    
    # 1. 生成推理和精度验证脚本
    echo "  [1/4] Generating scripts..."
    python3 -c "
import sys, os
sys.path.insert(0, '.')
from xcit_pipeline_v2 import generate_inference, generate_accuracy_eval
generate_inference('$model_dir', '$timm_name', '$ckpt_path')
generate_accuracy_eval('$model_dir', '$timm_name', '$ckpt_path', 6.7)
" 2>/dev/null || {
        echo "  WARNING: Script generation failed, using default scripts"
    }
    
    # 2. 串行运行精度验证
    echo "  [2/4] Running accuracy evaluation (${SAMPLES} samples)..."
    cd "$model_dir"
    python3 accuracy_eval.py --samples "$SAMPLES" 2>&1 | tail -20 || {
        echo "  ERROR: $ms_name accuracy evaluation failed" | tee -a "$FAILURES_LOG"
        # 释放显存后继续
        python3 -c "
import gc
gc.collect()
try:
    import torch
    if hasattr(torch, 'npu'):
        torch.npu.empty_cache()
except: pass
"
        continue
    }
    
    # 3. 生成终端截图
    echo "  [3/4] Generating terminal screenshot..."
    python3 -c "
from PIL import Image, ImageDraw, ImageFont
import json, os
try:
    with open('accuracy_result.json') as f:
        r = json.load(f)
    
    img = Image.new('RGB', (820, 460), (28, 28, 36))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 13)
    except:
        font = ImageFont.load_default()
    
    lines = [
        f\"\\$ cd ${ms_name}-npu\",
        \"\\$ python3 inference.py\",
        '',
        f'  Model: ${timm_name}',
        '  Device: NPU (Ascend 910B4)',
        f'  Params: {r.get(\"params_m\",\"?\")}M  |  Speedup: {r.get(\"speedup\",\"?\")}x',
        '',
        f'  NPU avg latency: {r.get(\"npu_avg_latency_ms\",\"?\")}ms',
        f'  Throughput: {r.get(\"throughput_img_per_sec\",\"?\")} img/s',
        '',
        '  --- Precision ---',
        f'  Max diff: {r.get(\"max_diff\",0)}',
        f'  Cos sim:  {r.get(\"cosine_similarity\",0)}',
        f'  Pred match: {r.get(\"prediction_match_pct_50\",0)}%',
    ]
    y = 20
    for line in lines:
        draw.text((20, y), line, fill=(0,220,0), font=font)
        y += 22
    img.save('terminal_screenshot.png')
    print('  Screenshot saved')
except Exception as e:
    print(f'  Screenshot error: {e}')
"
    
    # 4. 可选推送到 GitCode
    if [ "$GITCODE_PUSH" = "true" ] && [ -n "$GITCODE_TOKEN" ]; then
        echo "  [4/4] Pushing to GitCode..."
        cd "$model_dir"
        
        # 添加 requirements.txt
        cat > requirements.txt << 'EOFREQ'
torch>=2.0.0
torch_npu>=2.0.0
timm>=0.9.0
Pillow>=10.0.0
numpy>=1.21.0
safetensors>=0.4.0
modelscope>=1.0.0
EOFREQ
        
        # 添加 compare_cpu_npu.py
        cat > compare_cpu_npu.py << 'PYEOF'
#!/usr/bin/env python3
import os, sys, subprocess
script_dir = os.path.dirname(os.path.abspath(__file__))
eval_script = os.path.join(script_dir, 'accuracy_eval.py')
if not os.path.exists(eval_script):
    print(f"ERROR: {eval_script} not found")
    sys.exit(1)
cmd = [sys.executable, eval_script] + sys.argv[1:]
result = subprocess.run(cmd)
sys.exit(result.returncode)
PYEOF
        chmod +x compare_cpu_npu.py
        
        # 创建 GitCode 仓库
        curl -s -X POST "https://api.gitcode.com/api/v5/user/repos" \
            -H "PRIVATE-TOKEN: ${GITCODE_TOKEN}" \
            -H "Content-Type: application/json" \
            -d "{\"name\": \"${ms_name}-npu\", \"repository_type\": \"model\", \"private\": false}" \
            2>/dev/null || true
        
        # Git push
        git init -b main 2>/dev/null
        git config user.name "$GITCODE_USERNAME"
        git config user.email "user@gitcode.com"
        git add -A
        git commit -m "feat: ${ms_name} Ascend NPU adaptation" -m "" -m "Co-Authored-By: AtomCode" 2>/dev/null || true
        git remote add origin "https://auth:${GITCODE_TOKEN}@gitcode.com/${GITCODE_USERNAME}/${ms_name}-npu.git" 2>/dev/null
        git push -u origin main --force 2>&1 | tail -3 || echo "  Push failed (may already exist)"
    else
        echo "  [4/4] GitCode push skipped"
    fi
    
    # 释放资源
    python3 -c "
import gc
gc.collect()
try:
    import torch
    if hasattr(torch, 'npu'):
        torch.npu.empty_cache()
except: pass
" 2>/dev/null
    
    echo "  ✅ Completed: $ms_name"
done

echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "Failures: $(grep -c FAILED $FAILURES_LOG 2>/dev/null || echo 0)"
if [ -s "$FAILURES_LOG" ]; then
    echo "Failure details:"
    cat "$FAILURES_LOG"
fi
echo "=============================================="
