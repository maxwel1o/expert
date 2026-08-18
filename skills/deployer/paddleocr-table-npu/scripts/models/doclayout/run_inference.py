"""
PP-DocLayout_plus-L NPU 推理(按仓里 README 路径,走 PaddleX 框架)
"""
import os, sys, argparse

# 仓里 infer.py 用 PaddleX 的 LayoutDetection,PP-DocLayout 仓的 from paddleocr import LayoutDetection
# 这里保持一致
from paddleocr import LayoutDetection


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="PP-DocLayout_plus-L 模型目录(含 inference.yml)")
    ap.add_argument("--image_dir", required=True, help="测试图片路径")
    ap.add_argument("--device", default="cpu", help="cpu / gpu / npu,默认 cpu(PaddleX 探测)")
    ap.add_argument("--out_dir", default="./output", help="结果输出目录")
    args = ap.parse_args()

    # 跟仓里 README 一样用 LayoutDetection
    model = LayoutDetection(model_name="PP-DocLayout_plus-L", model_dir=args.model_dir)
    for res in model.predict(args.image_dir):
        res.print()
        res.save_to_img(args.out_dir)
        res.save_to_json(args.out_dir)


if __name__ == "__main__":
    main()
