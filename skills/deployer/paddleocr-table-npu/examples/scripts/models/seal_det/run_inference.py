"""
PP-OCRv4_server_seal_det NPU 推理(按仓里 infer.py 路径,走 PaddleX create_predictor)
"""
import os, sys, argparse
from paddlex.inference import create_predictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="PP-OCRv4_server_seal_det 模型目录")
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out_dir", default="./output")
    args = ap.parse_args()

    # 跟仓里 infer.py 一模一样
    model = create_predictor(model_name="PP-OCRv4_server_seal_det", model_dir=args.model_dir)
    for res in model.predict(args.image_dir):
        res.print()
        res.save_to_img(args.out_dir)
        res.save_to_json(args.out_dir)


if __name__ == "__main__":
    main()
