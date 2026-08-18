"""
PP-LCNet_x1_0_table_cls NPU 推理(按仓里 infer.py 路径,走 PaddleX ClasPredictor)
"""
import os, sys, argparse
from paddlex.inference.models.image_classification.predictor import ClasPredictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="PP-LCNet_x1_0_table_cls 模型目录")
    ap.add_argument("--image", required=True, help="测试图片路径")
    ap.add_argument("--out_dir", default="./output")
    args = ap.parse_args()

    # 跟仓里 infer.py 一模一样
    model = ClasPredictor(model_dir=args.model_dir)
    results = model([args.image])
    for res in results:
        res.save_to_img(args.out_dir)
    print("推理成功!")


if __name__ == "__main__":
    main()
