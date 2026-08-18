"""
RT-DETR-L_wired_table_cell_det NPU 推理(按仓里 infer.py 路径,走 PaddleX DetPredictor)
"""
import os, sys, argparse
from paddlex.inference.models.object_detection.predictor import DetPredictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="RT-DETR-L_wired_table_cell_det 模型目录")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out_dir", default="./output")
    args = ap.parse_args()

    # 跟仓里 infer.py 一模一样
    model = DetPredictor(model_dir=args.model_dir)
    results = model([args.image])
    for res in results:
        res.save_to_img(args.out_dir)
    print("结果已保存到 output 目录")


if __name__ == "__main__":
    main()
