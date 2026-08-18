"""4 个 PaddleOCR 表格模型在昇腾 NPU 部署的配置表"""
MODELS = {
    "doclayout": {
        "name": "PP-DocLayout_plus-L",
        "task": "文档版面区域检测(RT-DETR-L 800x800)",
        "modelscope_id": "PaddlePaddle/PP-DocLayout_plus-L",
        "local_dir": "/workspace/PP-DocLayout_plus-L",
        "om_filename": "inference_linux_aarch64.om",
        "onnx_input_shapes": [
            "im_shape:1,2", "image:1,3,800,800", "scale_factor:1,2",
        ],
        "onnx_staticize_batch": True,
        "onnx_upgrade_opset": False,
        "auto_optimizer": True,
        "demo_image": "/workspace/PP-DocLayout_plus-L-sact/test_Doclayout_plus.png",
        "paddlex_adapt": "patch",
        "site_packages_fixes": 2,
    },
    "seal_det": {
        "name": "PP-OCRv4_server_seal_det",
        "task": "印章文本检测(DBNet 640x640)",
        "modelscope_id": "PaddlePaddle/PP-OCRv4_server_seal_det",
        "local_dir": "/workspace/PP-OCRv4_server_seal_det",
        "om_filename": "inference_linux_aarch64.om",
        "onnx_input_shapes": ["x:1,3,640,640"],
        "onnx_staticize_batch": True,
        "onnx_upgrade_opset": False,
        "auto_optimizer": True,
        "demo_image": "/workspace/PP-OCRv4_server_seal_det-OM/test_seal.png",
        "paddlex_adapt": "cp",
        "site_packages_fixes": 0,
    },
    "table_cls": {
        "name": "PP-LCNet_x1_0_table_cls",
        "task": "表格分类(LCNet 224x224,有线/无线二分类)",
        "modelscope_id": "PaddlePaddle/PP-LCNet_x1_0_table_cls",
        "local_dir": "/workspace/PP-LCNet_x1_0_table_cls",
        "om_filename": "inference_linux_aarch64.om",
        "onnx_input_shapes": ["x:1,3,224,224"],
        "onnx_staticize_batch": True,
        "onnx_upgrade_opset": True,
        "auto_optimizer": False,
        "demo_image": "/workspace/PP-DocLayout_plus-L-sact/PP-LCNet_x1_0_table_cls/test_table.png",
        "paddlex_adapt": "cp",
        "site_packages_fixes": 0,
    },
    "table_cell": {
        "name": "RT-DETR-L_wired_table_cell_det",
        "task": "表格单元格检测(RT-DETR-L 640x640)",
        "modelscope_id": "PaddlePaddle/RT-DETR-L_wired_table_cell_det",
        "local_dir": "/workspace/RT-DETR-L_wired_table_cell_det",
        "om_filename": "inference_linux_aarch64.om",
        "onnx_input_shapes": [
            "im_shape:1,2", "image:1,3,640,640", "scale_factor:1,2",
        ],
        "onnx_staticize_batch": True,
        "onnx_upgrade_opset": False,
        "auto_optimizer": True,
        "demo_image": "/workspace/PP-DocLayout_plus-L-sact/RT-DETR-L_wired_table_cell_det-OM/test_rt.png",
        "paddlex_adapt": "patch",
        "site_packages_fixes": 2,
    },
}


def get(model_key):
    if model_key not in MODELS:
        raise KeyError(f"unknown model {model_key!r}, valid: {list(MODELS)}")
    return MODELS[model_key]
