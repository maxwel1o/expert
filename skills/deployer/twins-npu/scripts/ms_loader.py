"""从 ModelScope 或本地缓存加载 timm 预训练权重的辅助模块"""
import os
import glob
import torch
import timm

MS_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ms_cache")


def _find_model_dir(model_name):
    """在缓存中查找模型目录"""
    # ModelScope 将 . 替换为 ___ 作为缓存目录名
    cache_name = model_name.replace(".", "___")

    # 搜索 MS_CACHE 及其直接子目录
    search_paths = [
        os.path.join(MS_CACHE, cache_name),
        os.path.join(MS_CACHE, model_name),
    ]
    # 同时搜索子目录（如 ms_cache/timm/twins_xxx）
    if os.path.isdir(MS_CACHE):
        for sub in os.listdir(MS_CACHE):
            subdir = os.path.join(MS_CACHE, sub)
            if os.path.isdir(subdir):
                search_paths.append(os.path.join(subdir, cache_name))
                search_paths.append(os.path.join(subdir, model_name))

    for p in search_paths:
        if os.path.isdir(p) or os.path.islink(p):
            real = os.path.realpath(p)
            if os.path.isdir(real):
                return real
    return None


def _load_state_dict_from_dir(model_dir):
    """从模型目录加载 state_dict"""
    # 尝试 safetensors
    for ext in [".safetensors", ".bin", ".pt", ".pth"]:
        pattern = os.path.join(model_dir, f"*{ext}")
        files = glob.glob(pattern)
        if not files:
            # 尝试 model-00001-of-xxxxx.safetensors (sharded)
            files = sorted(glob.glob(os.path.join(model_dir, f"*{ext}")))
        for fpath in files:
            try:
                if fpath.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    sd = load_file(fpath)
                else:
                    sd = torch.load(fpath, map_location="cpu", weights_only=True)
                return sd
            except Exception:
                continue
    return None


def load_timm_model(model_name):
    """从 ModelScope 下载并加载 timm 模型，优先使用本地缓存"""
    # 如果模型名不包含命名空间前缀，尝试自动补全 timm/
    ms_ids = [model_name]
    if "/" not in model_name:
        ms_ids.insert(0, f"timm/{model_name}")  # 优先尝试 timm/ 前缀

    model_dir = None
    for ms_id in ms_ids:
        model_dir = _find_model_dir(ms_id)
        if model_dir is not None:
            break
    if model_dir is None:
        from modelscope import snapshot_download
        # 尝试 timm/ 前缀下载
        for ms_id in ms_ids:
            try:
                model_dir = snapshot_download(ms_id, cache_dir=MS_CACHE)
                model_dir = _find_model_dir(ms_id) or model_dir
                break
            except Exception:
                continue

    if model_dir is None:
        raise FileNotFoundError(f"No weight file found for {model_name}")

    sd = _load_state_dict_from_dir(model_dir)
    if sd is None:
        raise FileNotFoundError(f"No weight file found for {model_name} in {model_dir}")

    # 清理 state_dict 中的前缀
    new_sd = {}
    for k, v in sd.items():
        clean_k = k
        for prefix in ["model.", "module.", "state_dict."]:
            if clean_k.startswith(prefix):
                clean_k = clean_k[len(prefix):]
                break
        new_sd[clean_k] = v

    model = timm.create_model(model_name, pretrained=False)
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")
    model.eval()
    return model
