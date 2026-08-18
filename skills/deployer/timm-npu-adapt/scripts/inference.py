#!/usr/bin/env python3
"""
Inference script for timm models on Ascend NPU.
Usage: python3 inference.py --model <model_name> [--image <path>] [--cpu]

Example:
    python3 inference.py --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k
    python3 inference.py --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --image test.jpg
"""

import os, sys, argparse, time
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

os.environ["ASCEND_LOG"] = "3"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

try:
    import torch_npu  # noqa: F401
    NPU_AVAILABLE = torch.npu.is_available()
except ImportError:
    NPU_AVAILABLE = False

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


# ImageNet labels (class names)
IMAGENET_LABELS = [
    "tench", "goldfish", "great_white_shark", "tiger_shark", "hammerhead_shark",
    "electric_ray", "stingray", "cock", "hen", "ostrich",
    "brambling", "goldfinch", "house_finch", "junco", "indigo_bunting",
    "robin", "bulbul", "jay", "magpie", "chickadee",
    "water_ouzel", "kite", "bald_eagle", "vulture", "great_grey_owl",
    "European_fire_salamander", "common_newt", "eft", "spotted_salamander", "axolotl",
    "bullfrog", "tree_frog", "tailed_frog", "loggerhead", "leatherback_turtle",
    "mud_turtle", "terrapin", "box_turtle", "banded_gecko", "common_iguana",
    "American_chameleon", "whiptail", "agama", "frilled_lizard", "alligator_lizard",
    "Gila_monster", "green_lizard", "African_chameleon", "Komodo_dragon", "African_crocodile",
    "American_alligator", "triceratops", "thunder_snake", "ringneck_snake", "hognose_snake",
    "green_snake", "king_snake", "garter_snake", "water_snake", "vine_snake",
    "night_snake", "boa_constrictor", "rock_python", "Indian_cobra", "green_mamba",
    "sea_snake", "horned_viper", "diamondback", "sidewinder", "trilobite",
    "harvestman", "scorpion", "black_and_gold_garden_spider", "barn_spider", "garden_spider",
    "black_widow", "tarantula", "wolf_spider", "tick", "centipede",
    "black_grouse", "ptarmigan", "ruffed_grouse", "prairie_chicken", "peacock",
    "quail", "partridge", "African_grey", "macaw", "sulphur_crested_cockatoo",
    "lorikeet", "coucal", "bee_eater", "hornbill", "hummingbird",
    "jacamar", "toucan", "drake", "red_breasted_merganser", "goose",
    "black_swan", "tusker", "echidna", "platypus", "wallaby",
    "koala", "wombat", "jellyfish", "sea_anemone", "brain_coral",
    "flatworm", "nematode", "conch", "snail", "slug",
    "sea_slug", "chiton", "chambered_nautilus", "Dungeness_crab", "rock_crab",
    "fiddler_crab", "king_crab", "American_lobster", "spiny_lobster", "crayfish",
    "hermit_crab", "isopod", "white_stork", "black_stork", "spoonbill",
    "flamingo", "little_blue_heron", "American_egret", "bittern", "crane_bird",
    "limpkin", "European_gallinule", "American_coot", "bustard", "ruddy_turnstone",
    "red_backed_sandpiper", "redshank", "dowitcher", "oystercatcher", "pelican",
    "king_penguin", "albatross", "grey_whale", "killer_whale", "dugong",
    "sea_lion", "Chihuahua", "Japanese_spaniel", "Maltese_dog", "Pekinese",
]
# Full 1000 labels - using a lookup function
def get_label(idx):
    """Get ImageNet class label by index."""
    labels = IMAGENET_LABELS
    return labels[idx] if idx < len(labels) else f"class_{idx}"


def load_image(image_path, transform):
    """Load and preprocess an image."""
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)
    return img, input_tensor


def create_demo_image(size=(384, 384)):
    """Create a synthetic test image."""
    from PIL import ImageDraw
    img = Image.new("RGB", size, color=(128, 128, 128))
    draw = ImageDraw.Draw(img)
    draw.ellipse([size[0]//4, size[1]//4, size[0]*3//4, size[1]*3//4], fill=(180, 120, 60))
    arr = np.array(img)
    noise = np.random.randint(0, 30, arr.shape, dtype=np.uint8)
    arr = np.clip(arr.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def main():
    parser = argparse.ArgumentParser(description="timm model inference on Ascend NPU")
    parser.add_argument("--model", type=str, required=True, help="timm model name")
    parser.add_argument("--image", type=str, default=None, help="input image path")
    parser.add_argument("--cpu", action="store_true", help="force CPU inference")
    args = parser.parse_args()

    device = "cpu" if args.cpu or not NPU_AVAILABLE else "npu"
    if not NPU_AVAILABLE and not args.cpu:
        print("⚠️  NPU not available, falling back to CPU")
        device = "cpu"

    print(f"🚀 Loading model: {args.model}")
    print(f"💻 Device: {device.upper()}")

    # Load model
    t0 = time.time()
    model = timm.create_model(args.model, pretrained=True)
    model.eval()
    if device == "npu":
        model = model.npu()
    load_time = time.time() - t0
    params = sum(p.numel() for p in model.parameters())
    print(f"📦 Model loaded in {load_time:.2f}s | Params: {params/1e6:.1f}M")

    # Get input size
    cfg = timm.get_pretrained_cfg(args.model)
    input_size = cfg.input_size
    print(f"📐 Input size: {input_size}")

    # Create transform
    config = resolve_data_config({}, model=args.model)
    transform = create_transform(**config)

    # Load or create test image
    if args.image:
        img, input_tensor = load_image(args.image, transform)
        print(f"🖼️  Input image: {args.image}")
    else:
        img = create_demo_image(size=(input_size[1], input_size[2]))
        input_tensor = transform(img).unsqueeze(0)
        print(f"🖼️  Using synthetic test image ({input_size[1]}x{input_size[2]})")

    input_tensor = input_tensor.to(device)

    # Warmup
    if device == "npu":
        print("🔥 Warming up...")
        warmup = torch.randn(1, *input_size).to(device)
        with torch.no_grad():
            for _ in range(5):
                _ = model(warmup)
        torch.npu.synchronize()

    # Inference
    print("🧠 Running inference...")
    with torch.no_grad():
        t0 = time.perf_counter()
        output = model(input_tensor)
        if device == "npu":
            torch.npu.synchronize()
        infer_time = time.perf_counter() - t0

    # Post-process
    probs = F.softmax(output, dim=1)
    top5_probs, top5_indices = torch.topk(probs, k=5, dim=1)

    print(f"\n✅ Inference complete! ({infer_time*1000:.2f}ms)")
    print(f"\n{'─'*50}")
    print("  Top-5 predictions:")
    print(f"{'─'*50}")
    for i in range(5):
        idx = top5_indices[0, i].item()
        prob = top5_probs[0, i].item() * 100
        label = get_label(idx)
        print(f"  {i+1}. {label:<50s} {prob:.2f}%")
    print(f"{'─'*50}")

    # Benchmark (batch mode)
    print(f"\n📊 Throughput benchmark...")
    batch_size = 4 if input_size[1] > 300 else 8
    if params > 50e6:
        batch_size = 1
    bench_input = torch.randn(batch_size, *input_size).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(bench_input)
        if device == "npu":
            torch.npu.synchronize()
        n_iter = 50
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = model(bench_input)
        if device == "npu":
            torch.npu.synchronize()
        total = time.perf_counter() - t0
    throughput = (n_iter * batch_size) / total
    print(f"  Batch size: {batch_size}")
    print(f"  Throughput: {throughput:.1f} img/s")
    print(f"  Latency: {total/n_iter*1000:.1f}ms per batch")


if __name__ == "__main__":
    main()
