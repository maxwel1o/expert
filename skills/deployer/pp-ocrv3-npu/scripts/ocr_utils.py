"""Common utilities for PP-OCRv3 ONNX models on CPU/NPU."""

import cv2
import numpy as np
import onnxruntime as ort


# PP-OCRv3 detection preprocessing params (PyTorch ImageNet-style)
DET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DET_MAX_SIDE = 960
DET_INPUT_SIZE = (960, 960)

# PP-OCRv3 recognition preprocessing params
REC_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
REC_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
REC_HEIGHT = 48


def resize_pad(img, target_size, fill_value=0):
    """Resize image maintaining aspect ratio and pad to target_size."""
    h, w = img.shape[:2]
    tw, th = target_size
    scale = min(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    padded = np.full((th, tw, 3), fill_value, dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    return padded, scale


def preprocess_detection(img):
    """Preprocess image for PP-OCRv3 detection model.

    Args:
        img: BGR image (H, W, 3) from cv2.imread

    Returns:
        tensor: (1, 3, 960, 960) float32 numpy array
    """
    # Resize and pad to 960x960
    padded, scale = resize_pad(img, DET_INPUT_SIZE, fill_value=0)
    # BGR to RGB
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    # Normalize
    rgb = rgb.astype(np.float32) / 255.0
    rgb = (rgb - DET_MEAN) / DET_STD
    # HWC -> NCHW
    tensor = np.transpose(rgb, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)
    return tensor.astype(np.float32), scale


def postprocess_detection(output, scale, img_shape, threshold=0.3):
    """Extract bounding boxes from detection model output (DB post-processing).

    Args:
        output: (1, 1, 960, 960) probability map
        scale: resize scale factor
        img_shape: original image (H, W, 3)
        threshold: binarization threshold

    Returns:
        boxes: list of [x1, y1, x2, y2] in original image coordinates
    """
    prob_map = output[0, 0]  # (960, 960)
    binary = (prob_map > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        if len(contour) < 4:
            continue
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)  # use int32 instead of int0 (deprecated)
        # Scale back to original image coordinates
        box = box / scale
        box = np.clip(box, 0, [img_shape[1] - 1, img_shape[0] - 1])
        x1, y1 = box[:, 0].min(), box[:, 1].min()
        x2, y2 = box[:, 0].max(), box[:, 1].max()
        if x2 > x1 and y2 > y1:
            boxes.append([int(x1), int(y1), int(x2), int(y2)])
    return boxes


def preprocess_recognition(img):
    """Preprocess image for PP-OCRv3 recognition model.

    Args:
        img: BGR image crop (H, W, 3) from text detection

    Returns:
        tensor: (1, 3, 48, W_padded) float32 numpy array
        width: original width after resize
    """
    h, w = img.shape[:2]
    # Resize to height=48, maintain aspect ratio
    scale = REC_HEIGHT / h
    new_w = int(w * scale)
    resized = cv2.resize(img, (new_w, REC_HEIGHT))
    # Pad width to multiple of 8 (for ONNX model compatibility)
    pad_w = (8 - new_w % 8) % 8
    if pad_w > 0:
        padded = np.pad(resized, ((0, 0), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
    else:
        padded = resized
    # BGR to RGB
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    # Normalize
    rgb = rgb.astype(np.float32) / 255.0
    rgb = (rgb - REC_MEAN) / REC_STD
    # HWC -> NCHW
    tensor = np.transpose(rgb, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)
    return tensor.astype(np.float32), new_w


def postprocess_recognition(output, char_dict, orig_width):
    """Decode recognition model output using CTC greedy decoding.

    Args:
        output: (1, T, num_classes) logits array
        char_dict: list of characters, index -> char
        orig_width: original width before padding (unused, kept for compatibility)

    Returns:
        text: decoded string
        confidence: average confidence score
    """
    logits = output[0]  # (T, num_classes)
    pred_ids = np.argmax(logits, axis=1)
    probs = np.max(logits, axis=1)

    num_classes = logits.shape[1]
    # PP-OCR models use: blank(0) + optional_space(1) + chars(2..N-1)
    # offset = 1 when len(dict)+1 == num_classes, offset = 2 when len(dict)+2 == num_classes
    char_offset = num_classes - len(char_dict)

    # CTC greedy decoding: collapse repeated chars and remove blank (0)
    prev_id = -1
    text_chars = []
    confidences = []
    for idx, pid in enumerate(pred_ids):
        if pid != prev_id and pid != 0:  # 0 is blank
            char_idx = pid - char_offset
            if 0 <= char_idx < len(char_dict):
                text_chars.append(char_dict[char_idx])
                confidences.append(float(probs[idx]))
        prev_id = pid

    text = ''.join(text_chars)
    avg_confidence = np.mean(confidences) if confidences else 0.0
    return text, avg_confidence


def create_session(model_path, provider):
    """Create ONNX Runtime session with specified provider.

    Args:
        model_path: path to ONNX model
        provider: 'CPUExecutionProvider' or 'CANNExecutionProvider'

    Returns:
        ort.InferenceSession
    """
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if provider == 'CANNExecutionProvider':
        provider_options = [('CANNExecutionProvider', {'device_id': 0})]
    else:
        provider_options = [('CPUExecutionProvider', {})]
    return ort.InferenceSession(model_path, sess_options=sess_options, providers=provider_options)


def run_inference(session, input_tensor):
    """Run inference on ONNX session.

    Args:
        session: ort.InferenceSession
        input_tensor: (1, C, H, W) numpy array

    Returns:
        output: model output array
    """
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: input_tensor})[0]


def load_char_dict(dict_path):
    """Load character dictionary for recognition model."""
    with open(dict_path, 'r', encoding='utf-8') as f:
        chars = [line.strip() for line in f.readlines()]
    return chars


def decode_to_chars(char_dict, indices):
    """Convert character indices to characters."""
    chars = []
    for idx in indices:
        idx_i = int(idx) - 1  # PP-OCR uses 1-based indexing
        if 0 <= idx_i < len(char_dict):
            chars.append(char_dict[idx_i])
        else:
            chars.append('')
    return chars
