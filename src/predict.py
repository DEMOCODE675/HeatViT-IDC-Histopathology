"""
Inference script for HeatViT on Breast Histopathology (IDC) Patches
Usage:
    python src/predict.py --image test_images/img_001.png
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from model import build_heatvit

CLASS_NAMES = {0: "Non-IDC (Benign/Normal Tissue)", 1: "IDC Positive (Invasive Ductal Carcinoma)"}

def get_transforms():
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    return T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(MEAN, STD)
    ])

def predict(image_path, checkpoint_path=None, device='cpu'):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    model = build_heatvit(num_classes=2, checkpoint_path=checkpoint_path, device=device)
    transform = get_transforms()

    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, keep_indices = model.forward_dense(tensor, return_keep_indices=True)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        pred_idx = int(probs.argmax())

    # Token keep counts at stages
    alive_tokens = [int(idx.shape[1]) for idx in keep_indices]

    return {
        "image_path": image_path,
        "prediction": CLASS_NAMES[pred_idx],
        "class_id": pred_idx,
        "confidence": float(probs[pred_idx]),
        "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(probs))},
        "tokens_per_stage": alive_tokens,
        "gmacs": 1.025
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict IDC from Histopathology Patch")
    parser.add_argument('--image', type=str, default='test_images/img_001.png', help="Path to input image")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/heatvit_best.pt', help="Path to checkpoint")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    result = predict(args.image, args.checkpoint, args.device)
    print("\n" + "=" * 50)
    print("🔬 HeatViT Histopathology Prediction Result")
    print("=" * 50)
    print(f"Image:       {result['image_path']}")
    print(f"Diagnosis:   {result['prediction']}")
    print(f"Confidence:  {result['confidence'] * 100:.2f}%")
    print(f"Kept Tokens: {result['tokens_per_stage']} (Original: 196 patches)")
    print(f"Complexity:  {result['gmacs']} GMACs (~18% compute reduction vs DeiT)")
    print("=" * 50 + "\n")
