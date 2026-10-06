"""
AgentShield: An Adaptive Security Runtime for Autonomous AI Agents
Direct Image Vision Model Inference Module

This module provides reusable inference functionality for direct image threat classification
using the fine-tuned Vision Transformer (ViT).

Output format:
{
    "image_path": "...",
    "label": 0 or 1,
    "prediction": "benign" or "malicious",
    "benign_probability": 0.08,
    "malicious_probability": 0.92
}

Note:
The 'malicious_probability' represents the Visual Threat Score for future CATS integration.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["USE_TORCH"] = "1"

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Union, List
from PIL import Image

# Force UTF-8 stdout for Windows compatibility
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


class AgentShieldVisionPredictor:
    """
    Reusable inference engine for Vision Transformer (ViT) image threat classification.
    """
    def __init__(self, model_path: Union[str, Path] = None, device: str = None):
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        if model_path is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            model_path = project_root / "models" / "vision-image"

        self.model_path = Path(model_path)
        if not self.model_path.exists() or not any(self.model_path.iterdir()):
            raise FileNotFoundError(
                f"Vision Model directory empty or not found at: {self.model_path}. "
                "Ensure Vision Model fine-tuning has been completed."
            )

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[INFO] Loading Vision Model from: {self.model_path}")
        print(f"[INFO] Inference device: {self.device}")

        self.processor = AutoImageProcessor.from_pretrained(str(self.model_path))
        self.model = AutoModelForImageClassification.from_pretrained(str(self.model_path))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Run inference on a single image file.

        Args:
            image_path: Path to PNG/JPG/WebP image file

        Returns:
            Dict containing label (0/1), prediction, benign_probability, malicious_probability.
        """
        import torch
        import torch.nn.functional as F

        img_path = Path(image_path)
        if not img_path.exists():
            project_root = Path(__file__).resolve().parent.parent.parent
            idx = str(image_path).find("AgentShield")
            if idx != -1:
                rel = str(image_path)[idx + len("AgentShield") + 1:].replace("\\", "/")
                cand = project_root / rel
                if cand.exists():
                    img_path = cand
            elif (project_root / str(image_path).replace("\\", "/")).exists():
                img_path = project_root / str(image_path).replace("\\", "/")

        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            with Image.open(img_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                inputs = self.processor(img, return_tensors="pt")
        except Exception as e:
            raise ValueError(f"Could not open/process image {img_path}: {e}")

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).squeeze(0)

        benign_prob = float(probs[0].item())
        malicious_prob = float(probs[1].item())
        predicted_label = 1 if malicious_prob >= 0.5 else 0
        prediction_str = "malicious" if predicted_label == 1 else "benign"

        return {
            "image_path": str(img_path.resolve()),
            "label": predicted_label,
            "prediction": prediction_str,
            "benign_probability": round(benign_prob, 4),
            "malicious_probability": round(malicious_prob, 4)
        }

    def predict_batch(self, image_paths: List[Union[str, Path]], batch_size: int = 16) -> List[Dict[str, Any]]:
        """
        Run batch inference on multiple image paths.
        """
        results = []
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            for p in batch:
                results.append(self.predict(p))
        return results


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    default_model_dir = project_root / "models" / "vision-image"

    parser = argparse.ArgumentParser(description="AgentShield Vision Model Threat Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(default_model_dir),
        help="Path to saved fine-tuned Vision Model directory (default: models/vision-image)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image file to evaluate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda', 'cpu', or auto-detect if None)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive command-line evaluation loop"
    )

    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.is_absolute() and not model_path.exists():
        candidates = [
            Path.cwd() / args.model_path,
            project_root / "models" / "vision-image"
        ]
        for c in candidates:
            if c.exists() and any(c.iterdir()):
                model_path = c
                break

    predictor = AgentShieldVisionPredictor(model_path=model_path, device=args.device)

    if args.interactive:
        print("\n" + "=" * 65)
        print("AgentShield Vision Model Interactive Image Threat Tester")
        print("Type the path to an image file (or 'exit' / 'quit' to stop):")
        print("=" * 65 + "\n")

        while True:
            try:
                user_input = input("Enter image path > ").strip()
                if user_input.lower() in ("exit", "quit", "q"):
                    print("Exiting...")
                    break
                if not user_input:
                    continue

                result = predictor.predict(user_input)
                print("\n[PREDICTION RESULT]")
                print(json.dumps(result, indent=2))
                print("-" * 50 + "\n")
            except Exception as e:
                print(f"Error: {e}\n")

    elif args.image is not None:
        result = predictor.predict(args.image)
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
