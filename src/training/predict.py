"""
AgentShield: An Adaptive Security Runtime for Autonomous AI Agents
Common DeBERTa Prompt Injection Inference / Prediction Module

This module provides reusable inference functionality to detect prompt injection
attacks in standardized text extracted from external tool outputs across ALL domains
(Image OCR, PDF, Web, etc.).

Output format:
{
    "text": "...",
    "label": 0 or 1,
    "prediction": "benign" or "malicious",
    "benign_probability": 0.0512,
    "malicious_probability": 0.9488
}
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

# Force UTF-8 stdout for Windows compatibility
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


class PromptInjectionPredictor:
    """
    Reusable inference engine for Common DeBERTa-based prompt injection classification.
    """
    def __init__(self, model_path: Union[str, Path] = None, device: str = None):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        if model_path is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            model_path = project_root / "models" / "deberta-common"
        
        self.model_path = Path(model_path)
        if not self.model_path.exists() or not any(self.model_path.iterdir()):
            raise FileNotFoundError(
                f"Model directory empty or not found at: {self.model_path}. "
                "Ensure final common DeBERTa model training has been completed."
            )

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[INFO] Loading Common DeBERTa model from: {self.model_path}")
        print(f"[INFO] Inference device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str, max_length: int = 512) -> Dict[str, Any]:
        """
        Run inference on a single text string (Image OCR text, PDF text, or Web text).

        Args:
            text: Input text string
            max_length: Maximum sequence token length (default: 512)

        Returns:
            Dict containing label (0/1), prediction string, and probabilities.
        """
        import torch
        import torch.nn.functional as F

        text = str(text).strip() if text else ""

        # Check for empty text or standalone boilerplate domain prefixes
        is_empty_or_boilerplate = (
            not text
            or text in ("Image content:", "Image content", "Web content:", "Document content:")
            or (text.startswith("Image content:") and len(text.replace("Image content:", "").strip()) == 0)
            or (text.startswith("Web content:") and len(text.replace("Web content:", "").strip()) == 0)
            or (text.startswith("Document content:") and len(text.replace("Document content:", "").strip()) == 0)
        )

        if is_empty_or_boilerplate:
            return {
                "text": text,
                "label": 0,
                "prediction": "unspecified",
                "benign_probability": 0.5,
                "malicious_probability": 0.5,
                "has_text": False,
                "warning": (
                    "No textual content detected. If this input originates from an image tool response, "
                    "route directly to the Vision Model (models/vision-image) for visual threat scoring."
                )
            }

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        )

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
            "text": text,
            "label": predicted_label,
            "prediction": prediction_str,
            "benign_probability": round(benign_prob, 4),
            "malicious_probability": round(malicious_prob, 4),
            "has_text": True
        }

    def predict_batch(self, texts: List[str], max_length: int = 512, batch_size: int = 16) -> List[Dict[str, Any]]:
        """
        Run batch inference on multiple text inputs.
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for t in batch:
                results.append(self.predict(t, max_length=max_length))
        return results


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    default_model_dir = project_root / "models" / "deberta-common"

    parser = argparse.ArgumentParser(description="AgentShield Common DeBERTa Prompt Injection Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(default_model_dir),
        help="Path to saved fine-tuned DeBERTa model directory (default: models/deberta-common)"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Input text string to evaluate"
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
            project_root / "models" / "deberta-common"
        ]
        for c in candidates:
            if c.exists() and any(c.iterdir()):
                model_path = c
                break

    predictor = PromptInjectionPredictor(model_path=model_path, device=args.device)

    if args.interactive or (args.text is None and sys.stdin.isatty()):
        print("\n" + "=" * 65)
        print("AgentShield Common DeBERTa Interactive Prompt Injection Tester")
        print("Type your input text below (or 'exit' / 'quit' to stop):")
        print("=" * 65 + "\n")

        while True:
            try:
                user_input = input("Enter text > ").strip()
                if user_input.lower() in ("exit", "quit", "q"):
                    print("Exiting...")
                    break
                if not user_input:
                    continue

                result = predictor.predict(user_input)
                print("\n[PREDICTION RESULT]")
                print(json.dumps(result, indent=2))
                print("-" * 50 + "\n")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                break

    elif args.text is not None:
        result = predictor.predict(args.text)
        print(json.dumps(result, indent=2))
    else:
        input_text = sys.stdin.read().strip()
        if input_text:
            result = predictor.predict(input_text)
            print(json.dumps(result, indent=2))
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
