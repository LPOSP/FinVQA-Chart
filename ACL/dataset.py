
import os
from datasets import load_dataset

class FinVQAChartDataset:
    """
    Dataset loader for FinVQA-Chart: Financial Vision-Language Reasoning at Scale.
    Hosted at: https://huggingface.co/datasets/qsWDFGHN/FIN-VQA
    """
    def __init__(self, split="test", cache_dir=None):
        self.dataset_id = "qsWDFGHN/FIN-VQA"
        self.split = split
        self.cache_dir = cache_dir
        self.data = self._load_data()

    def _load_data(self):
        print(f"Loading {self.dataset_id} ({self.split})...")
        # According to HF page, splits might be defined or we use 'train'/'test'
        try:
            ds = load_dataset(self.dataset_id, split=self.split, cache_dir=self.cache_dir)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Attempting to load 'train' split as fallback or checks available splits.")
            ds = load_dataset(self.dataset_id, split="train", cache_dir=self.cache_dir)
        return ds

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Standardize output for models
        # Item expected keys: 'image', 'question', 'answer', 'question_type', etc.
        return {
            "image": item.get("image"),
            "question": item.get("question"),
            "correct_answer": item.get("answer") or item.get("label"),
            "question_type": item.get("question_type"),
            "difficulty": item.get("difficulty"),
            "options": item.get("options"), # If output is MC
            "image_id": item.get("image_id")
        }

if __name__ == "__main__":
    ds = FinVQAChartDataset(split="train")
    print(f"Loaded {len(ds)} samples.")
    print("Sample 0:", ds[0])
