
import torch
from .base import VLMBase
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    pass

class QwenVLModel(VLMBase):
    def __init__(self, model_id="Qwen/Qwen-VL-Chat"):
        super().__init__("Qwen-VL")
        self.model_id = model_id
        self.tokenizer = None
        self.model = None

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            device_map="auto", 
            trust_remote_code=True
        ).eval()

    def predict(self, image, question, options=None, **kwargs):
        # Qwen-VL expects paths usually, but supports PIL
        # We might need to save image to temp file if API requires path
        # But Qwen-VL-Chat `chat` method supports list of dicts or special format
        
        # Save temp image for path-based interface if needed
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            image.save(f.name)
            img_path = f.name
            
        query = self.tokenizer.from_list_format([
            {'image': img_path},
            {'text': question + (f"\nOptions: {options}" if options else "")},
        ])
        
        response, history = self.model.chat(self.tokenizer, query=query, history=None)
        
        import os
        os.remove(img_path)
        return response
