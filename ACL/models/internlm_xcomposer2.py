
import torch
from .base import VLMBase
try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    pass

class InternLMXComposer2Model(VLMBase):
    def __init__(self):
        super().__init__("InternLM-XComposer2")
        self.model_id = "internlm/internlm-xcomposer2-vl-7b"
        self.tokenizer = None
        self.model = None

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_id, 
            trust_remote_code=True
        ).cuda().eval()

    def predict(self, image, question, options=None, **kwargs):
        prompt = f"<ImageHere>{question}"
        if options:
            prompt += f"\nOptions: {options}"
            
        with torch.cuda.amp.autocast():
            response, _ = self.model.chat(
                self.tokenizer, 
                query=prompt, 
                image=image, 
                history=[], 
                do_sample=False
            )
        return response
