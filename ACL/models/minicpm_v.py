
import torch
from .base import VLMBase
try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    pass

class MiniCPMVModel(VLMBase):
    def __init__(self):
        super().__init__("MiniCPM-V-2.6")
        self.model_id = "openbmb/MiniCPM-V-2_6"
        self.tokenizer = None
        self.model = None

    def load_model(self):
        self.model = AutoModel.from_pretrained(
            self.model_id, 
            trust_remote_code=True,
            attn_implementation='sdpa', 
            torch_dtype=torch.bfloat16
        )
        self.model = self.model.eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, 
            trust_remote_code=True
        )

    def predict(self, image, question, options=None, **kwargs):
        prompt = question
        if options:
            prompt += f"\nOptions: {options}"
            
        msgs = [{'role': 'user', 'content': prompt}]
        
        res = self.model.chat(
            image=image,
            msgs=msgs,
            tokenizer=self.tokenizer,
            sampling=False
        )
        return res
