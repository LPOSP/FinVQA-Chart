
import torch
from .base import VLMBase
try:
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
except ImportError:
    LlavaNextProcessor = None

class LLaVAModel(VLMBase):
    def __init__(self, model_id="llava-hf/llava-v1.6-mistral-7b-hf"):
        super().__init__("LLaVA-NeXT")
        self.model_id = model_id
        self.processor = None
        self.model = None

    def load_model(self):
        if not LlavaNextProcessor:
            raise ImportError("Please install transformers>=4.39.0 accelerator")
        
        self.processor = LlavaNextProcessor.from_pretrained(self.model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True
        )
        if torch.cuda.is_available():
            self.model.to("cuda")

    def predict(self, image, question, options=None, **kwargs):
        prompt = f"[INST] <image>\n{question}"
        if options:
            prompt += f"\nOptions: {options}"
        prompt += " [/INST]"

        inputs = self.processor(prompt, image, return_tensors="pt").to(self.model.device)
        
        output = self.model.generate(**inputs, max_new_tokens=100)
        return self.processor.decode(output[0], skip_special_tokens=True)
