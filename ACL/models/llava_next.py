
import torch
from .base import VLMBase
try:
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
except ImportError:
    pass

class LLaVANext34BModel(VLMBase):
    def __init__(self):
        super().__init__("LLaVA-1.6-34B")
        self.model_id = "llava-hf/llava-v1.6-34b-hf"
        self.processor = None
        self.model = None

    def load_model(self):
        self.processor = LlavaNextProcessor.from_pretrained(self.model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True
        )
        if torch.cuda.is_available():
            self.model.to("cuda")

    def predict(self, image, question, options=None, **kwargs):
        prompt = f"<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n{question}"
        if options:
            prompt += f"\nOptions: {options}"
        prompt += "<|im_end|><|im_start|>assistant\n"

        inputs = self.processor(prompt, image, return_tensors="pt").to(self.model.device)
        
        output = self.model.generate(**inputs, max_new_tokens=100)
        return self.processor.decode(output[0], skip_special_tokens=True)
