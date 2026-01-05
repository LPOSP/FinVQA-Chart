
import torch
from .base import VLMBase
try:
    from transformers import AutoModelForCausalLM, AutoProcessor
except ImportError:
    pass

class Phi3VisionModel(VLMBase):
    def __init__(self):
        super().__init__("Phi-3.5-Vision")
        self.model_id = "microsoft/Phi-3.5-vision-instruct"
        self.processor = None
        self.model = None

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            device_map="cuda", 
            trust_remote_code=True, 
            torch_dtype="auto", 
            _attn_implementation='flash_attention_2'
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, 
            trust_remote_code=True
        )

    def predict(self, image, question, options=None, **kwargs):
        prompt = question
        if options:
            prompt += f"\nOptions: {options}"
            
        messages = [
            {"role": "user", "content": f"<|image_1|>\n{prompt}"},
        ]
        
        prompt_text = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(prompt_text, [image], return_tensors="pt").to("cuda")
        
        generation_args = {
            "max_new_tokens": 500,
            "temperature": 0.0,
            "do_sample": False,
        }
        
        generate_ids = self.model.generate(**inputs, eos_token_id=self.processor.tokenizer.eos_token_id, **generation_args)
        
        # remove input tokens
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        return response
