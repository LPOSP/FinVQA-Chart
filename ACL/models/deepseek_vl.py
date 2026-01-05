
import torch
from .base import VLMBase
try:
    from transformers import AutoModelForCausalLM, AutoProcessor
except ImportError:
    pass

class DeepSeekVLModel(VLMBase):
    def __init__(self):
        super().__init__("DeepSeek-VL-7B")
        self.model_id = "deepseek-ai/deepseek-vl-7b-chat"
        self.processor = None
        self.model = None

    def load_model(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        ).cuda()

    def predict(self, image, question, options=None, **kwargs):
        prompt = question
        if options:
            prompt += f"\nOptions: {options}"
            
        conversation = [
            {
                "role": "User",
                "content": "<image_placeholder>" + prompt,
                "images": [image]
            },
            {
                "role": "Assistant",
                "content": ""
            }
        ]
        
        # Deepseek-vl uses prepare_inputs
        prepare_inputs = self.processor(
            conversations=conversation,
            images=[image],
            force_batchify=True
        ).to(self.model.device)
        
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
        
        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            bos_token_id=self.processor.tokenizer.bos_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )
        
        answer = self.processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return answer
