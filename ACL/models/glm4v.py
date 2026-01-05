
import torch
from .base import VLMBase
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor
except ImportError:
    pass

class GLM4VModel(VLMBase):
    def __init__(self):
        super().__init__("GLM-4V-9B")
        self.model_id = "THUDM/glm-4v-9b"
        self.tokenizer = None
        self.model = None

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to("cuda").eval()

    def predict(self, image, question, options=None, **kwargs):
        prompt = question
        if options:
            prompt += f"\nOptions: {options}"
            
        inputs = self.model.build_inputs(self.tokenizer, query=prompt, image=image)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        gen_kwargs = {"max_new_tokens": 512, "do_sample": False}
        outputs = self.model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
