
import torch
from .base import VLMBase
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    pass

class CogVLM2Model(VLMBase):
    def __init__(self):
        super().__init__("CogVLM2-19B")
        self.model_id = "THUDM/cogvlm2-llama3-chat-19B"
        self.tokenizer = None
        self.model = None

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, 
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval()
        if torch.cuda.is_available():
            self.model.to("cuda")

    def predict(self, image, question, options=None, **kwargs):
        prompt = question
        if options:
            prompt += f"\nOptions: {options}"
            
        # CogVLM2 usage often involves specific input formatting
        # Using simplified transformers integration
        input_by_model = self.model.build_conversation_input_ids(
            self.tokenizer,
            query=prompt,
            history=[],
            images=[image],
            template_version='chat'
        )
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to('cuda'),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to('cuda'),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to('cuda'),
            'images': [[input_by_model['images'][0].to('cuda').to(torch.bfloat16)]] if image else None,
        }
        
        gen_kwargs = {
            "max_new_tokens": 2048,
            "pad_token_id": 128002,
        }
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0])
            
        return response
