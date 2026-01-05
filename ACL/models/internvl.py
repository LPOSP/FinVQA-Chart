
import torch
from .base import VLMBase
try:
    from transformers import AutoModel, AutoTokenizer
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
except ImportError:
    pass

class InternVLModel(VLMBase):
    def __init__(self, model_id="OpenGVLab/InternVL-Chat-V1-5"):
        super().__init__("InternVL")
        self.model_id = model_id
        self.tokenizer = None
        self.model = None

    def build_transform(self, input_size):
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        return transform

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True, use_fast=False)
        self.model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True).eval()
        if torch.cuda.is_available():
            self.model.to("cuda")

    def predict(self, image, question, options=None, **kwargs):
        pixel_values = self.build_transform(input_size=448)(image).unsqueeze(0).to(torch.bfloat16).cuda()
        
        prompt = question
        if options:
            prompt += f"\nOptions: {options}"
            
        response = self.model.chat(self.tokenizer, pixel_values, prompt, generation_config=dict(max_new_tokens=200))
        return response
