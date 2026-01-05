
import os
import base64
from io import BytesIO
from .base import VLMBase
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

class GPT4VModel(VLMBase):
    def __init__(self):
        super().__init__("GPT-4V")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = None

    def load_model(self):
        if not OpenAI:
            raise ImportError("Please install openai: pip install openai")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=self.api_key)

    def _encode_image(self, image):
        buffered = BytesIO()
        image.convert("RGB").save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def predict(self, image, question, options=None, **kwargs):
        base64_image = self._encode_image(image)
        
        prompt = question
        if options:
            prompt += f"\nOptions: {options}"

        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview", # Or latest equivalent
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        return response.choices[0].message.content
