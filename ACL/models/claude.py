
import os
import base64
from io import BytesIO
from .base import VLMBase
try:
    import anthropic
except ImportError:
    anthropic = None

class ClaudeModel(VLMBase):
    def __init__(self, model_version="claude-3-5-sonnet-20240620"):
        super().__init__("Claude")
        self.model_version = model_version
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = None

    def load_model(self):
        if not anthropic:
            raise ImportError("Please install anthropic: pip install anthropic")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def _encode_image(self, image):
        buffered = BytesIO()
        image.convert("RGB").save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def predict(self, image, question, options=None, **kwargs):
        base64_image = self._encode_image(image)
        
        prompt = question
        if options:
            prompt += f"\nOptions: {options}"

        message = self.client.messages.create(
            model=self.model_version,
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )
        return message.content[0].text
