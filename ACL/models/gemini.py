
import os
from .base import VLMBase
try:
    import google.generativeai as genai
except ImportError:
    genai = None

class GeminiModel(VLMBase):
    def __init__(self, model_version="gemini-1.5-pro"):
        super().__init__(f"Gemini ({model_version})")
        self.model_version = model_version
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.model = None

    def load_model(self):
        if not genai:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_version)

    def predict(self, image, question, options=None, **kwargs):
        prompt = question
        if options:
            prompt += f"\nOptions: {options}"
            
        response = self.model.generate_content([prompt, image])
        return response.text
