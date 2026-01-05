
import os
from .base import VLMBase
try:
    import google.generativeai as genai
except ImportError:
    genai = None

class GeminiFlashModel(VLMBase):
    def __init__(self):
        super().__init__("Gemini 1.5 Flash")
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.model = None

    def load_model(self):
        if not genai:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def predict(self, image, question, options=None, **kwargs):
        prompt = question
        if options:
            prompt += f"\nOptions: {options}"
            
        response = self.model.generate_content([prompt, image])
        return response.text
