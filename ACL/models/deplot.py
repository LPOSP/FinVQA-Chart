
from .base import VLMBase
import torch
try:
    from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
except ImportError:
    pass

class DePlotModel(VLMBase):
    """
    DePlot + LLM pipeline.
    First converts chart to table using DePlot, then asks LLM (e.g. Flan-T5 or GPT) to reason.
    For this implementation, we simply return the linearized table or use a small LLM if requested.
    """
    def __init__(self, model_id="google/deplot"):
        super().__init__("DePlot")
        self.model_id = model_id
        self.processor = None
        self.model = None

    def load_model(self):
        self.processor = Pix2StructProcessor.from_pretrained(self.model_id)
        self.model = Pix2StructForConditionalGeneration.from_pretrained(self.model_id)
        if torch.cuda.is_available():
            self.model.to("cuda")

    def predict(self, image, question, options=None, **kwargs):
        inputs = self.processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
            
        predictions = self.model.generate(**inputs, max_new_tokens=512)
        linearized_table = self.processor.decode(predictions[0], skip_special_tokens=True)
        
        # In a full pipeline, we would pass 'linearized_table' + 'question' to an LLM.
        # Here we just return the table as the "vision" part output, 
        # acknowledging that a separate LLM step is needed for full VQA.
        return f"Linearized Table: {linearized_table} \n(Requires downstream LLM for full answer)"
