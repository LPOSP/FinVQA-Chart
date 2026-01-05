
from abc import ABC, abstractmethod

class VLMBase(ABC):
    """
    Abstract base class for Vision-Language Models evaluated on FinVQA-Chart.
    """
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.config = kwargs

    @abstractmethod
    def predict(self, image, question, options=None, **kwargs):
        """
        Predict the answer for a given image and question.
        
        Args:
            image (PIL.Image or str): The financial chart image.
            question (str): The query about the chart.
            options (list, optional): List of options for multiple choice questions.
            
        Returns:
            str: The predicted answer.
        """
        pass

    def load_model(self):
        """
        Load weights or initialize API clients.
        """
        pass
