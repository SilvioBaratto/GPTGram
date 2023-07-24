from ..base import BaseGram
from abc import ABCMeta, abstractmethod

class BaseMetric(BaseGram, metaclass=ABCMeta):
    """
    Abstract base class for all metrics.
    """
    
    @abstractmethod
    def score(self, references, candidate):
        """
        Calculates the score of the metric for the given reference and candidate sentences.
        
        Args:
            references (list of str): The list of reference sentences.
            candidate (str): The candidate sentence.
            
        Returns:
            float: The calculated score.
        """
        pass