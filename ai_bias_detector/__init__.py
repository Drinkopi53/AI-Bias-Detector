"""
AI Bias Detector: Sebuah pustaka Python untuk mendeteksi dan membantu memitigasi bias
dalam model dan dataset machine learning.
"""

__version__ = "0.1.0"

from .dataset_analyzer import DatasetAnalyzer
from .model_analyzer import ModelAnalyzer
from .mitigation import MitigationRecommender

__all__ = [
    'DatasetAnalyzer',
    'ModelAnalyzer',
    'MitigationRecommender'
]
