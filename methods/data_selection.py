# methods/data_selection.py
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader

class DataSelection(ABC):
    @abstractmethod
    def select(self, *args, **kwargs) -> DataLoader:
        """返回一个用于训练的核心子集 DataLoader"""
        pass
