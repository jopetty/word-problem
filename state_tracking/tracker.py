from typing import List
from abc import ABCMeta, abstractmethod


class Tracker(metaclass=ABCMeta):

    @abstractmethod
    def transpose(self, i, j):
        return NotImplemented

    @abstractmethod
    def get_history(self) -> List[str]:
        return NotImplemented

    @abstractmethod
    def get_state(self) -> str:
        return NotImplemented