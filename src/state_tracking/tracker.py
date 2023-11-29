from abc import ABCMeta, abstractmethod


class Tracker(metaclass=ABCMeta):
    @abstractmethod
    def transpose(self, i, j):
        return NotImplemented

    @abstractmethod
    def get_history(self) -> list[str]:
        return NotImplemented

    @abstractmethod
    def get_state(self) -> str:
        return NotImplemented
