from abc import ABCMeta, abstractmethod


class Tracker(metaclass=ABCMeta):

    @abstractmethod
    def transpose(self, i, j):
        return NotImplemented

    @abstractmethod
    def get_history(self):
        return NotImplemented

    @abstractmethod
    def get_state(self):
        return NotImplemented