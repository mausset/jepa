from abc import ABC, abstractmethod

from jepa.models.model import JEPA


class BasePlanner(ABC):
    def __init__(
        self,
        wm: JEPA,
        action_dim,
        pre_processor,
    ):
        self.wm = wm
        self.action_dim = action_dim
        self.pre_processor = pre_processor

    @abstractmethod
    def plan(self, x):
        pass
