from torch.nn import Module
from allennlp.common.registrable import Registrable
from abc import abstractmethod, abstractclassmethod, ABCMeta
from typing import Any, T, Type


class BaseEncoder(Module, Registrable, metaclass=ABCMeta):
    def __init__(self):
        super(BaseEncoder, self).__init__()

    @property
    @abstractmethod
    def path_to_model(self) -> str:
        pass

    @property
    @abstractmethod
    def max_bpe_token(self) -> int:
        pass

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractclassmethod
    def load_from_last_checkpoint(cls, *args: Any, **kwargs: Any) -> Type[T]:
        pass
