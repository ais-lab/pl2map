import torch.nn as nn
from omegaconf import OmegaConf
from abc import ABCMeta, abstractmethod

class MetaModel(ABCMeta):
    def __prepare__(name, bases, **kwds):
        total_conf = OmegaConf.create()
        for base in bases:
            for key in ('base_default_conf', 'default_conf'):
                update = getattr(base, key, {})
                if isinstance(update, dict):
                    update = OmegaConf.create(update)
                total_conf = OmegaConf.merge(total_conf, update)
        return dict(base_default_conf=total_conf)

class BaseModel(nn.Module, metaclass=MetaModel):
    default_conf = {
        'name': None,
        'trainable': False,
    }
    required_data = []

    def __init__(self, conf):
        super().__init__()
        default_conf = OmegaConf.merge(
                self.base_default_conf, OmegaConf.create(self.default_conf))
        self.conf = conf = OmegaConf.merge(default_conf, conf)
        self._init(conf)
        if not conf.trainable:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, data):
        """Check the data and call the _forward method of the child model."""
        def recursive_key_check(expected, given):
            for key in expected:
                assert key in given, f'Missing key {key} in data'
                if isinstance(expected, dict):
                    recursive_key_check(expected[key], given[key])
        recursive_key_check(self.required_data, data)
        return self._forward(data)

    @abstractmethod
    def _init(self, conf):
        """To be implemented by child class."""
        raise NotImplementedError
    @abstractmethod
    def _forward(self, data):
        """To be implemented by child class."""
        raise NotImplementedError
    @abstractmethod
    def loss(self, pred, data):
        """To be implemented by child class."""
        raise NotImplementedError
    