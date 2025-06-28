from models.base_model import BaseModel
from models.d2s import D2S
from models.l2s import L2S

class PL2Map(BaseModel):
    default_conf = {
        'trainable': True,
    }
    required_data = ['points_descriptors', 'lines_descriptors']

    def _init(self, conf):
        self.d2s = D2S(conf={})
        self.l2s = L2S(conf={})
        
    def _forward(self, data):
        pred_points = self.d2s(data)
        pred_lines = self.l2s(data)
        return {**pred_points, **pred_lines}
    
    def loss(self, pred, data):
        pass

