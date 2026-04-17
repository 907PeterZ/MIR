import torch
import logging
from torch import nn
from .__init__ import methods_map
import os

__all__ = ['ModelManager']

class MIA(nn.Module):

    def __init__(self, args):

        super(MIA, self).__init__()

        fusion_method = methods_map[args.method]
        self.model = fusion_method(args)

    def forward(self, text_feats, video_feats, audio_feats):

        video_feats, audio_feats = video_feats.float(), audio_feats.float()
        mm_model = self.model(text_feats, video_feats, audio_feats)

        return mm_model
        
class ModelManager:

    def __init__(self, args):
        
        self.logger = logging.getLogger(args.logger_name)
        self.device = args.device = torch.device('cuda:%d' % int(args.gpu_id) if torch.cuda.is_available() else 'cpu')
        self.model = self._set_model(args)

        if hasattr(args, 'pretrained_model_path') and args.pretrained_model_path:
            self._load_pretrained_model(args.pretrained_model_path)

    def _set_model(self, args):
        model = MIA(args) 
        model.to(self.device)
        return model

    def _load_pretrained_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Pretrained model not found at {model_path}")
        
        self.logger.info(f"Loading pretrained model from: {model_path}")
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=True)
        self.logger.info("✅ Pretrained model loaded successfully.")