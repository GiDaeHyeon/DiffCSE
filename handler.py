"""
DiffCSE torchserve handler

Author: DaeHyeon Gi
"""

from pathlib import Path

import torch
from ts.torch_handler.base_handler import BaseHandler

import model as MODEL


class Handler(BaseHandler):
    """Model handler for torchserve-model-archiver"""
    def initialize(self, context):
        # load the model
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + \
            str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = Path(model_dir) / serialized_file
        if not model_pt_path.exists():
            raise RuntimeError("Missing the model.pt file")
        hparams = {
            'checkpoint': model_pt_path
        }
        self.model = MODEL(hparams, is_train=False).to(self.device)

        self.initialized = True

    def preprocess(self, data):
        inp = [d.get("body").decode('utf-8') for d in data]
        return inp

    def inference(self, data, *args, **kwargs):
        """
        The Inference Function is used to make a prediction call on the given input request.
        The user needs to override the inference function to customize it.
        Args:
            data (Torch Tensor): A Torch Tensor is passed to make the Inference Request.
            The shape should match the model input shape.
        Returns:
            Torch Tensor : The Predicted Torch Tensor is returned in this function.
        """
        with torch.no_grad():
            results = self.model(data)
        return results

    def postprocess(self, data):
        return data.tolist()
