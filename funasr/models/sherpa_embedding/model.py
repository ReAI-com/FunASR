#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import torch
import sherpa
import logging
from funasr.register import tables

@tables.register("model_classes", "SherpaEmbedding")
class SherpaEmbeddingModel(torch.nn.Module):
    """Wrapper for Sherpa's SpeakerEmbeddingExtractor.
    
    This model provides speaker embedding extraction using Sherpa's implementation.
    """
    
    def __init__(self, model_path, device="cpu", **kwargs):
        super().__init__()
        config = sherpa.SpeakerEmbeddingExtractorConfig(
            model=model_path,
            use_gpu=(device != "cpu"),
        )
        self.extractor = sherpa.SpeakerEmbeddingExtractor(config)
        self.device = device
        
    def forward(self, x):
        """Extract speaker embedding from input audio.
        
        Args:
            x (torch.Tensor): Input audio waveform tensor
            
        Returns:
            torch.Tensor: Speaker embedding tensor
        """
        # Convert to CPU numpy array for Sherpa
        x_np = x.cpu().numpy()
        
        # Create stream and compute embedding
        stream = self.extractor.create_stream()
        stream.accept_waveform(x_np)
        embedding = self.extractor.compute(stream)
        
        # Convert back to torch tensor on correct device
        return torch.from_numpy(embedding).to(self.device)
        
    def inference(self, data_in, data_lengths=None, key=None, **kwargs):
        """Inference interface matching FunASR's requirements.
        
        Args:
            data_in: Input audio data
            data_lengths: Audio lengths
            key: Optional key for batch items
            **kwargs: Additional arguments
            
        Returns:
            tuple: (results, meta_data)
        """
        embedding = self.forward(data_in)
        results = [{"spk_embedding": embedding}]
        meta_data = {}
        return results, meta_data
