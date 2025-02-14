#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import torch
import sherpa
import logging
from contextlib import contextmanager
from distutils.version import LooseVersion
from funasr.register import tables

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield

@tables.register("model_classes", "SherpaEmbedding")
class SherpaEmbeddingModel(torch.nn.Module):
    """Wrapper for Sherpa's SpeakerEmbeddingExtractor.
    
    This model provides speaker embedding extraction using Sherpa's implementation.
    """
    
    def __init__(self, model_path, device="cpu", **kwargs):
        super().__init__()
        self.device = device
        config = sherpa.SpeakerEmbeddingExtractorConfig(
            model=model_path,
            use_gpu=(str(device) != "cpu"),
        )
        self.extractor = sherpa.SpeakerEmbeddingExtractor(config)
        
        # Initialize weights if any trainable parameters
        for m in self.modules():
            if isinstance(m, (torch.nn.Conv1d, torch.nn.Linear)):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        
    def forward(self, x):
        """Extract speaker embedding from input audio.
        
        Args:
            x (torch.Tensor): Input audio waveform tensor
            
        Returns:
            torch.Tensor: Speaker embedding tensor
        """
        with autocast(enabled=False):  # Ensure fp32 for embedding extraction
            # Convert to CPU numpy array for Sherpa
            x_np = x.detach().cpu().numpy()
            
            # Create stream and compute embedding
            stream = self.extractor.create_stream()
            stream.accept_waveform(x_np)
            embedding = self.extractor.compute(stream)
            
            # Convert back to torch tensor on correct device
            embedding_tensor = torch.from_numpy(embedding).to(self.device)
            
            if self.device.type == "cuda":
                # Ensure contiguous memory layout for CUDA operations
                embedding_tensor = embedding_tensor.contiguous()
            
            return embedding_tensor
        
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
