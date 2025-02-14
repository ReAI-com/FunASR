#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import os
import time
import torch
import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from funasr.auto.auto_model import AutoModel
from funasr.utils.speaker_manager import SpeakerManager

# Mock sherpa module
mock_sherpa = MagicMock()
mock_sherpa.SpeakerEmbeddingExtractorConfig = MagicMock()
mock_sherpa.SpeakerEmbeddingExtractor = MagicMock()
mock_sherpa.SpeakerEmbeddingExtractor().compute.return_value = np.random.randn(192)
mock_sherpa.SpeakerEmbeddingExtractor().create_stream = MagicMock()


class TestSpeakerLabeling(unittest.TestCase):
    """测试说话人标注功能"""
    
    def setUp(self):
        """初始化测试环境"""
        # Initialize speaker manager directly for testing
        self.speaker_manager = SpeakerManager(similarity_threshold=0.75, cache_size=100)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def test_speaker_labeling(self):
        """测试实时说话人标注"""
        # Test direct speaker manager functionality
        embedding1 = torch.randn(192, device=self.device)  # First speaker embedding
        embedding2 = torch.randn(192, device=self.device)  # Second speaker embedding
        
        # Should assign first speaker ID
        speaker1 = self.speaker_manager.get_speaker_id(embedding1)
        self.assertEqual(speaker1, "user1")
        
        # Should assign different ID to second speaker
        speaker2 = self.speaker_manager.get_speaker_id(embedding2)
        self.assertEqual(speaker2, "user2")
        
        # Should recognize same speaker
        speaker1_again = self.speaker_manager.get_speaker_id(embedding1)
        self.assertEqual(speaker1_again, "user1")
        
        # Test speaker name update
        self.speaker_manager.update_speaker_name("user1", "测试说话人")
        speaker1_updated = self.speaker_manager.get_speaker_id(embedding1)
        self.assertEqual(speaker1_updated, "测试说话人")
        
        # Test cache functionality
        cached_speaker = self.speaker_manager.get_speaker_id(embedding1)
        self.assertEqual(cached_speaker, "测试说话人")
        
        # Test cache size limit
        for i in range(150):  # More than cache_size
            emb = torch.randn(192, device=self.device)
            self.speaker_manager.get_speaker_id(emb)
        self.assertLessEqual(len(self.speaker_manager.embedding_cache), 100)
        
    def test_performance(self):
        """测试性能影响"""
        # Test speaker manager performance with many embeddings
        embeddings = [torch.randn(192, device=self.device) for _ in range(100)]
        
        # Warm-up run
        for _ in range(10):
            self.speaker_manager.get_speaker_id(embeddings[0])
            
        # Test batch performance
        t0 = time.time()
        for emb in embeddings:
            self.speaker_manager.get_speaker_id(emb)
        time_total = time.time() - t0
        
        # Performance should be reasonable
        time_per_embedding = time_total / len(embeddings)
        self.assertLess(time_per_embedding, 0.01)  # Less than 10ms per embedding
        
    @patch.dict('sys.modules', {'sherpa': mock_sherpa})
    def test_model_integration(self):
        """测试模型集成"""
        # Mock the model registration
        from funasr.register import tables
        from funasr.models.sherpa_embedding.model import SherpaEmbeddingModel
        tables.model_classes["SherpaEmbedding"] = SherpaEmbeddingModel
        
        # Test integration with AutoModel
        kwargs = {
            "model": "SherpaEmbedding",
            "model_path": "path/to/model",
            "device": self.device,
            "speaker_similarity_threshold": 0.8,
            "model_conf": {}  # Skip model download
        }
        model = AutoModel(**kwargs)
        
        # Test speaker manager initialization
        self.assertTrue(hasattr(model, "speaker_manager"))
        self.assertEqual(model.speaker_manager.similarity_threshold, 0.8)
        
        # Test speaker name updates
        model.update_speaker_name("user1", "张三")
        self.assertIn("张三", model.speaker_manager.speaker_ids)

if __name__ == "__main__":
    unittest.main()
