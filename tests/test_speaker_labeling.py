#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import os
import time
import torch
import unittest
import numpy as np
from unittest.mock import MagicMock
from funasr.auto.auto_model import AutoModel
from funasr.utils.speaker_manager import SpeakerManager

class TestSpeakerLabeling(unittest.TestCase):
    """测试说话人标注功能"""
    
    def setUp(self):
        """初始化测试环境"""
        # Initialize speaker manager directly for testing
        self.speaker_manager = SpeakerManager(similarity_threshold=0.75)
        
    def test_speaker_labeling(self):
        """测试实时说话人标注"""
        # Test direct speaker manager functionality
        embedding1 = torch.randn(192)  # First speaker embedding
        embedding2 = torch.randn(192)  # Second speaker embedding
        
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
        
    def test_performance(self):
        """测试性能影响"""
        # Test speaker manager performance with many embeddings
        embeddings = [torch.randn(192) for _ in range(100)]
        
        t0 = time.time()
        for emb in embeddings:
            self.speaker_manager.get_speaker_id(emb)
        time_total = time.time() - t0
        
        # Performance should be reasonable
        time_per_embedding = time_total / len(embeddings)
        self.assertLess(time_per_embedding, 0.01)  # Less than 10ms per embedding

if __name__ == "__main__":
    unittest.main()
