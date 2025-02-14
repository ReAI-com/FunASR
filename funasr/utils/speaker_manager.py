#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import torch
import numpy as np
import logging

class SpeakerManager:
    """Manages speaker profiles and identification.
    
    This class maintains a list of known speakers and their embeddings,
    providing functionality to identify speakers and update their names.
    """
    
    def __init__(self, similarity_threshold=0.75):
        """Initialize the speaker manager.
        
        Args:
            similarity_threshold (float): Threshold for cosine similarity to consider
                                        two embeddings as the same speaker.
        """
        self.speaker_embeddings = []
        self.speaker_ids = []
        self.next_id = 1
        self.similarity_threshold = similarity_threshold
        logging.info(f"Initialized SpeakerManager with similarity threshold {similarity_threshold}")
        
    def get_speaker_id(self, embedding):
        """Get speaker ID for given embedding.
        
        Args:
            embedding (torch.Tensor): Speaker embedding tensor
            
        Returns:
            str: Speaker ID (either existing or new)
        """
        if not self.speaker_embeddings:
            self.speaker_embeddings.append(embedding)
            self.speaker_ids.append(f"user{self.next_id}")
            logging.info(f"Created first speaker profile: user{self.next_id}")
            self.next_id += 1
            return self.speaker_ids[-1]
            
        similarities = [
            torch.nn.functional.cosine_similarity(embedding, e, dim=0)
            for e in self.speaker_embeddings
        ]
        max_sim, max_idx = max(similarities), np.argmax(similarities)
        
        if max_sim > self.similarity_threshold:
            logging.debug(f"Matched existing speaker {self.speaker_ids[max_idx]} with similarity {max_sim:.3f}")
            return self.speaker_ids[max_idx]
        
        self.speaker_embeddings.append(embedding)
        self.speaker_ids.append(f"user{self.next_id}")
        logging.info(f"Created new speaker profile: user{self.next_id}")
        self.next_id += 1
        return self.speaker_ids[-1]
        
    def update_speaker_name(self, speaker_id, name):
        """Update speaker ID with real name.
        
        Args:
            speaker_id (str): Current speaker ID (e.g., "user1")
            name (str): New name to assign
        """
        if speaker_id in self.speaker_ids:
            idx = self.speaker_ids.index(speaker_id)
            old_name = self.speaker_ids[idx]
            self.speaker_ids[idx] = name
            logging.info(f"Updated speaker name from {old_name} to {name}")
        else:
            logging.warning(f"Speaker ID {speaker_id} not found")
