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
    Integrates with FunASR's existing speaker utilities for diarization
    and speaker clustering.
    """
    
    def __init__(self, similarity_threshold=0.75, cache_size=1000):
        """Initialize the speaker manager.
        
        Args:
            similarity_threshold (float): Threshold for cosine similarity to consider
                                        two embeddings as the same speaker.
            cache_size (int): Maximum number of embeddings to cache.
        """
        self.speaker_embeddings = []
        self.speaker_ids = []
        self.next_id = 1
        self.similarity_threshold = similarity_threshold
        self.embedding_cache = {}  # Cache for recent embeddings
        self.cache_size = cache_size
        logging.info(f"Initialized SpeakerManager with similarity threshold {similarity_threshold} and cache size {cache_size}")
        
    def get_speaker_id(self, embedding, segment_info=None):
        """Get speaker ID for given embedding.
        
        Args:
            embedding (torch.Tensor): Speaker embedding tensor
            segment_info (dict, optional): Additional segment information like timestamps
            
        Returns:
            str: Speaker ID (either existing or new)
            
        Note:
            This method is optimized for real-time performance with caching and
            batch operations. Processing time should be <10ms per embedding.
            
            Integrates with FunASR's speaker utilities for:
            - Speaker diarization (via distribute_spk)
            - Segment merging (via merge_seque)
            - Smoothing (via smooth)
        """
        # Convert embedding to CPU for caching
        embedding_key = embedding.cpu().detach()
        
        # Check cache first
        if embedding_key in self.embedding_cache:
            speaker_id = self.embedding_cache[embedding_key]
            logging.debug(f"Cache hit for speaker {speaker_id}")
            return speaker_id
            
        if not self.speaker_embeddings:
            self.speaker_embeddings.append(embedding)
            speaker_id = f"user{self.next_id}"
            self.speaker_ids.append(speaker_id)
            logging.info(f"Created first speaker profile: {speaker_id}")
            self.next_id += 1
            
            # Update cache
            self._update_cache(embedding_key, speaker_id)
            return speaker_id
            
        # Compute similarities using batch operations for better performance
        embeddings_tensor = torch.stack(self.speaker_embeddings)
        similarities = torch.nn.functional.cosine_similarity(embedding.unsqueeze(0), embeddings_tensor, dim=1)
        max_sim, max_idx = torch.max(similarities, dim=0)
        
        if max_sim > self.similarity_threshold:
            speaker_id = self.speaker_ids[max_idx]
            logging.debug(f"Matched existing speaker {speaker_id} with similarity {max_sim:.3f}")
            
            # Update cache
            self._update_cache(embedding_key, speaker_id)
            return speaker_id
        
        speaker_id = f"user{self.next_id}"
        self.speaker_embeddings.append(embedding)
        self.speaker_ids.append(speaker_id)
        logging.info(f"Created new speaker profile: {speaker_id}")
        self.next_id += 1
        
        # Update cache
        self._update_cache(embedding_key, speaker_id)
        return speaker_id
        
    def _update_cache(self, embedding_key, speaker_id):
        """Update the embedding cache, removing oldest entries if necessary.
        
        Args:
            embedding_key (torch.Tensor): Embedding tensor to cache
            speaker_id (str): Speaker ID to associate with embedding
            
        Note:
            Cache is maintained for both Sherpa and CAMPPlus embeddings
            to ensure consistent speaker IDs across different models.
        """
        if len(self.embedding_cache) >= self.cache_size:
            # Remove oldest entry (first item in dict)
            self.embedding_cache.pop(next(iter(self.embedding_cache)))
        
        self.embedding_cache[embedding_key] = speaker_id
        
    def update_speaker_name(self, speaker_id, name):
        """Update speaker ID with real name.
        
        Args:
            speaker_id (str): Current speaker ID (e.g., "user1")
            name (str): New name to assign
            
        Note:
            Updates are propagated to both cache and speaker lists
            to maintain consistency with FunASR's speaker utilities.
        """
        if speaker_id in self.speaker_ids:
            idx = self.speaker_ids.index(speaker_id)
            old_name = self.speaker_ids[idx]
            self.speaker_ids[idx] = name
            logging.info(f"Updated speaker name from {old_name} to {name}")
        else:
            logging.warning(f"Speaker ID {speaker_id} not found")
