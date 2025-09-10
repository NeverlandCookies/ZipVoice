#!/usr/bin/env python3
# Copyright    2024-2025  Xiaomi Corp.        (authors: Wei Kang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Voice Conversion Dataset for ZipVoice

This dataset is designed for training voice conversion models using self-reconstruction.
Each sample contains:
- Source features: The original speech features (for content encoding)
- Target features: The same speech features (for reconstruction target)
- Speech condition: Masked version of the same speech features (for speaker conditioning)
"""

import logging
import random
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F
from lhotse import CutSet
from lhotse.dataset.collation import collate_features
from lhotse.dataset.input_strategies import OnTheFlyFeatures
from torch.utils.data import Dataset

from zipvoice.utils.feature import VocosFbank


class VoiceConversionDataset(Dataset):
    """
    Voice Conversion Dataset for self-reconstruction training
    
    This dataset implements the self-reconstruction training strategy where:
    1. Source features = original speech features (for content encoding)
    2. Target features = original speech features (reconstruction target)
    3. Speech condition = masked original speech features (speaker conditioning)
    """
    
    def __init__(
        self,
        cuts: CutSet,
        feature_extractor: Optional[VocosFbank] = None,
        mask_prob: float = 0.8,  # Probability of masking each frame
        mask_length: int = 10,   # Length of consecutive masked segments
        max_duration: float = 20.0,  # Maximum duration in seconds
        shuffle: bool = True,
    ):
        """
        Args:
            cuts: Lhotse CutSet containing audio data
            feature_extractor: Feature extractor (VocosFbank)
            mask_prob: Probability of masking frames in speech condition
            mask_length: Length of consecutive masked segments
            max_duration: Maximum duration for filtering cuts
            shuffle: Whether to shuffle the dataset
        """
        super().__init__()
        
        self.cuts = cuts
        self.feature_extractor = feature_extractor or VocosFbank()
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.shuffle = shuffle
        
        # Filter cuts by duration
        if max_duration > 0:
            self.cuts = self.cuts.filter(lambda c: c.duration <= max_duration)
            
        # Convert to list for indexing
        self.cuts_list = list(self.cuts)
        
        logging.info(f"VoiceConversionDataset initialized with {len(self.cuts_list)} cuts")
        
    def __len__(self):
        return len(self.cuts_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cut = self.cuts_list[idx]
        
        # Extract features
        if hasattr(cut, 'features'):
            # Pre-computed features
            features = cut.load_features()
        else:
            # On-the-fly feature extraction
            audio = cut.load_audio()
            features = self.feature_extractor(audio)
        
        # Convert to tensor if needed
        if not isinstance(features, torch.Tensor):
            features = torch.from_numpy(features)
        
        features = features.float()  # [T, feat_dim]
        
        # Create masked speech condition
        speech_condition = self._create_masked_condition(features)
        
        return {
            'source_features': features.clone(),      # For content encoding
            'target_features': features.clone(),      # Reconstruction target
            'speech_condition': speech_condition,     # Masked features for speaker conditioning
            'cut_id': cut.id,
            'speaker_id': getattr(cut, 'speaker', 'unknown'),
        }
    
    def _create_masked_condition(self, features: torch.Tensor) -> torch.Tensor:
        """
        Create masked speech condition for speaker conditioning
        
        Args:
            features: Original features [T, feat_dim]
            
        Returns:
            Masked features [T, feat_dim]
        """
        T, feat_dim = features.shape
        masked_features = features.clone()
        
        if self.mask_prob <= 0:
            return masked_features
        
        # Create mask
        mask = torch.zeros(T, dtype=torch.bool)
        
        # Apply random masking with consecutive segments
        i = 0
        while i < T:
            if random.random() < self.mask_prob:
                # Mask a consecutive segment
                mask_end = min(i + self.mask_length, T)
                mask[i:mask_end] = True
                i = mask_end
            else:
                i += 1
        
        # Apply mask (set masked frames to zero)
        masked_features[mask] = 0.0
        
        return masked_features


class VoiceConversionCollate:
    """Collate function for Voice Conversion dataset"""
    
    def __init__(self, pad_value: float = 0.0):
        self.pad_value = pad_value
    
    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of Voice Conversion samples
        
        Args:
            batch: List of samples from VoiceConversionDataset
            
        Returns:
            Batched tensors with proper padding and lengths
        """
        # Extract individual components
        source_features = [item['source_features'] for item in batch]
        target_features = [item['target_features'] for item in batch]
        speech_condition = [item['speech_condition'] for item in batch]
        cut_ids = [item['cut_id'] for item in batch]
        speaker_ids = [item['speaker_id'] for item in batch]
        
        # Get lengths before padding
        source_lens = torch.tensor([f.shape[0] for f in source_features])
        target_lens = torch.tensor([f.shape[0] for f in target_features])
        speech_condition_lens = torch.tensor([f.shape[0] for f in speech_condition])
        
        # Pad sequences
        source_features_padded = self._pad_sequences(source_features)
        target_features_padded = self._pad_sequences(target_features)
        speech_condition_padded = self._pad_sequences(speech_condition)
        
        return {
            'source_features': source_features_padded,
            'source_features_lens': source_lens,
            'target_features': target_features_padded,
            'target_features_lens': target_lens,
            'speech_condition': speech_condition_padded,
            'speech_condition_lens': speech_condition_lens,
            'cut_ids': cut_ids,
            'speaker_ids': speaker_ids,
        }
    
    def _pad_sequences(self, sequences):
        """Pad sequences to same length"""
        return torch.nn.utils.rnn.pad_sequence(
            sequences, 
            batch_first=True, 
            padding_value=self.pad_value
        )


def _test_vc_dataset():
    """Test function for Voice Conversion dataset"""
    from lhotse import CutSet
    from lhotse.testing.dummies import DummyManifest
    
    # Create dummy cuts
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    
    # Create dataset
    dataset = VoiceConversionDataset(cuts, mask_prob=0.5, max_duration=10.0)
    
    # Test single sample
    sample = dataset[0]
    print("Sample keys:", sample.keys())
    print("Source features shape:", sample['source_features'].shape)
    print("Target features shape:", sample['target_features'].shape)
    print("Speech condition shape:", sample['speech_condition'].shape)
    
    # Test collate function
    collate_fn = VoiceConversionCollate()
    batch = collate_fn([dataset[i] for i in range(3)])
    
    print("\nBatch keys:", batch.keys())
    print("Batch source features shape:", batch['source_features'].shape)
    print("Batch target features shape:", batch['target_features'].shape)
    print("Batch speech condition shape:", batch['speech_condition'].shape)
    print("Batch lengths:", batch['source_features_lens'])
    
    print("VoiceConversionDataset test passed!")


if __name__ == "__main__":
    _test_vc_dataset()
