#!/usr/bin/env python3
# Copyright    2024-2025  Xiaomi Corp.        (authors: Wei Kang, Fangjun Kuang)
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
ZipVoice Voice Conversion Model

This model adapts ZipVoice for voice conversion by:
1. Replacing text encoder with pre-trained Zipformer (ASR model) as content encoder
2. Using CTC linear layer outputs as content features
3. Maintaining the original Flow Matching decoder for speech generation
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from zipvoice.models.modules.zipformer import Zipformer
from zipvoice.models.modules.solver import FlowMatchingSolver
from zipvoice.utils.common import (
    get_tokens_index,
    prepare_avg_tokens_durations,
)


class ZipVoiceVC(nn.Module):
    """
    ZipVoice Voice Conversion Model
    
    Architecture:
    - Content Encoder: Pre-trained Zipformer (from ASR model)
    - Speech Condition: VocosFbank features (same as original ZipVoice)
    - Flow Matching Decoder: Same as original ZipVoice
    """
    
    def __init__(
        self,
        # Content encoder (Zipformer) parameters
        content_encoder_dim: int = 512,
        content_encoder_downsampling_factor: List[int] = [1, 2, 4, 2, 1],
        content_encoder_num_layers: List[int] = [2, 2, 4, 4, 4],
        content_encoder_feedforward_dim: int = 1536,
        content_encoder_num_heads: int = 4,
        content_encoder_cnn_module_kernel: List[int] = [31, 15, 7, 15, 31],
        
        # CTC head parameters
        vocab_size: int = 500,  # Vocabulary size for CTC
        
        # Flow matching decoder parameters (same as original ZipVoice)
        fm_decoder_downsampling_factor: List[int] = [1, 2, 4, 2, 1],
        fm_decoder_num_layers: List[int] = [2, 2, 4, 4, 4],
        fm_decoder_cnn_module_kernel: List[int] = [31, 15, 7, 15, 31],
        fm_decoder_feedforward_dim: int = 1536,
        fm_decoder_num_heads: int = 4,
        fm_decoder_dim: int = 512,
        
        # Other parameters
        query_head_dim: int = 32,
        value_head_dim: int = 12,
        pos_head_dim: int = 4,
        pos_dim: int = 48,
        time_embed_dim: int = 192,
        content_embed_dim: int = 192,  # Renamed from text_embed_dim
        feat_dim: int = 100,
    ):
        super().__init__()
        
        self.feat_dim = feat_dim
        self.content_embed_dim = content_embed_dim
        self.vocab_size = vocab_size
        
        # Content encoder: Pre-trained Zipformer
        self.content_encoder = Zipformer(
            encoder_dim=content_encoder_dim,
            downsampling_factor=content_encoder_downsampling_factor,
            num_encoder_layers=content_encoder_num_layers,
            feedforward_dim=content_encoder_feedforward_dim,
            num_heads=content_encoder_num_heads,
            cnn_module_kernel=content_encoder_cnn_module_kernel,
            query_head_dim=query_head_dim,
            value_head_dim=value_head_dim,
            pos_head_dim=pos_head_dim,
            pos_dim=pos_dim,
        )
        
        # CTC head for content encoder
        self.ctc_head = nn.Linear(content_encoder_dim, vocab_size)
        
        # Content embedding layer (converts CTC posteriors to content embeddings)
        self.content_embedding = nn.Linear(vocab_size, content_embed_dim)
        
        # Flow matching solver (same as original ZipVoice)
        self.solver = FlowMatchingSolver(
            downsampling_factor=fm_decoder_downsampling_factor,
            num_layers=fm_decoder_num_layers,
            cnn_module_kernel=fm_decoder_cnn_module_kernel,
            feedforward_dim=fm_decoder_feedforward_dim,
            num_heads=fm_decoder_num_heads,
            encoder_dim=fm_decoder_dim,
            query_head_dim=query_head_dim,
            value_head_dim=value_head_dim,
            pos_head_dim=pos_head_dim,
            pos_dim=pos_dim,
            time_embed_dim=time_embed_dim,
            content_embed_dim=content_embed_dim,
            feat_dim=feat_dim,
        )
        
    def load_pretrained_content_encoder(self, checkpoint_path: str, freeze: bool = True):
        """
        Load pre-trained Zipformer weights from icefall ASR model
        
        Args:
            checkpoint_path: Path to the pre-trained ASR model checkpoint
            freeze: Whether to freeze the content encoder parameters
        """
        logging.info(f"Loading pre-trained content encoder from {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Extract Zipformer and CTC head parameters
        model_state = checkpoint.get("model", checkpoint)
        
        # Load Zipformer parameters
        content_encoder_state = {}
        ctc_head_state = {}
        
        for key, value in model_state.items():
            if key.startswith("encoder."):
                # Remove "encoder." prefix for Zipformer
                new_key = key[8:]  # Remove "encoder."
                content_encoder_state[new_key] = value
            elif key.startswith("ctc_head."):
                # Remove "ctc_head." prefix
                new_key = key[9:]  # Remove "ctc_head."
                ctc_head_state[new_key] = value
        
        # Load states
        self.content_encoder.load_state_dict(content_encoder_state, strict=False)
        self.ctc_head.load_state_dict(ctc_head_state, strict=False)
        
        # Freeze parameters if requested
        if freeze:
            logging.info("Freezing content encoder parameters")
            for param in self.content_encoder.parameters():
                param.requires_grad = False
            for param in self.ctc_head.parameters():
                param.requires_grad = False
        
        logging.info("Successfully loaded pre-trained content encoder")
    
    def forward_content_condition(
        self, 
        source_features: torch.Tensor, 
        source_features_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract content condition from source speech features
        
        Args:
            source_features: Source speech features [B, T, feat_dim]
            source_features_lens: Length of source features [B]
            
        Returns:
            content_condition: Content condition [B, T', content_embed_dim]
            content_condition_lens: Length of content condition [B]
        """
        # Extract content features using Zipformer
        content_features, content_features_lens = self.content_encoder(
            source_features, source_features_lens
        )
        
        # Get CTC posteriors
        ctc_logits = self.ctc_head(content_features)  # [B, T', vocab_size]
        ctc_posteriors = F.softmax(ctc_logits, dim=-1)
        
        # Convert to content embeddings
        content_condition = self.content_embedding(ctc_posteriors)  # [B, T', content_embed_dim]
        
        return content_condition, content_features_lens
    
    def expand_content_condition(
        self, 
        content_condition: torch.Tensor,
        content_condition_lens: torch.Tensor,
        target_features_lens: torch.Tensor
    ) -> torch.Tensor:
        """
        Expand content condition to match target feature length
        
        This is similar to the original ZipVoice's text condition expansion,
        but adapted for content features from speech.
        
        Args:
            content_condition: Content condition [B, T_content, content_embed_dim]
            content_condition_lens: Length of content condition [B]
            target_features_lens: Target feature lengths [B]
            
        Returns:
            expanded_content_condition: [B, T_target, content_embed_dim]
        """
        batch_size = content_condition.shape[0]
        max_target_len = target_features_lens.max().item()
        
        # Prepare durations for content frames (similar to token durations)
        content_durations = prepare_avg_tokens_durations(
            target_features_lens, content_condition_lens
        )
        
        # Get content index mapping
        content_index = get_tokens_index(content_durations, max_target_len)
        content_index = content_index.to(content_condition.device)
        
        # Expand content condition
        expanded_content_condition = torch.gather(
            content_condition,
            dim=1,
            index=content_index.unsqueeze(-1).expand(
                batch_size, max_target_len, self.content_embed_dim
            ),
        )
        
        return expanded_content_condition
    
    def forward(
        self,
        source_features: torch.Tensor,
        source_features_lens: torch.Tensor,
        target_features: torch.Tensor,
        target_features_lens: torch.Tensor,
        speech_condition: torch.Tensor,
        speech_condition_lens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training
        
        Args:
            source_features: Source speech features [B, T_src, feat_dim]
            source_features_lens: Source feature lengths [B]
            target_features: Target speech features [B, T_tgt, feat_dim]
            target_features_lens: Target feature lengths [B]
            speech_condition: Speech condition features [B, T_tgt, feat_dim]
            speech_condition_lens: Speech condition lengths [B]
            
        Returns:
            Dictionary containing loss and other metrics
        """
        # Extract content condition from source features
        content_condition, content_condition_lens = self.forward_content_condition(
            source_features, source_features_lens
        )
        
        # Expand content condition to match target length
        expanded_content_condition = self.expand_content_condition(
            content_condition, content_condition_lens, target_features_lens
        )
        
        # Flow matching forward pass
        outputs = self.solver(
            features=target_features,
            features_lens=target_features_lens,
            speech_condition=speech_condition,
            speech_condition_lens=speech_condition_lens,
            content_condition=expanded_content_condition,
        )
        
        return outputs
    
    def inference(
        self,
        source_features: torch.Tensor,
        source_features_lens: torch.Tensor,
        speech_condition: torch.Tensor,
        speech_condition_lens: torch.Tensor,
        num_steps: int = 16,
        guidance_scale: float = 1.0,
        t_shift: float = 0.0,
    ) -> torch.Tensor:
        """
        Inference for voice conversion
        
        Args:
            source_features: Source speech features [B, T_src, feat_dim]
            source_features_lens: Source feature lengths [B]
            speech_condition: Target speaker condition [B, T_tgt, feat_dim]
            speech_condition_lens: Speech condition lengths [B]
            num_steps: Number of flow matching steps
            guidance_scale: Guidance scale for generation
            t_shift: Time shift parameter
            
        Returns:
            Generated features [B, T_tgt, feat_dim]
        """
        # Extract content condition
        content_condition, content_condition_lens = self.forward_content_condition(
            source_features, source_features_lens
        )
        
        # Expand content condition to match speech condition length
        expanded_content_condition = self.expand_content_condition(
            content_condition, content_condition_lens, speech_condition_lens
        )
        
        # Flow matching inference
        generated_features = self.solver.inference(
            speech_condition=speech_condition,
            speech_condition_lens=speech_condition_lens,
            content_condition=expanded_content_condition,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            t_shift=t_shift,
        )
        
        return generated_features


def _test_zipvoice_vc():
    """Test function for ZipVoiceVC model"""
    
    # Model configuration
    model = ZipVoiceVC(
        content_encoder_dim=512,
        vocab_size=500,
        fm_decoder_dim=512,
        feat_dim=100,
    )
    
    # Test data
    batch_size = 2
    src_len = 150
    tgt_len = 200
    feat_dim = 100
    
    source_features = torch.randn(batch_size, src_len, feat_dim)
    source_features_lens = torch.tensor([src_len, src_len - 20])
    
    target_features = torch.randn(batch_size, tgt_len, feat_dim)
    target_features_lens = torch.tensor([tgt_len, tgt_len - 30])
    
    speech_condition = torch.randn(batch_size, tgt_len, feat_dim)
    speech_condition_lens = target_features_lens.clone()
    
    # Test forward pass
    outputs = model(
        source_features=source_features,
        source_features_lens=source_features_lens,
        target_features=target_features,
        target_features_lens=target_features_lens,
        speech_condition=speech_condition,
        speech_condition_lens=speech_condition_lens,
    )
    
    print("Forward pass outputs:", outputs.keys())
    
    # Test inference
    generated = model.inference(
        source_features=source_features,
        source_features_lens=source_features_lens,
        speech_condition=speech_condition,
        speech_condition_lens=speech_condition_lens,
        num_steps=8,
    )
    
    print(f"Generated features shape: {generated.shape}")
    print("ZipVoiceVC test passed!")


if __name__ == "__main__":
    _test_zipvoice_vc()
