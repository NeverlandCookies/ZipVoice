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
Inference script for ZipVoice Voice Conversion

This script performs voice conversion using a trained ZipVoiceVC model.
It converts the content of source speech to the voice of target speaker.
"""

import argparse
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from vocos import Vocos

from zipvoice.models.zipvoice_vc import ZipVoiceVC
from zipvoice.utils.common import setup_logger
from zipvoice.utils.feature import VocosFbank


def get_args():
    parser = argparse.ArgumentParser(
        description="ZipVoice Voice Conversion Inference"
    )
    
    # Model configuration
    parser.add_argument(
        "--model-name",
        type=str,
        default="zipvoice_vc",
        help="Model name (zipvoice_vc).",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing the model checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        required=True,
        help="Name of the checkpoint file.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="conf/zipvoice_vc_base.json",
        help="Path to model configuration file.",
    )
    
    # Input/Output
    parser.add_argument(
        "--source-audio",
        type=str,
        help="Path to source audio file (for single conversion).",
    )
    parser.add_argument(
        "--target-audio",
        type=str,
        help="Path to target speaker audio file (for single conversion).",
    )
    parser.add_argument(
        "--test-list",
        type=str,
        help="Path to test list TSV file (for batch conversion).",
    )
    parser.add_argument(
        "--res-dir",
        type=str,
        default="results/vc_inference",
        help="Directory to save converted audio files.",
    )
    
    # Inference parameters
    parser.add_argument(
        "--num-step",
        type=int,
        default=16,
        help="Number of flow matching steps.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
        help="Guidance scale for generation.",
    )
    parser.add_argument(
        "--t-shift",
        type=float,
        default=0.0,
        help="Time shift parameter.",
    )
    
    # Audio processing
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=24000,
        help="Sampling rate for audio processing.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Maximum duration for input audio (seconds).",
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on.",
    )
    
    return parser.parse_args()


class ZipVoiceVCInference:
    """ZipVoice Voice Conversion Inference Engine"""
    
    def __init__(
        self,
        model_dir: str,
        checkpoint_name: str,
        model_config: str,
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.model_dir = model_dir
        self.checkpoint_name = checkpoint_name
        
        # Load model configuration
        with open(model_config, 'r') as f:
            config = json.load(f)
        
        # Create and load model
        self.model = self._load_model(config)
        
        # Create feature extractor and vocoder
        self.feature_extractor = VocosFbank()
        self.vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        self.vocoder = self.vocoder.to(self.device)
        
        logging.info(f"ZipVoiceVC inference engine initialized on {self.device}")
    
    def _load_model(self, config: Dict) -> ZipVoiceVC:
        """Load the trained model"""
        
        # Create model
        model_config = config["model"]
        model = ZipVoiceVC(**model_config)
        
        # Load checkpoint
        checkpoint_path = os.path.join(self.model_dir, self.checkpoint_name)
        logging.info(f"Loading model from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def load_audio(self, audio_path: str, max_duration: float = 30.0) -> torch.Tensor:
        """Load and preprocess audio"""
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=24000)
        
        # Trim silence
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Limit duration
        if len(audio) > max_duration * sr:
            audio = audio[:int(max_duration * sr)]
            logging.warning(f"Audio truncated to {max_duration}s")
        
        return torch.from_numpy(audio).float()
    
    def extract_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract VocosFbank features from audio"""
        
        # Add batch dimension if needed
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Extract features
        features = self.feature_extractor(audio)
        
        # Convert to tensor if needed
        if not isinstance(features, torch.Tensor):
            features = torch.from_numpy(features)
        
        return features.float()
    
    def convert_voice(
        self,
        source_audio: torch.Tensor,
        target_audio: torch.Tensor,
        num_steps: int = 16,
        guidance_scale: float = 1.0,
        t_shift: float = 0.0,
    ) -> torch.Tensor:
        """
        Perform voice conversion
        
        Args:
            source_audio: Source speech audio
            target_audio: Target speaker audio (for speaker conditioning)
            num_steps: Number of flow matching steps
            guidance_scale: Guidance scale
            t_shift: Time shift parameter
            
        Returns:
            Converted audio
        """
        
        with torch.no_grad():
            # Extract features
            source_features = self.extract_features(source_audio)
            target_features = self.extract_features(target_audio)
            
            # Move to device
            source_features = source_features.to(self.device)
            target_features = target_features.to(self.device)
            
            # Add batch dimension
            if source_features.dim() == 2:
                source_features = source_features.unsqueeze(0)
            if target_features.dim() == 2:
                target_features = target_features.unsqueeze(0)
            
            # Get lengths
            source_lens = torch.tensor([source_features.shape[1]], device=self.device)
            target_lens = torch.tensor([target_features.shape[1]], device=self.device)
            
            # Use target features as speech condition
            speech_condition = target_features
            speech_condition_lens = target_lens
            
            # Perform inference
            generated_features = self.model.inference(
                source_features=source_features,
                source_features_lens=source_lens,
                speech_condition=speech_condition,
                speech_condition_lens=speech_condition_lens,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                t_shift=t_shift,
            )
            
            # Convert features to audio using vocoder
            generated_audio = self.vocoder.decode(generated_features.transpose(1, 2))
            
            return generated_audio.squeeze(0).cpu()
    
    def convert_single(
        self,
        source_path: str,
        target_path: str,
        output_path: str,
        max_duration: float = 30.0,
        **kwargs
    ):
        """Convert a single audio file"""
        
        logging.info(f"Converting: {source_path} -> {target_path}")
        
        # Load audio files
        source_audio = self.load_audio(source_path, max_duration)
        target_audio = self.load_audio(target_path, max_duration)
        
        # Perform conversion
        converted_audio = self.convert_voice(source_audio, target_audio, **kwargs)
        
        # Save result
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, converted_audio.numpy(), 24000)
        
        logging.info(f"Saved converted audio: {output_path}")
    
    def convert_batch(
        self,
        test_list: str,
        output_dir: str,
        max_duration: float = 30.0,
        **kwargs
    ):
        """Convert a batch of audio files from test list"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Read test list
        with open(test_list, 'r') as f:
            lines = f.readlines()
        
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split('\t')
            if len(parts) < 3:
                logging.warning(f"Invalid line {line_idx + 1}: {line}")
                continue
            
            # Parse line: id, source_path, target_path
            sample_id = parts[0]
            source_path = parts[1]
            target_path = parts[2]
            
            # Output path
            output_path = os.path.join(output_dir, f"{sample_id}.wav")
            
            try:
                self.convert_single(
                    source_path, target_path, output_path, max_duration, **kwargs
                )
            except Exception as e:
                logging.error(f"Failed to convert {sample_id}: {e}")
                continue
        
        logging.info(f"Batch conversion completed. Results saved in: {output_dir}")


def main():
    args = get_args()
    
    # Setup logging
    setup_logger()
    
    # Create inference engine
    engine = ZipVoiceVCInference(
        model_dir=args.model_dir,
        checkpoint_name=args.checkpoint_name,
        model_config=args.model_config,
        device=args.device,
    )
    
    # Inference parameters
    inference_kwargs = {
        'num_steps': args.num_step,
        'guidance_scale': args.guidance_scale,
        't_shift': args.t_shift,
    }
    
    if args.test_list:
        # Batch conversion
        engine.convert_batch(
            test_list=args.test_list,
            output_dir=args.res_dir,
            max_duration=args.max_duration,
            **inference_kwargs
        )
    elif args.source_audio and args.target_audio:
        # Single conversion
        output_path = os.path.join(args.res_dir, "converted.wav")
        engine.convert_single(
            source_path=args.source_audio,
            target_path=args.target_audio,
            output_path=output_path,
            max_duration=args.max_duration,
            **inference_kwargs
        )
    else:
        logging.error("Please provide either --test-list or both --source-audio and --target-audio")
        return
    
    logging.info("Voice conversion completed!")


if __name__ == "__main__":
    main()
