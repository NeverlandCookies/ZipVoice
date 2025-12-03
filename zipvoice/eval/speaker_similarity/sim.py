#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors:  Han Zhu
#                                                   Wei Kang)
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
Computes speaker similarity (SIM-o) using a WavLM-based
    ECAPA-TDNN speaker verification model.
    
Modified for Voice Conversion: Supports VC test format and saves detailed scores.
"""
import argparse
import logging
import os
import warnings
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from zipvoice.eval.models.ecapa_tdnn_wavlm import ECAPA_TDNN_WAVLM
from zipvoice.eval.utils import load_waveform

warnings.filterwarnings("ignore")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate speaker similarity (SIM-o) score for Voice Conversion."
    )

    parser.add_argument(
        "--wav-path",
        type=str,
        required=True,
        help="Path to the directory containing evaluated speech files (converted audio).",
    )
    parser.add_argument(
        "--test-list",
        type=str,
        required=True,
        help="Path to the test list file. For VC, each line contains "
        "(wav_name, source_wav, target_wav) or "
        "(wav_name, source_wav, target_wav, source_gender, target_gender) "
        "separated by tabs.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Local path of our evaluation model repository. "
        "Download from https://huggingface.co/k2-fsa/TTS_eval_models. "
        "Will use 'tts_eval_models/speaker_similarity/wavlm_large_finetune.pth' "
        "and 'tts_eval_models/speaker_similarity/wavlm_large/' in this script",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save detailed similarity scores for each test pair. "
        "If not specified, only the average score will be printed.",
    )
    parser.add_argument(
        "--extension",
        type=str,
        default="wav",
        help="Extension of the speech files. Default: wav",
    )
    return parser


class SpeakerSimilarity:
    """
    Computes speaker similarity (SIM-o) using a WavLM-based
        ECAPA-TDNN speaker verification model.
    """

    def __init__(
        self,
        sv_model_path: str = "speaker_similarity/wavlm_large_finetune.pth",
        ssl_model_path: str = "speaker_similarity/wavlm_large/",
    ):
        """
        Initializes the speaker similarity evaluator with the specified models.

        Args:
            sv_model_path (str): Path of the wavlm-based ECAPA-TDNN model checkpoint.
            ssl_model_path (str): Path of the wavlm SSL model directory.
        """
        self.sample_rate = 16000
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        logging.info(f"Using device: {self.device}")
        self.model = ECAPA_TDNN_WAVLM(
            feat_dim=1024,
            channels=512,
            emb_dim=256,
            sr=self.sample_rate,
            ssl_model_path=ssl_model_path,
        )
        state_dict = torch.load(
            sv_model_path, map_location=lambda storage, loc: storage
        )
        self.model.load_state_dict(state_dict["model"], strict=False)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def get_embeddings(self, wav_paths: List[str]) -> List[torch.Tensor]:
        """
        Extracts speaker embeddings from a list of audio files.

        Args:
            wav_paths (List[str]): List of paths to audio files.

        Returns:
            List[torch.Tensor]: List of speaker embeddings.
        """
        embeddings = []
        for wav_path in tqdm(wav_paths, desc="Extracting speaker embeddings"):
            # Load and preprocess waveform
            speech = load_waveform(
                wav_path, self.sample_rate, device=self.device, max_seconds=120
            )
            # Extract embedding
            embedding = self.model([speech])
            embeddings.append(embedding)

        return embeddings

    def score(
        self, 
        wav_path: str, 
        extension: str, 
        test_list: str, 
        output_path: str = None
    ) -> Tuple[float, List[Tuple[str, float]]]:
        """
        Computes the Speaker Similarity (SIM-o) score between target and
            converted speech for Voice Conversion.

        Args:
            wav_path (str): Path to the directory containing converted speech files.
            extension (str): File extension of the audio files.
            test_list (str): Path to the test list file. For VC, format is:
                {wav_name}\t{source_wav}\t{target_wav}[\t{source_gender}\t{target_gender}]
            output_path (str, optional): Path to save detailed scores.

        Returns:
            Tuple[float, List[Tuple[str, float]]]: 
                - Average similarity score
                - List of (wav_name, similarity_score) tuples
        """
        logging.info(f"Calculating Speaker Similarity (SIM-o) score for {wav_path}")
        
        # Read test pairs
        try:
            with open(test_list, "r", encoding="utf-8") as f:
                lines = [line.strip().split("\t") for line in f if line.strip()]
        except Exception as e:
            logging.error(f"Failed to read test list: {e}")
            raise

        if not lines:
            raise ValueError(f"Test list {test_list} is empty or malformed")
        
        # Parse test pairs (support both 3-field and 5-field formats)
        target_wavs = []  # Reference speaker audio (target_wav in VC)
        eval_wavs = []    # Converted audio
        wav_names = []
        
        for line in lines:
            if len(line) == 3:
                # Format: wav_name, source_wav, target_wav
                wav_name, source_wav, target_wav = line
            elif len(line) == 5:
                # Format: wav_name, source_wav, target_wav, source_gender, target_gender
                wav_name, source_wav, target_wav, source_gender, target_gender = line
            else:
                raise ValueError(
                    f"Invalid line format (expected 3 or 5 fields): {line}"
                )
            
            # For VC similarity: compare converted audio with target speaker
            eval_wav_path = os.path.join(wav_path, f"{wav_name}.{extension}")
            
            # Validate file existence
            if not os.path.exists(target_wav):
                raise FileNotFoundError(f"Target file not found: {target_wav}")
            if not os.path.exists(eval_wav_path):
                raise FileNotFoundError(f"Converted file not found: {eval_wav_path}")
            
            target_wavs.append(target_wav)
            eval_wavs.append(eval_wav_path)
            wav_names.append(wav_name)
        
        logging.info(f"Found {len(target_wavs)} valid test pairs")
        
        # Extract embeddings
        target_embeddings = self.get_embeddings(target_wavs)
        eval_embeddings = self.get_embeddings(eval_wavs)

        if len(target_embeddings) != len(eval_embeddings):
            raise RuntimeError(
                f"Mismatch: {len(target_embeddings)} target vs "
                f"{len(eval_embeddings)} eval embeddings"
            )

        # Calculate similarity scores
        scores = []
        detailed_results = []
        
        for wav_name, target_emb, eval_emb in zip(
            wav_names, target_embeddings, eval_embeddings
        ):
            # Compute cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                target_emb, eval_emb, dim=-1
            )
            score = similarity.item()
            scores.append(score)
            detailed_results.append((wav_name, score))

        # Save detailed scores if output path is provided
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("wav_name\tsimilarity\n")
                for wav_name, score in detailed_results:
                    f.write(f"{wav_name}\t{score:.6f}\n")
            
            logging.info(f"Detailed similarity scores saved to: {output_path}")

        return float(np.mean(scores)), detailed_results


if __name__ == "__main__":

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    parser = get_parser()
    args = parser.parse_args()
    # Initialize evaluator
    sv_model_path = os.path.join(
        args.model_dir, "speaker_similarity/wavlm_large_finetune.pth"
    )
    ssl_model_path = os.path.join(args.model_dir, "speaker_similarity/wavlm_large/")
    if not os.path.exists(sv_model_path) or not os.path.exists(ssl_model_path):
        logging.error(
            "Please download evaluation models from "
            "https://huggingface.co/k2-fsa/TTS_eval_models"
            " and pass this dir with --model-dir"
        )
        exit(1)
    sim_evaluator = SpeakerSimilarity(
        sv_model_path=sv_model_path, ssl_model_path=ssl_model_path
    )
    # Compute similarity score
    avg_score, detailed_results = sim_evaluator.score(
        args.wav_path, 
        args.extension, 
        args.test_list,
        args.output_path
    )
    
    print("-" * 50)
    logging.info(f"SIM-o score: {avg_score:.4f}")
    print("-" * 50)

