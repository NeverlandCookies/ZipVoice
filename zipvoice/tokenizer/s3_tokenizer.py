import logging
from typing import List, Union
from pathlib import Path

import torch

from lhotse import CutSet

class S3SpeechTokenizer:
    def __init__(
        self,
        model_name: str = "speech_tokenizer_v1_25hz",
        device: str = "cuda",
        freeze: bool = True,
    ):
        """
        Args:
            model_name: S3Tokenizer 
                - "speech_tokenizer_v1": 50Hz, codebook_size=4096
                - "speech_tokenizer_v1_25hz": 25Hz, codebook_size=4096
                - "speech_tokenizer_v2_25hz": 25Hz, codebook_size=6561
            device
            freeze
        """
        import s3tokenizer
        
        logging.info(f"Loading S3Tokenizer: {model_name}")
        self.s3tokenizer_model = s3tokenizer.load_model(model_name)
        
        if device == "cuda" and torch.cuda.is_available():
            self.s3tokenizer_model = self.s3tokenizer_model.cuda()
        elif device == "cpu":
            self.s3tokenizer_model = self.s3tokenizer_model.cpu()
        self.device = device
        
        if freeze:
            self.s3tokenizer_model.freeze()
            logging.info("S3Tokenizer frozen (parameters not trainable)")
        
        self.vocab_size = self._get_vocab_size()
        
        self.pad_id = 0
        
        self.has_tokens = True
        self.model_name = model_name
        
        logging.info(f"S3SpeechTokenizer initialized:")
        logging.info(f"  Model: {model_name}")
        logging.info(f"  Vocab size: {self.vocab_size}")
        logging.info(f"  Pad ID: {self.pad_id}")
        logging.info(f"  Device: {self.device}")
    
    def _get_vocab_size(self) -> int:
        if hasattr(self.s3tokenizer_model.quantizer, 'codebook_size'):
            return self.s3tokenizer_model.quantizer.codebook_size
        
        if hasattr(self.s3tokenizer_model, 'config'):
            return self.s3tokenizer_model.config.n_codebook_size
        
        if 'v2' in self.model_name:
            logging.warning("Using fallback vocab_size for V2: 6561")
            return 3**8  # 6561
        else:
            logging.warning("Using fallback vocab_size for V1: 4096")
            return 4096
    
    def speech_to_token_ids(
        self,
        audio_paths: List[str],  
    ) -> List[List[int]]:
        import s3tokenizer
        
        mels = []
        for audio_path in audio_paths:
            audio = s3tokenizer.load_audio(audio_path)  
            mel = s3tokenizer.log_mel_spectrogram(audio)  # (128, T)
            mels.append(mel)
        
        mels, mels_lens = s3tokenizer.padding(mels)
        
        mels = mels.to(self.device)
        mels_lens = mels_lens.to(self.device)
        
        with torch.no_grad():
            codes, codes_lens = self.s3tokenizer_model.quantize(mels, mels_lens)
        
        token_ids_list = []
        for i in range(codes.shape[0]):
            token_ids = codes[i, :codes_lens[i].item()].cpu().tolist()
            token_ids_list.append(token_ids)
        
        return token_ids_list




def add_tokens(cut_set: CutSet, tokenizer: str):
    if tokenizer == "s3":
        tokenizer = S3SpeechTokenizer()
    else:
        raise ValueError(f"Unsupported tokenizer: {tokenizer}.")
    
    def _prepare_cut(cut):
        assert len(cut.supervisions) == 1, (len(cut.supervisions), cut)
        
        audio_path = None
        try:
            srcs = getattr(cut.recording, "sources", [])
            if srcs:
                audio_path = getattr(srcs[0], "source", None)
        except Exception as e:
            logging.warning(f"Failed to get audio path from cut {cut.id}: {e}")
            return cut
        
        if audio_path is None:
            logging.warning(f"No audio path found for cut {cut.id}")
            return cut
        
        token_ids = tokenizer.speech_to_token_ids([str(audio_path)])
        
        cut.supervisions[0].tokens = token_ids[0]
        return cut
    
    cut_set = cut_set.map(_prepare_cut)
    return cut_set
