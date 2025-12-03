#!/usr/bin/env python3
"""
Prepare metadata for LibriTTS test-clean dataset for Voice Conversion evaluation.

This script scans the LibriTTS test-clean directory and generates a metadata file
containing information about each utterance, including:
- File path
- Speaker ID
- Gender
- Duration (in seconds)
- Transcript

Output format (TSV):
wav_path    speaker_id    gender    duration    transcript
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import soundfile as sf
from tqdm import tqdm


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare metadata for LibriTTS test-clean dataset."
    )
    parser.add_argument(
        "--test-clean-dir",
        type=str,
        required=True,
        help="Path to LibriTTS test-clean directory",
    )
    parser.add_argument(
        "--speakers-file",
        type=str,
        required=True,
        help="Path to LibriTTS speakers.tsv file containing speaker gender info",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/test_clean_metadata.tsv",
        help="Output path for metadata TSV file",
    )
    return parser


def load_speaker_info(speakers_file: str) -> Dict[str, str]:
    """
    Load speaker gender information from speakers.tsv.

    Args:
        speakers_file: Path to speakers.tsv

    Returns:
        Dictionary mapping speaker_id to gender (M/F)
    """
    speaker_info = {}
    with open(speakers_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                speaker_id = parts[0]
                gender = parts[1]
                speaker_info[speaker_id] = gender

    logging.info(f"Loaded gender info for {len(speaker_info)} speakers")
    return speaker_info


def load_transcripts(trans_file: str) -> Dict[str, str]:
    """
    Load transcripts from a .trans.tsv file.

    Args:
        trans_file: Path to {speaker}_{chapter}.trans.tsv

    Returns:
        Dictionary mapping utterance_id to normalized transcript
    """
    transcripts = {}
    if not os.path.exists(trans_file):
        return transcripts

    with open(trans_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                utt_id = parts[0]
                transcript = parts[1]
                transcripts[utt_id] = transcript

    return transcripts


def scan_test_clean(
    test_clean_dir: str, speaker_info: Dict[str, str]
) -> List[Tuple[str, str, str, float, str]]:
    """
    Scan LibriTTS test-clean directory and collect metadata.

    Args:
        test_clean_dir: Path to test-clean directory
        speaker_info: Speaker gender information

    Returns:
        List of tuples: (wav_path, speaker_id, gender, duration, transcript)
    """
    metadata = []
    test_clean_path = Path(test_clean_dir)

    speaker_dirs = sorted([d for d in test_clean_path.iterdir() if d.is_dir()])

    for speaker_dir in tqdm(speaker_dirs, desc="Scanning speakers"):
        speaker_id = speaker_dir.name

        gender = speaker_info.get(speaker_id, "Unknown")
        if gender == "Unknown":
            logging.warning(f"Unknown gender for speaker {speaker_id}, skipping")
            continue

        chapter_dirs = sorted([d for d in speaker_dir.iterdir() if d.is_dir()])

        for chapter_dir in chapter_dirs:
            chapter_id = chapter_dir.name

            trans_file = chapter_dir / f"{speaker_id}_{chapter_id}.trans.tsv"
            transcripts = load_transcripts(str(trans_file))

            wav_files = sorted(chapter_dir.glob("*.wav"))

            for wav_file in wav_files:
                utt_id = wav_file.stem  

                transcript = transcripts.get(utt_id, "")
                if not transcript:
                    logging.warning(f"No transcript found for {utt_id}, skipping")
                    continue

                try:
                    info = sf.info(str(wav_file))
                    duration = info.duration
                except Exception as e:
                    logging.warning(f"Failed to read {wav_file}: {e}, skipping")
                    continue

                metadata.append(
                    (str(wav_file), speaker_id, gender, duration, transcript)
                )

    return metadata


def clean_transcript(transcript: str) -> str:
    """
    Clean transcript text to make it TSV-safe.
    
    Replaces problematic characters that can cause TSV parsing issues:
    - Quotes (") -> single quotes (')
    - Tabs (\t) -> spaces
    - Newlines (\n) -> spaces
    - Carriage returns (\r) -> spaces
    - Multiple consecutive spaces -> single space
    
    Args:
        transcript: Original transcript text
        
    Returns:
        Cleaned transcript text safe for TSV format
    """
    cleaned = transcript.replace('"', "'")  
    cleaned = cleaned.replace('\t', ' ')    
    cleaned = cleaned.replace('\n', ' ')     
    cleaned = cleaned.replace('\r', ' ')    
    
    cleaned = re.sub(r' +', ' ', cleaned)
    return cleaned.strip()


def save_metadata(metadata: List[Tuple], output_path: str):
    """
    Save metadata to TSV file.

    Args:
        metadata: List of metadata tuples
        output_path: Output file path
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("wav_path\tspeaker_id\tgender\tduration\ttranscript\n")

        for wav_path, speaker_id, gender, duration, transcript in metadata:
            transcript_clean = clean_transcript(transcript)
            f.write(
                f"{wav_path}\t{speaker_id}\t{gender}\t{duration:.2f}\t{transcript_clean}\n"
            )

    logging.info(f"Metadata saved to {output_path}")


def print_statistics(metadata: List[Tuple]):
    """Print statistics about the metadata."""
    total = len(metadata)
    durations = [d for _, _, _, d, _ in metadata]
    genders = [g for _, _, g, _, _ in metadata]
    speakers = set([s for _, s, _, _, _ in metadata])

    male_count = sum(1 for g in genders if g == "M")
    female_count = sum(1 for g in genders if g == "F")

    logging.info("=" * 60)
    logging.info("Dataset Statistics:")
    logging.info(f"  Total utterances: {total}")
    logging.info(f"  Total speakers: {len(speakers)}")
    logging.info(f"  Male utterances: {male_count}")
    logging.info(f"  Female utterances: {female_count}")
    logging.info(f"  Duration range: {min(durations):.2f}s - {max(durations):.2f}s")
    logging.info(f"  Average duration: {sum(durations)/len(durations):.2f}s")
    logging.info("=" * 60)


def main():
    parser = get_parser()
    args = parser.parse_args()

    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    if not os.path.isdir(args.test_clean_dir):
        logging.error(f"Test-clean directory not found: {args.test_clean_dir}")
        sys.exit(1)

    if not os.path.isfile(args.speakers_file):
        logging.error(f"Speakers file not found: {args.speakers_file}")
        sys.exit(1)

    logging.info("Loading speaker information...")
    speaker_info = load_speaker_info(args.speakers_file)

    logging.info("Scanning test-clean directory...")
    metadata = scan_test_clean(args.test_clean_dir, speaker_info)

    if not metadata:
        logging.error("No valid utterances found!")
        sys.exit(1)

    logging.info(f"Found {len(metadata)} valid utterances")

    print_statistics(metadata)

    logging.info("Saving metadata...")
    save_metadata(metadata, args.output_path)

    logging.info("Done!")


if __name__ == "__main__":
    main()

