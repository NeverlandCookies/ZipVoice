#!/usr/bin/env python3
"""
Analyze Voice Conversion evaluation results by gender combinations.

This script reads:
1. test_vc.tsv (with gender information)
2. Detailed evaluation results (similarity, UTMOS, WER)

And outputs:
- Overall statistics
- Per-gender-combination statistics (M2M, F2F, M2F, F2M)
"""

import argparse
import logging
import os
import sys
from typing import Dict, List

import pandas as pd


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze VC evaluation results by gender."
    )
    parser.add_argument(
        "--test-vc-file",
        type=str,
        required=True,
        help="Path to test_vc.tsv file (with gender information)",
    )
    parser.add_argument(
        "--similarity-file",
        type=str,
        default=None,
        help="Path to detailed similarity results TSV",
    )
    parser.add_argument(
        "--utmos-file",
        type=str,
        default=None,
        help="Path to detailed UTMOS results TSV",
    )
    parser.add_argument(
        "--wer-file",
        type=str,
        default=None,
        help="Path to detailed WER results TSV (converted audio)",
    )
    parser.add_argument(
        "--wer-source-file",
        type=str,
        default=None,
        help="Path to detailed WER results TSV (source audio baseline)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/analysis",
        help="Output directory for analysis results",
    )
    return parser


def load_test_vc(test_vc_file: str) -> pd.DataFrame:
    """
    Load test_vc.tsv file.

    Expected format:
    wav_name    source_wav    target_wav    source_gender    target_gender
    """
    df = pd.read_csv(test_vc_file, sep="\t", header=None)
    df.columns = ["wav_name", "source_wav", "target_wav", "source_gender", "target_gender"]
    df["gender_combo"] = df["source_gender"] + "2" + df["target_gender"]
    logging.info(f"Loaded {len(df)} test pairs from {test_vc_file}")
    return df


def load_metric_results(metric_file: str, metric_name: str) -> pd.DataFrame:
    """
    Load detailed metric results.

    Expected format:
    wav_name    <metric_value>
    """
    if not metric_file or not os.path.exists(metric_file):
        logging.warning(f"{metric_name} file not found: {metric_file}")
        return None

    df = pd.read_csv(metric_file, sep="\t")
    logging.info(f"Loaded {len(df)} {metric_name} scores from {metric_file}")
    return df


def merge_results(
    test_vc_df: pd.DataFrame,
    similarity_df: pd.DataFrame = None,
    utmos_df: pd.DataFrame = None,
    wer_df: pd.DataFrame = None,
    wer_source_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Merge all results into a single DataFrame.
    Also checks for missing values and logs warnings.
    """
    merged = test_vc_df.copy()
    total_pairs = len(merged)

    if similarity_df is not None:
        merged = merged.merge(similarity_df, on="wav_name", how="left")
        similarity_missing = merged["similarity"].isna().sum()
        if similarity_missing > 0:
            logging.warning(
                f"⚠️  Similarity: {similarity_missing}/{total_pairs} pairs have missing values. "
                f"Statistics will be based on {total_pairs - similarity_missing} valid samples."
            )

    if utmos_df is not None:
        merged = merged.merge(utmos_df, on="wav_name", how="left")
        utmos_missing = merged["utmos"].isna().sum()
        if utmos_missing > 0:
            logging.warning(
                f"⚠️  UTMOS: {utmos_missing}/{total_pairs} pairs have missing values. "
                f"Statistics will be based on {total_pairs - utmos_missing} valid samples."
            )

    if wer_df is not None:
        wer_df_renamed = wer_df.rename(columns={"Name": "wav_name", "WER": "wer_converted"})
        merged = merged.merge(
            wer_df_renamed[["wav_name", "wer_converted"]], on="wav_name", how="left"
        )
        wer_missing = merged["wer_converted"].isna().sum()
        if wer_missing > 0:
            logging.warning(
                f"⚠️  WER (converted): {wer_missing}/{total_pairs} pairs have missing values. "
                f"Statistics will be based on {total_pairs - wer_missing} valid samples."
            )

    if wer_source_df is not None:
        wer_source_df_renamed = wer_source_df.rename(columns={"Name": "wav_name_source", "WER": "wer_source"})
        wer_source_df_renamed["wav_name"] = wer_source_df_renamed["wav_name_source"].str.replace("_source$", "", regex=True)
        merged = merged.merge(
            wer_source_df_renamed[["wav_name", "wer_source"]], on="wav_name", how="left"
        )
        wer_source_missing = merged["wer_source"].isna().sum()
        if wer_source_missing > 0:
            logging.warning(
                f"⚠️  WER (source baseline): {wer_source_missing}/{total_pairs} pairs have missing values. "
                f"Statistics will be based on {total_pairs - wer_source_missing} valid samples."
            )
        
        merged["wer_degradation"] = merged["wer_converted"] - merged["wer_source"]

    return merged


def compute_statistics(df: pd.DataFrame, metric_cols: List[str]) -> Dict:
    """
    Compute statistics for given metrics.
    Records both total count and valid sample count for each metric.

    Args:
        df: DataFrame with results
        metric_cols: List of metric column names

    Returns:
        Dictionary with statistics including count and metric-specific counts
    """
    stats = {"count": len(df)}  

    for metric in metric_cols:
        if metric in df.columns:
            values = df[metric].dropna()
            valid_count = len(values)
            
            stats[f"{metric}_count"] = valid_count
            
            if valid_count > 0:
                stats[f"{metric}_mean"] = values.mean()
                stats[f"{metric}_std"] = values.std()
                stats[f"{metric}_min"] = values.min()
                stats[f"{metric}_max"] = values.max()
            else:
                # All values are missing
                stats[f"{metric}_mean"] = None
                stats[f"{metric}_std"] = None
                stats[f"{metric}_min"] = None
                stats[f"{metric}_max"] = None

    return stats


def analyze_by_gender(merged_df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    """
    Analyze results by gender combination.

    Returns:
        DataFrame with per-gender-combo statistics
    """
    gender_stats = []

    overall_stats = compute_statistics(merged_df, metric_cols)
    overall_stats["gender_combo"] = "Overall"
    gender_stats.append(overall_stats)

    for combo in ["M2M", "F2F", "M2F", "F2M"]:
        combo_df = merged_df[merged_df["gender_combo"] == combo]
        if len(combo_df) > 0:
            combo_stats = compute_statistics(combo_df, metric_cols)
            combo_stats["gender_combo"] = combo
            gender_stats.append(combo_stats)

    return pd.DataFrame(gender_stats)


def save_analysis(stats_df: pd.DataFrame, output_dir: str):
    """Save analysis results to CSV and print to console."""
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "gender_analysis.csv")
    stats_df.to_csv(output_path, index=False)
    logging.info(f"Analysis results saved to {output_path}")

    print("\n" + "=" * 80)
    print("Voice Conversion Evaluation Results by Gender Combination")
    print("=" * 80)

    for _, row in stats_df.iterrows():
        gender_combo = row["gender_combo"]
        count = int(row["count"])

        print(f"\n{gender_combo} (Total: {count} test pairs):")
        print("-" * 40)

        sample_counts = {}
        if "similarity_count" in row:
            sample_counts["similarity"] = int(row["similarity_count"])
        if "utmos_count" in row:
            sample_counts["utmos"] = int(row["utmos_count"])
        if "wer_converted_count" in row:
            sample_counts["wer_converted"] = int(row["wer_converted_count"])
        if "wer_source_count" in row:
            sample_counts["wer_source"] = int(row["wer_source_count"])
        if "wer_degradation_count" in row:
            sample_counts["wer_degradation"] = int(row["wer_degradation_count"])

        if "similarity_mean" in row and pd.notna(row["similarity_mean"]):
            sim_count = sample_counts.get("similarity", count)
            count_note = f" (n={sim_count})" if sim_count != count else ""
            print(
                f"  Speaker Similarity: {row['similarity_mean']:.4f} "
                f"(±{row['similarity_std']:.4f}){count_note}"
            )
            if sim_count != count:
                print(f"    ⚠️  Note: Only {sim_count}/{count} pairs have valid similarity scores")

        if "utmos_mean" in row and pd.notna(row["utmos_mean"]):
            utmos_count = sample_counts.get("utmos", count)
            count_note = f" (n={utmos_count})" if utmos_count != count else ""
            print(
                f"  UTMOS:             {row['utmos_mean']:.4f} "
                f"(±{row['utmos_std']:.4f}){count_note}"
            )
            if utmos_count != count:
                print(f"    ⚠️  Note: Only {utmos_count}/{count} pairs have valid UTMOS scores")

        if "wer_converted_mean" in row and pd.notna(row["wer_converted_mean"]):
            wer_conv_count = sample_counts.get("wer_converted", count)
            count_note = f" (n={wer_conv_count})" if wer_conv_count != count else ""
            print(
                f"  WER (Converted):   {row['wer_converted_mean'] * 100:.2f}% "
                f"(±{row['wer_converted_std']:.2f}%){count_note}"
            )
            if wer_conv_count != count:
                print(f"    ⚠️  Note: Only {wer_conv_count}/{count} pairs have valid converted WER scores")
        
        if "wer_source_mean" in row and pd.notna(row["wer_source_mean"]):
            wer_src_count = sample_counts.get("wer_source", count)
            count_note = f" (n={wer_src_count})" if wer_src_count != count else ""
            print(
                f"  WER (Source):      {row['wer_source_mean'] * 100:.2f}% "
                f"(±{row['wer_source_std']:.2f}%){count_note}"
            )
            if wer_src_count != count:
                print(f"    ⚠️  Note: Only {wer_src_count}/{count} pairs have valid source WER scores")
        
        if "wer_degradation_mean" in row and pd.notna(row["wer_degradation_mean"]):
            wer_deg_count = sample_counts.get("wer_degradation", count)
            count_note = f" (n={wer_deg_count})" if wer_deg_count != count else ""
            degradation_sign = "+" if row["wer_degradation_mean"] >= 0 else ""
            print(
                f"  WER Degradation:   {degradation_sign}{row['wer_degradation_mean'] * 100:.2f}% "
                f"(±{row['wer_degradation_std']:.2f}%){count_note}"
            )
            if wer_deg_count != count:
                print(f"    ⚠️  Note: Only {wer_deg_count}/{count} pairs have valid degradation values")

    print("\n" + "=" * 80)


def main():
    parser = get_parser()
    args = parser.parse_args()

    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    if not os.path.isfile(args.test_vc_file):
        logging.error(f"Test VC file not found: {args.test_vc_file}")
        sys.exit(1)

    logging.info("Loading test pairs...")
    test_vc_df = load_test_vc(args.test_vc_file)

    logging.info("Loading evaluation results...")
    similarity_df = load_metric_results(args.similarity_file, "Similarity")
    utmos_df = load_metric_results(args.utmos_file, "UTMOS")
    wer_df = load_metric_results(args.wer_file, "WER (Converted)")
    wer_source_df = load_metric_results(args.wer_source_file, "WER (Source)")

    if similarity_df is None and utmos_df is None and wer_df is None:
        logging.error("No evaluation results found! Please provide at least one metric file.")
        sys.exit(1)

    logging.info("Merging results...")
    merged_df = merge_results(test_vc_df, similarity_df, utmos_df, wer_df, wer_source_df)

    total_pairs = len(merged_df)
    logging.info("=" * 60)
    logging.info("Sample Count Summary:")
    logging.info(f"  Total test pairs: {total_pairs}")
    
    if "similarity" in merged_df.columns:
        sim_valid = merged_df["similarity"].notna().sum()
        logging.info(f"  Valid similarity scores: {sim_valid}/{total_pairs}")
        if sim_valid < total_pairs:
            logging.warning(
                f"  ⚠️  Missing similarity scores: {total_pairs - sim_valid} pairs"
            )
    
    if "utmos" in merged_df.columns:
        utmos_valid = merged_df["utmos"].notna().sum()
        logging.info(f"  Valid UTMOS scores: {utmos_valid}/{total_pairs}")
        if utmos_valid < total_pairs:
            logging.warning(
                f"  ⚠️  Missing UTMOS scores: {total_pairs - utmos_valid} pairs"
            )
    
    if "wer_converted" in merged_df.columns:
        wer_conv_valid = merged_df["wer_converted"].notna().sum()
        logging.info(f"  Valid WER (converted) scores: {wer_conv_valid}/{total_pairs}")
        if wer_conv_valid < total_pairs:
            logging.warning(
                f"  ⚠️  Missing WER (converted) scores: {total_pairs - wer_conv_valid} pairs"
            )
    
    if "wer_source" in merged_df.columns:
        wer_src_valid = merged_df["wer_source"].notna().sum()
        logging.info(f"  Valid WER (source) scores: {wer_src_valid}/{total_pairs}")
        if wer_src_valid < total_pairs:
            logging.warning(
                f"  ⚠️  Missing WER (source) scores: {total_pairs - wer_src_valid} pairs"
            )
    
    if "wer_degradation" in merged_df.columns:
        wer_deg_valid = merged_df["wer_degradation"].notna().sum()
        logging.info(f"  Valid WER degradation values: {wer_deg_valid}/{total_pairs}")
        if wer_deg_valid < total_pairs:
            logging.warning(
                f"  ⚠️  Missing WER degradation values: {total_pairs - wer_deg_valid} pairs"
            )
    
    logging.info("=" * 60)

    metric_cols = []
    if "similarity" in merged_df.columns:
        metric_cols.append("similarity")
    if "utmos" in merged_df.columns:
        metric_cols.append("utmos")
    if "wer_converted" in merged_df.columns:
        metric_cols.append("wer_converted")
    if "wer_source" in merged_df.columns:
        metric_cols.append("wer_source")
    if "wer_degradation" in merged_df.columns:
        metric_cols.append("wer_degradation")

    logging.info(f"Available metrics: {', '.join(metric_cols)}")

    logging.info("Computing statistics by gender...")
    stats_df = analyze_by_gender(merged_df, metric_cols)

    save_analysis(stats_df, args.output_dir)

    logging.info("Done!")


if __name__ == "__main__":
    main()

