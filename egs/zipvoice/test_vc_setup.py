#!/usr/bin/env python3
"""
Test script to verify ZipVoice VC setup

This script tests:
1. Model architecture creation
2. Dataset loading
3. Basic forward pass
4. Configuration loading
"""

import sys
import os
import json
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from zipvoice.models.zipvoice_vc import ZipVoiceVC
from zipvoice.dataset.dataset_vc import VoiceConversionDataset, VoiceConversionCollate

def test_model_creation():
    """Test model creation and basic functionality"""
    print("=" * 50)
    print("Testing ZipVoiceVC Model Creation")
    print("=" * 50)
    
    try:
        # Load configuration
        config_path = "conf/zipvoice_vc_base.json"
        if not os.path.exists(config_path):
            print(f"❌ Config file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"✅ Configuration loaded from {config_path}")
        
        # Create model
        model_config = config["model"]
        model = ZipVoiceVC(**model_config)
        
        print(f"✅ ZipVoiceVC model created successfully")
        print(f"   - Content encoder dim: {model_config['content_encoder_dim']}")
        print(f"   - Vocab size: {model_config['vocab_size']}")
        print(f"   - Feature dim: {model_config['feat_dim']}")
        
        # Test forward pass with dummy data
        batch_size = 2
        src_len = 100
        tgt_len = 120
        feat_dim = model_config['feat_dim']
        
        # Create dummy inputs
        source_features = torch.randn(batch_size, src_len, feat_dim)
        source_features_lens = torch.tensor([src_len, src_len - 10])
        
        target_features = torch.randn(batch_size, tgt_len, feat_dim)
        target_features_lens = torch.tensor([tgt_len, tgt_len - 15])
        
        speech_condition = torch.randn(batch_size, tgt_len, feat_dim)
        speech_condition_lens = target_features_lens.clone()
        
        print("✅ Dummy data created")
        
        # Test content encoding
        content_condition, content_lens = model.forward_content_condition(
            source_features, source_features_lens
        )
        
        print(f"✅ Content encoding test passed")
        print(f"   - Content condition shape: {content_condition.shape}")
        print(f"   - Content lengths: {content_lens}")
        
        # Test content expansion
        expanded_content = model.expand_content_condition(
            content_condition, content_lens, target_features_lens
        )
        
        print(f"✅ Content expansion test passed")
        print(f"   - Expanded content shape: {expanded_content.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_creation():
    """Test dataset creation (with dummy data)"""
    print("\n" + "=" * 50)
    print("Testing VoiceConversionDataset")
    print("=" * 50)
    
    try:
        # This test uses dummy data since we may not have real data yet
        from lhotse.testing.dummies import DummyManifest
        from lhotse import CutSet
        
        # Create dummy cuts
        cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
        print("✅ Dummy cuts created")
        
        # Create dataset
        dataset = VoiceConversionDataset(
            cuts=cuts,
            mask_prob=0.8,
            mask_length=10,
            max_duration=20.0,
        )
        
        print(f"✅ VoiceConversionDataset created")
        print(f"   - Dataset size: {len(dataset)}")
        print(f"   - Mask probability: 0.8")
        print(f"   - Mask length: 10")
        
        # Test single sample
        sample = dataset[0]
        print(f"✅ Sample loading test passed")
        print(f"   - Source features shape: {sample['source_features'].shape}")
        print(f"   - Target features shape: {sample['target_features'].shape}")
        print(f"   - Speech condition shape: {sample['speech_condition'].shape}")
        
        # Test collate function
        collate_fn = VoiceConversionCollate()
        batch = collate_fn([dataset[i] for i in range(3)])
        
        print(f"✅ Batch collation test passed")
        print(f"   - Batch source shape: {batch['source_features'].shape}")
        print(f"   - Batch target shape: {batch['target_features'].shape}")
        print(f"   - Batch speech condition shape: {batch['speech_condition'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_structure():
    """Test if all required files are in place"""
    print("\n" + "=" * 50)
    print("Testing File Structure")
    print("=" * 50)
    
    required_files = [
        "conf/zipvoice_vc_base.json",
        "local/prepare_libritts_vc.sh",
        "run_zipvoice_vc.sh",
        "test_vc.tsv",
        "README_VC.md",
        "../../zipvoice/models/zipvoice_vc.py",
        "../../zipvoice/dataset/dataset_vc.py",
        "../../zipvoice/bin/train_zipvoice_vc.py",
        "../../zipvoice/bin/infer_zipvoice_vc.py",
    ]
    
    all_exist = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (missing)")
            all_exist = False
    
    return all_exist

def test_dependencies():
    """Test if required dependencies are available"""
    print("\n" + "=" * 50)
    print("Testing Dependencies")
    print("=" * 50)
    
    dependencies = [
        ("torch", "PyTorch"),
        ("lhotse", "Lhotse"),
        ("librosa", "Librosa"),
        ("soundfile", "SoundFile"),
        ("vocos", "Vocos"),
        ("torchaudio", "TorchAudio"),
    ]
    
    all_available = True
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} (not installed)")
            all_available = False
    
    return all_available

def main():
    """Run all tests"""
    print("ZipVoice Voice Conversion Setup Test")
    print("=" * 60)
    
    # Change to egs/zipvoice directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    tests = [
        ("File Structure", test_file_structure),
        ("Dependencies", test_dependencies),
        ("Model Creation", test_model_creation),
        ("Dataset Creation", test_dataset_creation),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your ZipVoice VC setup is ready.")
        print("\nNext steps:")
        print("1. Download pre-trained Zipformer model")
        print("2. Run: bash run_zipvoice_vc.sh --stage 1 --stop-stage 2")
        print("3. Start training: bash run_zipvoice_vc.sh --stage 3")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
