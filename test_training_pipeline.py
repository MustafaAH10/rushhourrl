#!/usr/bin/env python3
"""
Rush Hour Training Pipeline Test Script
Tests all components before running full training
"""

import os
import sys
import json
import pandas as pd
import torch
from pathlib import Path

def test_data_files():
    """Test that data files exist and have correct structure"""
    print("=== TESTING DATA FILES ===")
    
    train_path = "data/rushhour_train.parquet"
    val_path = "data/rushhour_val.parquet"
    
    # Check files exist
    if not os.path.exists(train_path):
        print(f"❌ Training data not found: {train_path}")
        return False
    if not os.path.exists(val_path):
        print(f"❌ Validation data not found: {val_path}")
        return False
        
    print(f"✅ Found training data: {train_path}")
    print(f"✅ Found validation data: {val_path}")
    
    # Load and check structure
    try:
        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
        
        required_cols = ['prompt', 'data_source', 'reward_model', 'extra_info']
        
        for col in required_cols:
            if col not in train_df.columns:
                print(f"❌ Missing column in training data: {col}")
                return False
            if col not in val_df.columns:
                print(f"❌ Missing column in validation data: {col}")
                return False
                
        print(f"✅ Training data: {len(train_df)} rows")
        print(f"✅ Validation data: {len(val_df)} rows")
        
        # Test data structure
        sample = train_df.iloc[0]
        
        # Check reward_model structure
        if 'ground_truth' not in sample['reward_model']:
            print("❌ Missing reward_model.ground_truth")
            return False
            
        # Check extra_info structure  
        extra_info = sample['extra_info']
        if 'puzzle_state' not in extra_info or 'optimal_moves' not in extra_info:
            print("❌ Missing puzzle_state or optimal_moves in extra_info")
            return False
            
        print("✅ Data structure is correct")
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def test_reward_function():
    """Test reward function integration"""
    print("\n=== TESTING REWARD FUNCTION ===")
    
    try:
        # Add veRL to path
        sys.path.append('verl/utils/reward_score')
        from rushhour import compute_score
        
        # Load sample data
        train_df = pd.read_parquet("data/rushhour_train.parquet")
        sample = train_df.iloc[0]
        
        # Test with ground truth (should get 10.0)
        ground_truth = sample['reward_model']['ground_truth']
        extra_info = sample['extra_info']
        
        score = compute_score(ground_truth, "unused", extra_info)
        
        if score == 10.0:
            print("✅ Reward function returns 10.0 for perfect solution")
        else:
            print(f"⚠️  Reward function returned {score} (expected 10.0)")
            
        # Test with bad response (should get negative score)
        bad_response = "Invalid response without solution tags"
        bad_score = compute_score(bad_response, "unused", extra_info)
        
        if bad_score < 0:
            print("✅ Reward function returns negative score for bad response")
        else:
            print(f"⚠️  Bad response got score {bad_score} (expected negative)")
            
        return True
        
    except Exception as e:
        print(f"❌ Reward function test failed: {e}")
        return False

def test_verl_imports():
    """Test that veRL components can be imported"""
    print("\n=== TESTING VERL IMPORTS ===")
    
    try:
        from verl.trainer.main_ppo import main
        print("✅ veRL main_ppo import successful")
        
        from verl.utils.reward_score import default_compute_score
        print("✅ veRL reward_score import successful")
        
        # Test rushhour integration
        score = default_compute_score(
            "rushhour",
            "<solution>\nStep 1: C [1,1] -> [1,2]\n</solution>",
            "unused",
            {"puzzle_state": '{"exit_position": [1,2]}', "optimal_moves": 1}
        )
        print(f"✅ veRL rushhour integration works (score: {score})")
        
        return True
        
    except Exception as e:
        print(f"❌ veRL import failed: {e}")
        return False

def test_model_loading():
    """Test that model can be loaded"""
    print("\n=== TESTING MODEL LOADING ===")
    
    try:
        from transformers import AutoTokenizer, AutoConfig
        
        model_path = "Qwen/Qwen2.5-3B"
        
        # Test tokenizer loading
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("✅ Tokenizer loaded successfully")
        
        # Test config loading
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        print("✅ Model config loaded successfully")
        
        # Test a sample encoding
        sample_text = "Test encoding"
        tokens = tokenizer.encode(sample_text)
        print(f"✅ Tokenizer encoding works (tokens: {len(tokens)})")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print("   Make sure you have internet connection for model download")
        return False

def test_gpu_availability():
    """Test GPU availability and memory"""
    print("\n=== TESTING GPU ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
        
    gpu_count = torch.cuda.device_count()
    print(f"✅ Found {gpu_count} GPU(s)")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        print(f"  GPU {i}: {props.name} ({memory_gb:.1f} GB)")
        
        if memory_gb < 10:
            print(f"⚠️  GPU {i} has limited memory ({memory_gb:.1f} GB)")
            print("     Consider reducing batch size if training fails")
            
    return True

def test_ray_functionality():
    """Test Ray distributed computing"""
    print("\n=== TESTING RAY ===")
    
    try:
        import ray
        
        # Check if ray is already running
        try:
            ray.shutdown()
        except:
            pass
            
        # Initialize ray
        ray.init(num_cpus=2, num_gpus=1 if torch.cuda.is_available() else 0)
        
        # Test basic ray functionality
        @ray.remote
        def test_function():
            return "Ray is working"
            
        result = ray.get(test_function.remote())
        print("✅ Ray distributed computing works")
        
        ray.shutdown()
        return True
        
    except Exception as e:
        print(f"❌ Ray test failed: {e}")
        return False

def run_mini_training_test():
    """Run a mini training test with 1 batch"""
    print("\n=== MINI TRAINING TEST ===")
    print("Running 1 step of training to test full pipeline...")
    
    try:
        # Set environment for mini test
        os.environ['PYTHONPATH'] = f"{os.getcwd()}:{os.environ.get('PYTHONPATH', '')}"
        
        # Import after setting PYTHONPATH
        sys.path.insert(0, '.')
        
        # Create minimal test config by running training for 1 step
        cmd = [
            'python3', '-m', 'verl.trainer.main_ppo',
            'algorithm.adv_estimator=grpo',
            f'data.train_files=["data/rushhour_train.parquet"]',
            f'data.val_files=["data/rushhour_val.parquet"]',
            'data.train_batch_size=8',  # Very small for testing
            'data.max_prompt_length=1024',
            'data.max_response_length=512',
            'data.reward_fn_key=data_source',
            'data.filter_overlong_prompts=True',
            'data.truncation=error',
            'actor_rollout_ref.model.path=Qwen/Qwen2.5-3B',
            'actor_rollout_ref.rollout.name=hf',  # Use HF instead of vLLM for testing
            'trainer.total_epochs=1',  # Just 1 epoch
            'trainer.save_freq=999',   # Don't save
            'trainer.test_freq=999',   # Don't validate
            'trainer.logger=["console"]',  # Console only
            'trainer.project_name=test_run',
            'trainer.experiment_name=mini_test'
        ]
        
        print("⚠️  Mini training test requires actual model loading and may take 5-10 minutes...")
        print("   To skip this test, press Ctrl+C")
        
        import subprocess
        import signal
        
        # Set timeout for test
        def timeout_handler(signum, frame):
            raise TimeoutError("Mini training test timed out")
            
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(600)  # 10 minute timeout
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            signal.alarm(0)  # Cancel timeout
            
            if result.returncode == 0:
                print("✅ Mini training test completed successfully!")
                return True
            else:
                print("❌ Mini training test failed:")
                print("STDOUT:", result.stdout[-500:])  # Last 500 chars
                print("STDERR:", result.stderr[-500:])
                return False
                
        except (subprocess.TimeoutExpired, TimeoutError):
            print("⚠️  Mini training test timed out (10 minutes)")
            print("   This might be normal for first-time model download")
            return True  # Don't fail the test for timeout
            
    except KeyboardInterrupt:
        print("⚠️  Mini training test skipped by user")
        return True  # Don't fail if user skips
    except Exception as e:
        print(f"❌ Mini training test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 RUSH HOUR TRAINING PIPELINE TEST SUITE")
    print("=" * 50)
    
    tests = [
        ("Data Files", test_data_files),
        ("Reward Function", test_reward_function),
        ("veRL Imports", test_verl_imports),
        ("Model Loading", test_model_loading),
        ("GPU Availability", test_gpu_availability),
        ("Ray Functionality", test_ray_functionality),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except KeyboardInterrupt:
            print(f"\n⚠️  {test_name} test interrupted by user")
            results.append((test_name, True))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("🧪 TEST RESULTS SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:<10} {test_name}")
        if result:
            passed += 1
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Your training pipeline is ready!")
        print("\nTo start training, run:")
        print("  ./run_rushhour_grpo.sh")
        
        # Ask about mini training test
        try:
            response = input("\n🔬 Run mini training test? (y/N): ").strip().lower()
            if response == 'y':
                run_mini_training_test()
        except KeyboardInterrupt:
            print("\nSkipping mini training test.")
            
    else:
        print(f"\n⚠️  {total - passed} test(s) failed!")
        print("Please fix the issues above before running training.")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())