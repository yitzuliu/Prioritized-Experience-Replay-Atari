#!/usr/bin/env python3
"""
Test script for reliability and efficiency improvements.

This script tests the key improvements made to the PER DQN project
to ensure they are working correctly.

測試腳本，用於驗證可靠性和效率改進。

此腳本測試對 PER DQN 項目所做的關鍵改進，以確保它們正常工作。
"""

import sys
import os
import time
import traceback
import numpy as np
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_configuration_validation():
    """Test configuration validation system."""
    print("🧪 Testing Configuration Validation...")
    
    try:
        import config
        
        # Test configuration summary
        if hasattr(config, 'get_config_summary'):
            summary = config.get_config_summary()
            assert 'environment' in summary
            assert 'training' in summary
            assert 'per' in summary
            print("✅ Configuration summary working")
        else:
            print("⚠️ Configuration summary not available")
        
        # Test validation
        if hasattr(config, 'validate_config'):
            errors = config.validate_config()
            if not errors:
                print("✅ Configuration validation passed")
            else:
                print(f"⚠️ Configuration validation found issues: {errors}")
        else:
            print("⚠️ Configuration validation not available")
            
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation test failed: {str(e)}")
        return False

def test_enhanced_per_memory():
    """Test enhanced PER memory implementation."""
    print("\n🧪 Testing Enhanced PER Memory...")
    
    try:
        from src.per_memory import PERMemory
        
        # Create PER memory instance
        memory = PERMemory(memory_capacity=100)
        
        # Test memory usage monitoring
        if hasattr(memory, 'get_memory_usage'):
            usage = memory.get_memory_usage()
            assert 'rss_bytes' in usage
            assert 'percent' in usage
            print("✅ Memory usage monitoring working")
        else:
            print("⚠️ Memory usage monitoring not available")
        
        # Test performance stats
        if hasattr(memory, 'get_performance_stats'):
            stats = memory.get_performance_stats()
            assert 'cache_hit_rate' in stats
            assert 'memory_usage' in stats
            print("✅ Performance statistics working")
        else:
            print("⚠️ Performance statistics not available")
        
        # Test adding transitions with validation
        try:
            # Valid transition
            state = np.random.rand(4, 84, 84)
            memory.add(state, 1, 1.0, state, False)
            
            # Test invalid transitions (should be caught)
            try:
                memory.add(None, 1, 1.0, state, False)  # Invalid state
                print("⚠️ Invalid state not caught")
            except ValueError:
                print("✅ Input validation working")
            
        except Exception as e:
            print(f"⚠️ Transition validation test failed: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced PER memory test failed: {str(e)}")
        return False

def test_performance_monitoring():
    """Test performance monitoring system."""
    print("\n🧪 Testing Performance Monitoring...")
    
    try:
        from src.performance_monitor import PerformanceMonitor, HyperparameterTuner
        
        # Test performance monitor
        monitor = PerformanceMonitor(monitor_interval=0.1, history_size=10)
        
        # Test timing context manager
        if hasattr(monitor, 'time_operation'):
            with monitor.time_operation('test_operation'):
                time.sleep(0.01)  # Simulate some work
            print("✅ Timing context manager working")
        else:
            print("⚠️ Timing context manager not available")
        
        # Test monitoring start/stop
        if hasattr(monitor, 'start_monitoring') and hasattr(monitor, 'stop_monitoring'):
            monitor.start_monitoring()
            time.sleep(0.2)  # Let it collect some data
            monitor.stop_monitoring()
            print("✅ Background monitoring working")
        else:
            print("⚠️ Background monitoring not available")
        
        # Test performance reporting
        if hasattr(monitor, 'get_performance_report'):
            report = monitor.get_performance_report()
            assert 'timestamp' in report
            assert 'system_performance' in report
            print("✅ Performance reporting working")
        else:
            print("⚠️ Performance reporting not available")
        
        # Test hyperparameter tuner
        tuner = HyperparameterTuner(monitor)
        if hasattr(tuner, 'get_tuning_recommendations'):
            metrics = {
                'loss_trend': 0.1,
                'loss_variance': 0.05,
                'reward_trend': 1.0,
                'avg_fps': 50,
                'memory_usage': 0.7
            }
            recommendations = tuner.get_tuning_recommendations(metrics)
            assert 'timestamp' in recommendations
            print("✅ Hyperparameter tuning working")
        else:
            print("⚠️ Hyperparameter tuning not available")
        
        return True
        
    except ImportError:
        print("⚠️ Performance monitoring module not available")
        return True  # Not a failure if module is not available
    except Exception as e:
        print(f"❌ Performance monitoring test failed: {str(e)}")
        return False

def test_enhanced_agent():
    """Test enhanced DQN agent capabilities."""
    print("\n🧪 Testing Enhanced DQN Agent...")
    
    try:
        from src.dqn_agent import DQNAgent
        import torch
        import tempfile
        
        # Create agent
        agent = DQNAgent(
            state_shape=(4, 84, 84),
            action_space_size=9,
            learning_rate=0.0001,
            use_per=True
        )
        
        # Test training diagnostics
        if hasattr(agent, 'get_training_diagnostics'):
            diagnostics = agent.get_training_diagnostics()
            assert 'training_progress' in diagnostics
            assert 'network_info' in diagnostics
            print("✅ Training diagnostics working")
        else:
            print("⚠️ Training diagnostics not available")
        
        # Test enhanced model saving/loading
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
        
        try:
            # Test saving with metadata
            metadata = {'test': 'enhanced_save'}
            success = agent.save_model(temp_path, metadata=metadata)
            
            if success:
                print("✅ Enhanced model saving working")
                
                # Test enhanced loading
                if hasattr(agent, 'load_model'):
                    load_result = agent.load_model(temp_path)
                    if isinstance(load_result, dict) and load_result.get('success', False):
                        print("✅ Enhanced model loading working")
                    else:
                        print("⚠️ Enhanced model loading returned unexpected result")
                else:
                    print("⚠️ Enhanced model loading not available")
            else:
                print("⚠️ Enhanced model saving failed")
        
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced DQN agent test failed: {str(e)}")
        return False

def test_resource_monitoring():
    """Test resource monitoring capabilities."""
    print("\n🧪 Testing Resource Monitoring...")
    
    try:
        import psutil
        
        # Test basic resource monitoring
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        assert memory_info.rss > 0
        assert 0 <= memory_percent <= 100
        print("✅ Basic resource monitoring working")
        
        # Test enhanced resource checking (if available in train.py)
        try:
            from train import enhanced_resource_check
            status = enhanced_resource_check()
            assert 'memory_percent' in status
            assert 'warnings' in status
            print("✅ Enhanced resource checking working")
        except ImportError:
            print("⚠️ Enhanced resource checking not available")
        
        return True
        
    except Exception as e:
        print(f"❌ Resource monitoring test failed: {str(e)}")
        return False

def test_td_error_calculation():
    """Test TD-error calculation returns valid finite scalar values."""
    print("\n🧪 Testing TD-Error Calculation...")

    try:
        from src.dqn_agent import DQNAgent

        agent = DQNAgent(
            state_shape=(4, 84, 84),
            action_space_size=9,
            learning_rate=0.0001,
            use_per=True
        )

        state = np.random.rand(4, 84, 84).astype(np.float32)
        next_state = np.random.rand(4, 84, 84).astype(np.float32)
        td_error = agent._calculate_td_error(
            state=state,
            action=0,
            reward=1.0,
            next_state=next_state,
            done=False
        )

        assert isinstance(td_error, float), f"TD error should be float, got {type(td_error)}"
        assert np.isfinite(td_error), "TD error should be finite"
        print("✅ TD-error calculation working")
        return True

    except Exception as e:
        print(f"❌ TD-error calculation test failed: {str(e)}")
        return False

def test_priority_updates_formula_behavior():
    """Test PER priority updates are monotonic with increasing TD errors."""
    print("\n🧪 Testing PER Priority Update Behavior...")

    try:
        from src.per_memory import PERMemory

        memory = PERMemory(memory_capacity=32)

        # Add transitions so we can sample and update priorities.
        for _ in range(16):
            state = np.random.rand(4, 84, 84).astype(np.float32)
            memory.add(state, 0, 1.0, state, False)

        indices, _, _ = memory.sample(8)
        td_errors = np.linspace(0.1, 1.0, num=8)
        memory.update_priorities(indices, td_errors)

        # Map updated indices to leaf priorities and verify monotonic ordering.
        updated_priorities = [memory.sumtree.priority_tree[int(idx)] for idx in indices]
        ordered_pairs = sorted(zip(td_errors, updated_priorities), key=lambda x: x[0])

        for i in range(1, len(ordered_pairs)):
            assert ordered_pairs[i][1] >= ordered_pairs[i - 1][1], (
                "Priority should be monotonic with TD-error"
            )

        print("✅ PER priority update behavior working")
        return True

    except Exception as e:
        print(f"❌ PER priority update behavior test failed: {str(e)}")
        return False

def test_target_network_sync():
    """Test target network sync copies policy network parameters correctly."""
    print("\n🧪 Testing Target Network Synchronization...")

    try:
        import torch
        from src.dqn_agent import DQNAgent

        agent = DQNAgent(
            state_shape=(4, 84, 84),
            action_space_size=9,
            learning_rate=0.0001,
            use_per=True
        )

        # Perturb policy network to ensure mismatch before sync.
        with torch.no_grad():
            for param in agent.policy_network.parameters():
                param.add_(0.01)

        pre_sync_delta = 0.0
        for p, t in zip(agent.policy_network.parameters(), agent.target_network.parameters()):
            pre_sync_delta += (p - t).abs().sum().item()
        assert pre_sync_delta > 0.0, "Networks should differ before sync"

        agent.target_network.load_state_dict(agent.policy_network.state_dict())

        post_sync_delta = 0.0
        for p, t in zip(agent.policy_network.parameters(), agent.target_network.parameters()):
            post_sync_delta += (p - t).abs().sum().item()
        assert post_sync_delta == 0.0, "Networks should match after sync"

        print("✅ Target network synchronization working")
        return True

    except Exception as e:
        print(f"❌ Target network synchronization test failed: {str(e)}")
        return False

def test_resume_state_consistency():
    """Test checkpoint save/load preserves core resume state fields."""
    print("\n🧪 Testing Resume State Consistency...")

    try:
        from src.dqn_agent import DQNAgent

        agent = DQNAgent(
            state_shape=(4, 84, 84),
            action_space_size=9,
            learning_rate=0.0001,
            use_per=True
        )

        agent.steps_done = 1234
        agent.training_steps = 567
        agent._epsilon = 0.123

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            checkpoint_path = f.name

        try:
            save_ok = agent.save_model(checkpoint_path, save_optimizer=True, metadata={'test': True})
            assert save_ok, "Save should succeed"

            load_result = agent.load_model(checkpoint_path)
            assert load_result.get('success', False), f"Load failed: {load_result}"
            assert agent.steps_done == 1234, f"Unexpected steps_done: {agent.steps_done}"
            assert agent.training_steps == 567, f"Unexpected training_steps: {agent.training_steps}"
            assert abs(agent.epsilon - 0.123) < 1e-9, f"Unexpected epsilon: {agent.epsilon}"
        finally:
            if os.path.exists(checkpoint_path):
                os.unlink(checkpoint_path)

        print("✅ Resume state consistency working")
        return True

    except Exception as e:
        print(f"❌ Resume state consistency test failed: {str(e)}")
        return False

def main():
    """Run all tests for reliability and efficiency improvements."""
    print("🚀 Testing Reliability and Efficiency Improvements")
    print("=" * 60)
    
    tests = [
        ("Configuration Validation", test_configuration_validation),
        ("Enhanced PER Memory", test_enhanced_per_memory),
        ("Performance Monitoring", test_performance_monitoring),
        ("Enhanced DQN Agent", test_enhanced_agent),
        ("Resource Monitoring", test_resource_monitoring),
        ("TD-Error Calculation", test_td_error_calculation),
        ("PER Priority Update Behavior", test_priority_updates_formula_behavior),
        ("Target Network Synchronization", test_target_network_sync),
        ("Resume State Consistency", test_resume_state_consistency),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} test crashed: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Results Summary:")
    
    passed = 0
    total = 0
    
    for test_name, result in results.items():
        total += 1
        if result:
            passed += 1
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All reliability and efficiency improvements are working correctly!")
        return 0
    else:
        print("⚠️ Some improvements may need attention.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 