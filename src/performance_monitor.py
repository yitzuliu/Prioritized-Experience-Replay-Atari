"""
Performance monitoring and profiling system for DQN training.

This module provides comprehensive performance monitoring, profiling capabilities,
and automated hyperparameter tuning to optimize training efficiency.

DQN 訓練的性能監控和分析系統。

此模組提供綜合性能監控、分析功能和自動超參數調優，以優化訓練效率。
"""

import time
import psutil
import numpy as np
import os
import json
import threading
from collections import defaultdict, deque
from datetime import datetime
import warnings

# Add parent directory to path to import config.py
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for DQN training.
    
    DQN 訓練的綜合性能監控系統。
    """
    
    def __init__(self, monitor_interval=1.0, history_size=1000):
        """
        Initialize the performance monitor.
        
        Args:
            monitor_interval: Interval in seconds between monitoring samples
            history_size: Maximum number of historical samples to keep
        """
        self.monitor_interval = monitor_interval
        self.history_size = history_size
        
        # Performance metrics storage
        self.metrics = {
            'cpu_percent': deque(maxlen=history_size),
            'memory_percent': deque(maxlen=history_size),
            'memory_rss_mb': deque(maxlen=history_size),
            'gpu_memory_mb': deque(maxlen=history_size),
            'fps': deque(maxlen=history_size),
            'learning_rate': deque(maxlen=history_size),
            'batch_time': deque(maxlen=history_size),
            'sample_time': deque(maxlen=history_size),
            'forward_time': deque(maxlen=history_size),
            'backward_time': deque(maxlen=history_size)
        }
        
        # Timing context managers
        self.timers = {}
        self.timer_stack = []
        
        # Monitoring state
        self.monitoring = False
        self.monitor_thread = None
        self.start_time = None
        
        # Frame counting for FPS calculation
        self.frame_count = 0
        self.last_fps_time = time.time()
        
        # Bottleneck detection
        self.bottleneck_thresholds = {
            'cpu_percent': 90.0,
            'memory_percent': 85.0,
            'batch_time': 1.0,  # seconds
            'sample_time': 0.5,  # seconds
        }
        
        self.bottleneck_alerts = []
    
    def start_monitoring(self):
        """
        Start continuous system monitoring in a background thread.
        
        開始在後台線程中進行連續系統監控。
        """
        if not self.monitoring:
            self.monitoring = True
            self.start_time = time.time()
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            print("Performance monitoring started")
    
    def stop_monitoring(self):
        """
        Stop system monitoring.
        
        停止系統監控。
        """
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        print("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """
        Main monitoring loop running in background thread.
        
        在後台線程中運行的主監控循環。
        """
        while self.monitoring:
            try:
                # Get current system metrics
                process = psutil.Process()
                
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                memory_percent = process.memory_percent()
                
                # Store metrics
                self.metrics['cpu_percent'].append(cpu_percent)
                self.metrics['memory_percent'].append(memory_percent)
                self.metrics['memory_rss_mb'].append(memory_info.rss / 1024 / 1024)
                
                # Try to get GPU memory if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
                        self.metrics['gpu_memory_mb'].append(gpu_memory)
                    else:
                        self.metrics['gpu_memory_mb'].append(0)
                except:
                    self.metrics['gpu_memory_mb'].append(0)
                
                # Check for bottlenecks
                self._check_bottlenecks()
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                warnings.warn(f"Error in monitoring loop: {str(e)}")
                time.sleep(self.monitor_interval)
    
    def _check_bottlenecks(self):
        """
        Check for performance bottlenecks and generate alerts.
        
        檢查性能瓶頸並生成警報。
        """
        current_time = time.time()
        
        # Check CPU usage
        if self.metrics['cpu_percent'] and self.metrics['cpu_percent'][-1] > self.bottleneck_thresholds['cpu_percent']:
            alert = {
                'type': 'high_cpu',
                'value': self.metrics['cpu_percent'][-1],
                'threshold': self.bottleneck_thresholds['cpu_percent'],
                'timestamp': current_time,
                'message': f"High CPU usage: {self.metrics['cpu_percent'][-1]:.1f}%"
            }
            self.bottleneck_alerts.append(alert)
        
        # Check memory usage
        if self.metrics['memory_percent'] and self.metrics['memory_percent'][-1] > self.bottleneck_thresholds['memory_percent']:
            alert = {
                'type': 'high_memory',
                'value': self.metrics['memory_percent'][-1],
                'threshold': self.bottleneck_thresholds['memory_percent'],
                'timestamp': current_time,
                'message': f"High memory usage: {self.metrics['memory_percent'][-1]:.1f}%"
            }
            self.bottleneck_alerts.append(alert)
        
        # Limit alert history
        if len(self.bottleneck_alerts) > 100:
            self.bottleneck_alerts = self.bottleneck_alerts[-50:]
    
    def time_operation(self, operation_name):
        """
        Context manager for timing operations.
        
        用於計時操作的上下文管理器。
        
        Args:
            operation_name: Name of the operation being timed
            
        Usage:
            with monitor.time_operation('batch_processing'):
                # ... code to time ...
        """
        return TimingContext(self, operation_name)
    
    def record_frame(self):
        """
        Record a frame for FPS calculation.
        
        記錄一幀用於 FPS 計算。
        """
        self.frame_count += 1
        current_time = time.time()
        
        # Calculate FPS every second
        if current_time - self.last_fps_time >= 1.0:
            fps = self.frame_count / (current_time - self.last_fps_time)
            self.metrics['fps'].append(fps)
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def get_current_stats(self):
        """
        Get current performance statistics.
        
        獲取當前性能統計信息。
        
        Returns:
            dict: Current performance statistics
        """
        stats = {}
        
        for metric_name, values in self.metrics.items():
            if values:
                stats[metric_name] = {
                    'current': values[-1],
                    'mean': np.mean(values),
                    'max': np.max(values),
                    'min': np.min(values),
                    'std': np.std(values)
                }
            else:
                stats[metric_name] = {
                    'current': 0,
                    'mean': 0,
                    'max': 0,
                    'min': 0,
                    'std': 0
                }
        
        # Add bottleneck information
        stats['bottlenecks'] = {
            'recent_alerts': len([a for a in self.bottleneck_alerts if time.time() - a['timestamp'] < 60]),
            'total_alerts': len(self.bottleneck_alerts)
        }
        
        # Add runtime information
        if self.start_time:
            stats['runtime_seconds'] = time.time() - self.start_time
        
        return stats
    
    def get_performance_report(self):
        """
        Generate a comprehensive performance report.
        
        生成綜合性能報告。
        
        Returns:
            dict: Detailed performance report
        """
        current_stats = self.get_current_stats()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'monitoring_duration': current_stats.get('runtime_seconds', 0),
            'system_performance': {
                'cpu': current_stats.get('cpu_percent', {}),
                'memory': current_stats.get('memory_percent', {}),
                'memory_rss_mb': current_stats.get('memory_rss_mb', {}),
                'gpu_memory_mb': current_stats.get('gpu_memory_mb', {})
            },
            'training_performance': {
                'fps': current_stats.get('fps', {}),
                'batch_time': current_stats.get('batch_time', {}),
                'sample_time': current_stats.get('sample_time', {}),
                'forward_time': current_stats.get('forward_time', {}),
                'backward_time': current_stats.get('backward_time', {})
            },
            'bottlenecks': current_stats.get('bottlenecks', {}),
            'recent_alerts': [
                alert for alert in self.bottleneck_alerts 
                if time.time() - alert['timestamp'] < 300  # Last 5 minutes
            ]
        }
        
        # Add recommendations
        report['recommendations'] = self._generate_recommendations(current_stats)
        
        return report
    
    def _generate_recommendations(self, stats):
        """
        Generate performance optimization recommendations.
        
        生成性能優化建議。
        
        Args:
            stats: Current performance statistics
            
        Returns:
            list: List of recommendations
        """
        recommendations = []
        
        # CPU recommendations
        cpu_stats = stats.get('cpu_percent', {})
        if cpu_stats.get('mean', 0) > 80:
            recommendations.append({
                'type': 'cpu_optimization',
                'priority': 'high',
                'message': 'High CPU usage detected. Consider reducing batch size or training frequency.',
                'suggestions': [
                    'Reduce BATCH_SIZE in config.py',
                    'Increase UPDATE_FREQUENCY to reduce training frequency',
                    'Enable multiprocessing if not already enabled'
                ]
            })
        
        # Memory recommendations
        memory_stats = stats.get('memory_percent', {})
        if memory_stats.get('mean', 0) > 75:
            recommendations.append({
                'type': 'memory_optimization',
                'priority': 'high',
                'message': 'High memory usage detected. Consider reducing memory-intensive parameters.',
                'suggestions': [
                    'Reduce MEMORY_CAPACITY in config.py',
                    'Reduce BATCH_SIZE',
                    'Disable memory-intensive logging features',
                    'Enable periodic garbage collection'
                ]
            })
        
        # Training speed recommendations
        batch_time = stats.get('batch_time', {})
        if batch_time.get('mean', 0) > 0.5:
            recommendations.append({
                'type': 'training_speed',
                'priority': 'medium',
                'message': 'Slow batch processing detected.',
                'suggestions': [
                    'Check if GPU is being utilized effectively',
                    'Consider reducing network complexity',
                    'Optimize data loading pipeline'
                ]
            })
        
        # FPS recommendations
        fps_stats = stats.get('fps', {})
        if fps_stats.get('mean', 0) < 30:
            recommendations.append({
                'type': 'fps_optimization',
                'priority': 'medium',
                'message': 'Low FPS detected. Training may be running slowly.',
                'suggestions': [
                    'Increase FRAME_SKIP to reduce computation per step',
                    'Reduce rendering if enabled during training',
                    'Optimize environment step processing'
                ]
            })
        
        return recommendations
    
    def save_report(self, filepath):
        """
        Save performance report to file.
        
        將性能報告保存到文件。
        
        Args:
            filepath: Path to save the report
        """
        try:
            report = self.get_performance_report()
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"Performance report saved to {filepath}")
            
        except Exception as e:
            print(f"Error saving performance report: {str(e)}")


class TimingContext:
    """
    Context manager for timing specific operations.
    
    用於計時特定操作的上下文管理器。
    """
    
    def __init__(self, monitor, operation_name):
        self.monitor = monitor
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            if self.operation_name in self.monitor.metrics:
                self.monitor.metrics[self.operation_name].append(duration)


class HyperparameterTuner:
    """
    Automated hyperparameter tuning system for DQN training.
    
    DQN 訓練的自動超參數調優系統。
    """
    
    def __init__(self, performance_monitor=None):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            performance_monitor: Optional performance monitor instance
        """
        self.performance_monitor = performance_monitor
        self.tuning_history = []
        self.current_params = None
        self.best_params = None
        self.best_score = float('-inf')
    
    def suggest_learning_rate(self, current_loss_trend, loss_variance):
        """
        Suggest learning rate adjustments based on loss trends.
        
        根據損失趨勢建議學習率調整。
        
        Args:
            current_loss_trend: Current trend in loss (positive = increasing)
            loss_variance: Variance in recent loss values
            
        Returns:
            dict: Learning rate adjustment suggestion
        """
        suggestion = {
            'current_lr': config.LEARNING_RATE,
            'recommended_lr': config.LEARNING_RATE,
            'adjustment_factor': 1.0,
            'reason': 'No change needed'
        }
        
        # If loss is increasing and variance is high, reduce learning rate
        if current_loss_trend > 0 and loss_variance > 0.1:
            adjustment_factor = 0.8
            suggestion.update({
                'recommended_lr': config.LEARNING_RATE * adjustment_factor,
                'adjustment_factor': adjustment_factor,
                'reason': 'Loss increasing with high variance - reduce learning rate'
            })
        
        # If loss is decreasing slowly and variance is low, increase learning rate
        elif current_loss_trend < -0.001 and loss_variance < 0.01:
            adjustment_factor = 1.2
            suggestion.update({
                'recommended_lr': min(config.LEARNING_RATE * adjustment_factor, 0.01),
                'adjustment_factor': adjustment_factor,
                'reason': 'Loss decreasing slowly with low variance - increase learning rate'
            })
        
        return suggestion
    
    def get_tuning_recommendations(self, training_metrics):
        """
        Get comprehensive hyperparameter tuning recommendations.
        
        獲取綜合超參數調優建議。
        
        Args:
            training_metrics: Current training metrics
            
        Returns:
            dict: Tuning recommendations
        """
        recommendations = {
            'timestamp': datetime.now().isoformat(),
            'learning_rate': None,
            'memory_params': None,
            'exploration_params': None,
            'overall_suggestions': []
        }
        
        # Learning rate recommendations
        if 'loss_trend' in training_metrics and 'loss_variance' in training_metrics:
            recommendations['learning_rate'] = self.suggest_learning_rate(
                training_metrics['loss_trend'],
                training_metrics['loss_variance']
            )
        
        # Overall training suggestions
        recommendations['overall_suggestions'] = self._generate_overall_suggestions(training_metrics)
        
        return recommendations
    
    def _generate_overall_suggestions(self, metrics):
        """
        Generate overall training suggestions.
        
        生成整體訓練建議。
        """
        suggestions = []
        
        # Check if training is progressing
        reward_trend = metrics.get('reward_trend', 0)
        if reward_trend <= 0:
            suggestions.append(
                "Consider adjusting hyperparameters: reward is not improving. "
                "Try slower exploration decay or different learning rate."
            )
        
        # Check training efficiency
        avg_fps = metrics.get('avg_fps', 0)
        if avg_fps < 30:
            suggestions.append(
                "Training FPS is low. Consider reducing batch size, frame stack, "
                "or enabling performance optimizations."
            )
        
        # Check resource utilization
        memory_usage = metrics.get('memory_usage', 0)
        if memory_usage > 0.9:
            suggestions.append(
                "High memory usage detected. Consider reducing memory capacity "
                "or batch size to prevent out-of-memory errors."
            )
        
        return suggestions 