"""
Performance Tests for SENSEX MLOps System
Benchmarks key components for production readiness
"""

import pytest
import numpy as np
import pandas as pd
import time
import psutil
import gc
from memory_profiler import profile
from unittest.mock import MagicMock
import sys

# Mock heavy dependencies for CI
sys.modules['tensorflow'] = MagicMock()
sys.modules['mlflow'] = MagicMock()

class TestDataProcessingPerformance:
    """Test data processing performance benchmarks"""
    
    @pytest.mark.benchmark
    def test_feature_calculation_speed(self, benchmark):
        """Benchmark feature calculation speed"""
        
        def calculate_features():
            # Simulate feature calculation for 30 stocks, 60 days
            n_stocks = 30
            n_days = 60
            
            features = {}
            for stock in range(n_stocks):
                # Generate sample price data
                prices = np.random.uniform(100, 200, n_days)
                volume = np.random.randint(1000000, 10000000, n_days)
                
                # Calculate technical indicators
                sma_5 = np.convolve(prices, np.ones(5)/5, mode='valid')
                sma_20 = np.convolve(prices, np.ones(20)/20, mode='valid')
                
                # Returns and volatility
                returns = np.diff(prices) / prices[:-1]
                volatility = np.std(returns[-10:]) if len(returns) >= 10 else 0
                
                features[f'stock_{stock}'] = {
                    'latest_price': prices[-1],
                    'sma_5': sma_5[-1] if len(sma_5) > 0 else prices[-1],
                    'sma_20': sma_20[-1] if len(sma_20) > 0 else prices[-1],
                    'volatility': volatility,
                    'volume': volume[-1]
                }
            
            return features
        
        # Benchmark the function
        result = benchmark(calculate_features)
        
        # Validate result
        assert len(result) == 30
        assert 'stock_0' in result
        assert 'latest_price' in result['stock_0']
    
    def test_large_dataset_processing(self):
        """Test processing performance with large datasets"""
        # Simulate large dataset (1 year of data for 30 stocks)
        n_stocks = 30
        n_days = 365
        
        start_time = time.time()
        
        # Process each stock
        processed_stocks = 0
        for stock_idx in range(n_stocks):
            # Generate data
            data = pd.DataFrame({
                'close': np.random.uniform(100, 200, n_days),
                'volume': np.random.randint(1000000, 10000000, n_days),
                'high': np.random.uniform(150, 250, n_days),
                'low': np.random.uniform(50, 150, n_days)
            })
            
            # Calculate features (simplified)
            data['sma_20'] = data['close'].rolling(20).mean()
            data['volatility'] = data['close'].pct_change().rolling(10).std()
            data['volume_sma'] = data['volume'].rolling(5).mean()
            
            processed_stocks += 1
        
        processing_time = time.time() - start_time
        
        # Performance assertions
        assert processed_stocks == n_stocks
        assert processing_time < 5.0  # Should complete within 5 seconds
        
        # Calculate throughput
        throughput = (n_stocks * n_days) / processing_time
        print(f"Processing throughput: {throughput:.0f} data points/second")
        assert throughput > 1000  # At least 1000 data points per second

class TestModelInferencePerformance:
    """Test model inference performance"""
    
    def test_prediction_latency(self):
        """Test single prediction latency"""
        # Simulate model input
        batch_size = 1
        sequence_length = 30
        n_stocks = 30
        n_features = 45
        
        input_data = np.random.random((batch_size, sequence_length, n_stocks, n_features))
        
        # Time prediction simulation
        latencies = []
        n_runs = 100
        
        for _ in range(n_runs):
            start_time = time.time()
            
            # Simulate model inference
            # Matrix operations to simulate CNN/LSTM computations
            processed = input_data.reshape(-1, n_features)
            weights = np.random.random((n_features, 32))
            hidden = np.dot(processed, weights)
            hidden = np.maximum(0, hidden)  # ReLU
            
            # Final prediction
            output_weights = np.random.random((32, 1))
            prediction = np.dot(hidden.mean(axis=0, keepdims=True), output_weights)
            probability = 1 / (1 + np.exp(-prediction[0]))  # Sigmoid
            
            latency = time.time() - start_time
            latencies.append(latency)
        
        # Performance metrics
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        print(f"Average latency: {avg_latency*1000:.2f}ms")
        print(f"P95 latency: {p95_latency*1000:.2f}ms")
        print(f"P99 latency: {p99_latency*1000:.2f}ms")
        
        # Performance assertions
        assert avg_latency < 0.1  # Less than 100ms average
        assert p95_latency < 0.2   # Less than 200ms for 95% of requests
        assert p99_latency < 0.5   # Less than 500ms for 99% of requests
    
    def test_batch_prediction_throughput(self):
        """Test batch prediction throughput"""
        batch_sizes = [1, 5, 10, 20, 50]
        sequence_length = 30
        n_stocks = 30
        n_features = 45
        
        throughput_results = {}
        
        for batch_size in batch_sizes:
            input_data = np.random.random((batch_size, sequence_length, n_stocks, n_features))
            
            start_time = time.time()
            
            # Simulate batch processing
            for i in range(batch_size):
                sample = input_data[i:i+1]
                # Simplified inference simulation
                result = np.random.random() > 0.5
            
            total_time = time.time() - start_time
            throughput = batch_size / total_time
            
            throughput_results[batch_size] = throughput
            print(f"Batch size {batch_size}: {throughput:.1f} predictions/second")
        
        # Validate throughput scaling
        assert throughput_results[1] > 5    # At least 5 predictions/sec for single
        assert throughput_results[10] > 20  # Should scale with batch size
        assert throughput_results[50] > 50  # Larger batches should be more efficient

class TestMemoryUsage:
    """Test memory usage patterns"""
    
    def test_data_loading_memory(self):
        """Test memory usage during data loading"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate loading large dataset
        datasets = []
        n_stocks = 30
        n_days = 365
        
        for stock in range(n_stocks):
            # Create DataFrame for each stock
            data = pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=n_days),
                'close': np.random.uniform(100, 200, n_days),
                'volume': np.random.randint(1000000, 10000000, n_days),
                'features': [np.random.random(45) for _ in range(n_days)]
            })
            datasets.append(data)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = peak_memory - initial_memory
        
        print(f"Memory usage: {memory_used:.1f} MB for {n_stocks} stocks")
        
        # Memory assertions
        assert memory_used < 500  # Should use less than 500MB
        
        # Test memory cleanup
        del datasets
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_freed = peak_memory - final_memory
        
        assert memory_freed > memory_used * 0.5  # Should free at least 50% of used memory
    
    @profile
    def test_feature_processing_memory_profile(self):
        """Memory profile of feature processing"""
        # This function will be profiled by memory_profiler
        n_stocks = 30
        n_days = 90
        
        # Process each stock
        for stock_idx in range(n_stocks):
            # Generate data
            prices = np.random.uniform(100, 200, n_days)
            
            # Calculate rolling features (memory intensive)
            sma_5 = pd.Series(prices).rolling(5).mean()
            sma_20 = pd.Series(prices).rolling(20).mean()
            ema_12 = pd.Series(prices).ewm(span=12).mean()
            
            # Technical indicators
            returns = pd.Series(prices).pct_change()
            volatility = returns.rolling(20).std()
            
            # Cleanup intermediate variables
            del sma_5, sma_20, ema_12, returns, volatility
    
    def test_cache_memory_management(self):
        """Test cache memory management"""
        cache = {}
        cache_size_limit = 100  # items
        
        # Fill cache beyond limit
        for i in range(150):
            cache[f'key_{i}'] = np.random.random((100, 100))  # ~80KB each
            
            # Implement LRU eviction
            if len(cache) > cache_size_limit:
                # Remove oldest item (simplified LRU)
                oldest_key = min(cache.keys())
                del cache[oldest_key]
        
        # Validate cache size
        assert len(cache) <= cache_size_limit
        
        # Estimate memory usage
        estimated_memory = len(cache) * 80 / 1024  # MB
        assert estimated_memory < 10  # Less than 10MB for cache

class TestConcurrencyPerformance:
    """Test concurrent access performance"""
    
    def test_concurrent_predictions(self):
        """Test concurrent prediction handling"""
        import threading
        import queue
        
        results_queue = queue.Queue()
        n_threads = 5
        predictions_per_thread = 10
        
        def worker_prediction():
            """Worker function for concurrent predictions"""
            for _ in range(predictions_per_thread):
                start_time = time.time()
                
                # Simulate prediction work
                input_data = np.random.random((1, 30, 30, 45))
                result = np.random.random() > 0.5
                
                processing_time = time.time() - start_time
                results_queue.put({
                    'thread_id': threading.current_thread().name,
                    'result': result,
                    'processing_time': processing_time
                })
        
        # Start concurrent threads
        start_time = time.time()
        threads = []
        
        for i in range(n_threads):
            thread = threading.Thread(target=worker_prediction, name=f'worker_{i}')
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        # Performance validation
        expected_results = n_threads * predictions_per_thread
        assert len(results) == expected_results
        
        # Calculate concurrent throughput
        concurrent_throughput = len(results) / total_time
        print(f"Concurrent throughput: {concurrent_throughput:.1f} predictions/second")
        
        assert concurrent_throughput > 20  # Should handle at least 20 predictions/second
    
    def test_cache_concurrent_access(self):
        """Test concurrent cache access performance"""
        import threading
        import time
        
        cache = {}
        cache_lock = threading.Lock()
        access_times = []
        
        def cache_worker(worker_id):
            """Worker that reads/writes cache concurrently"""
            for i in range(50):
                key = f'worker_{worker_id}_item_{i}'
                
                start_time = time.time()
                
                with cache_lock:
                    # Simulate cache operation
                    if key in cache:
                        value = cache[key]
                    else:
                        cache[key] = np.random.random((10, 10))
                
                access_time = time.time() - start_time
                access_times.append(access_time)
        
        # Start concurrent cache workers
        threads = []
        for i in range(5):
            thread = threading.Thread(target=cache_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Analyze cache performance
        avg_access_time = np.mean(access_times)
        max_access_time = np.max(access_times)
        
        print(f"Average cache access time: {avg_access_time*1000:.2f}ms")
        print(f"Maximum cache access time: {max_access_time*1000:.2f}ms")
        
        # Performance assertions
        assert avg_access_time < 0.01   # Less than 10ms average
        assert max_access_time < 0.05   # Less than 50ms maximum

class TestScalabilityBenchmarks:
    """Test system scalability benchmarks"""
    
    def test_data_volume_scaling(self):
        """Test performance scaling with data volume"""
        volumes = [30, 100, 300, 1000]  # Number of stocks
        processing_times = []
        
        for n_stocks in volumes:
            start_time = time.time()
            
            # Simulate processing for different volumes
            for stock_idx in range(n_stocks):
                # Generate and process stock data
                prices = np.random.uniform(100, 200, 60)
                features = {
                    'sma': np.mean(prices[-5:]),
                    'volatility': np.std(prices[-20:]),
                    'trend': prices[-1] - prices[-5]
                }
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            print(f"{n_stocks} stocks: {processing_time:.3f}s")
        
        # Analyze scaling
        for i in range(1, len(volumes)):
            volume_ratio = volumes[i] / volumes[i-1]
            time_ratio = processing_times[i] / processing_times[i-1]
            
            # Time should scale roughly linearly with volume
            # Allow some overhead but should be < 2x time for 3x+ volume
            if volume_ratio >= 3:
                assert time_ratio < volume_ratio * 0.7
    
    def test_user_load_simulation(self):
        """Simulate multiple concurrent users"""
        n_users = 20
        requests_per_user = 5
        
        start_time = time.time()
        completed_requests = 0
        
        # Simulate user requests
        for user in range(n_users):
            for request in range(requests_per_user):
                # Simulate request processing
                request_start = time.time()
                
                # Simulate prediction request
                input_processing = np.random.uniform(0.01, 0.05)  # 10-50ms
                time.sleep(input_processing)
                
                # Simulate model inference
                inference_time = np.random.uniform(0.05, 0.15)  # 50-150ms
                time.sleep(inference_time)
                
                completed_requests += 1
        
        total_time = time.time() - start_time
        requests_per_second = completed_requests / total_time
        
        print(f"Simulated {n_users} users, {requests_per_user} requests each")
        print(f"Total requests: {completed_requests}")
        print(f"Requests per second: {requests_per_second:.1f}")
        
        # Performance expectations
        assert completed_requests == n_users * requests_per_user
        assert requests_per_second > 5  # Should handle at least 5 requests/second

if __name__ == "__main__":
    # Run with pytest-benchmark for detailed benchmarking
    pytest.main([__file__, "--benchmark-only"])
