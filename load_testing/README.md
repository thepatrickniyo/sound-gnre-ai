# Load Testing for Sound Genre AI

This directory contains load testing tools and results for the Sound Genre AI API.

## Setup

1. Install Locust:
```bash
pip install locust
```

2. Ensure the API is running:
```bash
# In one terminal
python scripts/run_api.py
```

## Running Load Tests

### Quick Start

Run a basic load test with default settings:
```bash
python load_testing/run_load_test.py
```

### Custom Test Configuration

```bash
# Test with 50 concurrent users, spawn 5 per second, run for 5 minutes
python load_testing/run_load_test.py --users 50 --spawn-rate 5 --run-time 5m

# Test with web UI (interactive mode)
python load_testing/run_load_test.py --ui

# Test against a different host
python load_testing/run_load_test.py --host http://localhost:8000
```

### Using Locust Directly

```bash
# Start Locust web UI
locust -f load_testing/locustfile.py --host=http://localhost:8000

# Run headless
locust -f load_testing/locustfile.py --host=http://localhost:8000 --headless --users 10 --spawn-rate 2 --run-time 2m
```

## Test Scenarios

The load test simulates the following user behaviors:

1. **Health Check** (5x weight) - Most common request
2. **Predict Genre** (10x weight) - Main functionality
3. **Model Status** (3x weight) - Status checks
4. **Get Metrics** (2x weight) - Performance metrics
5. **Get Stats** (1x weight) - API statistics

## Results

Test results are saved in the `results/` directory:
- `load_test_*_stats.csv` - Request statistics
- `load_test_*_failures.csv` - Failed requests
- `load_test_*_exceptions.csv` - Exceptions
- `load_test_*.html` - HTML report
- `load_test_*_summary.json` - Test summary

## Metrics Collected

- **Response Time**: p50, p95, p99 percentiles
- **Requests per Second (RPS)**: Throughput
- **Failure Rate**: Percentage of failed requests
- **Response Times by Endpoint**: Per-endpoint performance

