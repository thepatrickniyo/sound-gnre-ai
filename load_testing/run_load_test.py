#!/usr/bin/env python3
"""
Script to run load tests and generate results report.
"""
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import argparse


def run_locust_test(host, users, spawn_rate, run_time, headless=True, html_report=None, csv_report=None):
    """
    Run Locust load test.
    
    Args:
        host: API host URL
        users: Number of concurrent users
        spawn_rate: Users to spawn per second
        run_time: Test duration (e.g., "5m", "30s")
        headless: Run in headless mode
        html_report: Path to HTML report
        csv_report: Path to CSV report prefix
    """
    locustfile = Path(__file__).parent / "locustfile.py"
    
    if not locustfile.exists():
        print(f"Error: {locustfile} not found!")
        return False
    
    # Use python3 -m locust to ensure we use the correct Python environment
    import sys
    cmd = [
        sys.executable, "-m", "locust",
        "-f", str(locustfile),
        "--host", host,
        "--users", str(users),
        "--spawn-rate", str(spawn_rate),
        "--run-time", run_time,
    ]
    
    if headless:
        cmd.append("--headless")
    
    if html_report:
        cmd.extend(["--html", html_report])
    
    if csv_report:
        cmd.extend(["--csv", csv_report])
    
    print(f"Running load test with command: {' '.join(cmd)}")
    print(f"Host: {host}")
    print(f"Users: {users}, Spawn Rate: {spawn_rate}/s, Duration: {run_time}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running load test: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: Locust not found. Please install it with: pip install locust")
        return False


def generate_results_summary(results_dir):
    """Generate a summary of all test results."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        results_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "test_runs": [],
        "generated_at": datetime.now().isoformat()
    }
    
    # Find all CSV result files
    csv_files = list(results_dir.glob("*.csv"))
    if csv_files:
        print(f"\nFound {len(csv_files)} result files")
        for csv_file in sorted(csv_files):
            print(f"  - {csv_file.name}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run load tests for Sound Genre AI API")
    parser.add_argument(
        "--host",
        default="http://localhost:8000",
        help="API host URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--users",
        type=int,
        default=10,
        help="Number of concurrent users (default: 10)"
    )
    parser.add_argument(
        "--spawn-rate",
        type=int,
        default=2,
        help="Users to spawn per second (default: 2)"
    )
    parser.add_argument(
        "--run-time",
        default="2m",
        help="Test duration (e.g., '5m', '30s') (default: 2m)"
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Run with web UI instead of headless mode"
    )
    parser.add_argument(
        "--results-dir",
        default="load_testing/results",
        help="Directory to save results (default: load_testing/results)"
    )
    
    args = parser.parse_args()
    
    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for this test run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_name = f"load_test_{args.users}users_{timestamp}"
    
    html_report = str(results_dir / f"{test_name}.html")
    csv_prefix = str(results_dir / test_name)
    
    print("=" * 60)
    print("Sound Genre AI - Load Testing")
    print("=" * 60)
    print(f"Results will be saved to: {results_dir}")
    print(f"HTML Report: {html_report}")
    print(f"CSV Reports: {csv_prefix}_*.csv")
    print("=" * 60)
    
    # Run the test
    success = run_locust_test(
        host=args.host,
        users=args.users,
        spawn_rate=args.spawn_rate,
        run_time=args.run_time,
        headless=not args.ui,
        html_report=html_report,
        csv_report=csv_prefix
    )
    
    if success:
        print("\n" + "=" * 60)
        print("Load test completed successfully!")
        print("=" * 60)
        print(f"\nResults saved to: {results_dir}")
        print(f"View HTML report: {html_report}")
        
        # Generate summary
        summary = generate_results_summary(results_dir)
        summary_file = results_dir / f"{test_name}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_file}")
    else:
        print("\nLoad test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

