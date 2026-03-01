import subprocess
import sys
import os
import time
import signal
import threading
from datetime import datetime

# ── ANSI Colors for terminal output ─────────────────────────────────────────
class Color:
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    RESET   = "\033[0m"
    BOLD    = "\033[1m"

# ── Service Definitions ──────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
PYTHON    = sys.executable  # Uses the currently active venv's Python

SERVICES = [
    {
        "name"   : "Emotion Detection",
        "command": [PYTHON, "behavioral_anomaly/services/emotion_detection/emotion_service.py"],
        "color"  : Color.CYAN,
        "port"   : 8001,  # update if different
    },
    {
        "name"   : "Drowsiness Detection",
        "command": [PYTHON, "behavioral_anomaly/services/drowsiness_detection/drowsiness_service.py"],
        "color"  : Color.BLUE,
        "port"   : 8002,
    },
    {
        "name"   : "Pose Detection",
        "command": [PYTHON, "behavioral_anomaly/services/pose_detection/pose_service.py"],
        "color"  : Color.MAGENTA,
        "port"   : 8004,  # update if different
    },
    {
        "name"   : "Behavioral Analysis",
        "command": [PYTHON, "behavioral_anomaly/services/behavioral_analysis/behavioral_service.py"],
        "color"  : Color.YELLOW,
        "port"   : 8005,  # update if different
        "delay"  : 5,     # Wait 5s for above 3 services to boot before starting
    },
    {
        "name"   : "Dashboard (Streamlit)",
        "command": [
            PYTHON, "-m", "streamlit", "run",
            "behavioral_anomaly/services/dashboard/dashboard_service.py",
            "--server.port", "8501",
            "--server.headless", "true",
        ],
        "color"  : Color.GREEN,
        "port"   : 8501,
        "delay"  : 8,     # Wait for behavioral analysis to be ready
    },
]

# ── Global process registry ──────────────────────────────────────────────────
processes = []


def log(service_name, message, color=Color.RESET):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{color}{Color.BOLD}[{timestamp}] [{service_name}]{Color.RESET}{color} {message}{Color.RESET}")


def stream_output(process, service_name, color):
    """Stream stdout and stderr from a subprocess to the terminal."""
    def _stream(pipe, label):
        try:
            for line in iter(pipe.readline, b""):
                decoded = line.decode("utf-8", errors="replace").rstrip()
                if decoded:
                    log(service_name, decoded, color)
        except Exception:
            pass

    t_out = threading.Thread(target=_stream, args=(process.stdout, "OUT"), daemon=True)
    t_err = threading.Thread(target=_stream, args=(process.stderr, "ERR"), daemon=True)
    t_out.start()
    t_err.start()


def start_service(service):
    """Start a single service as a subprocess."""
    delay = service.get("delay", 0)
    if delay:
        log(service["name"], f"⏳ Waiting {delay}s for dependencies...", service["color"])
        time.sleep(delay)

    log(service["name"], f"🚀 Starting on port {service['port']}...", service["color"])

    process = subprocess.Popen(
        service["command"],
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},  # Ensures real-time log output
    )
    processes.append((service["name"], process))
    stream_output(process, service["name"], service["color"])
    log(service["name"], f"✅ Started (PID: {process.pid})", service["color"])
    return process


def shutdown_all(signum=None, frame=None):
    """Gracefully terminate all running services."""
    print(f"\n{Color.RED}{Color.BOLD}🛑 Shutting down all services...{Color.RESET}")
    for name, process in processes:
        if process.poll() is None:  # Still running
            process.terminate()
            log(name, f"🔴 Terminated (PID: {process.pid})", Color.RED)
    
    # Wait up to 5s for processes to exit, then force kill
    time.sleep(2)
    for name, process in processes:
        if process.poll() is None:
            process.kill()
            log(name, f"💀 Force killed (PID: {process.pid})", Color.RED)

    print(f"{Color.RED}{Color.BOLD}✅ All services stopped.{Color.RESET}")
    sys.exit(0)


def monitor_processes():
    """Monitor and report if any service crashes unexpectedly."""
    while True:
        time.sleep(10)
        for name, process in processes:
            if process.poll() is not None:
                log(name, f"💥 CRASHED with exit code {process.returncode}", Color.RED)


# ── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Register Ctrl+C and kill signal handlers
    signal.signal(signal.SIGINT,  shutdown_all)
    signal.signal(signal.SIGTERM, shutdown_all)

    print(f"\n{Color.BOLD}{'='*55}")
    print(f"  🧠  Behavioral Anomaly Detection - Service Launcher")
    print(f"{'='*55}{Color.RESET}\n")

    # Start services 1-3 immediately in parallel threads
    threads = []
    for service in SERVICES:
        t = threading.Thread(target=start_service, args=(service,), daemon=True)
        threads.append(t)
        t.start()
        time.sleep(0.3)  # Small stagger to avoid terminal output collision

    # Start crash monitor in background
    monitor_thread = threading.Thread(target=monitor_processes, daemon=True)
    monitor_thread.start()

    print(f"\n{Color.GREEN}{Color.BOLD}🟢 All services launched! Press Ctrl+C to stop all.{Color.RESET}\n")

    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown_all()
