"""
Queue Execution Script
Iterates through all .json config files in the queue/ folder and executes them
sequentially. Each config must have a "queue" field with an "execution" command.

Completed configs are moved to queue/completed/, failed ones to queue/failed/.
"""

import json
import os
import re
import sys
import shutil
import subprocess
import time
from pathlib import Path
from datetime import datetime

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent.parent
print(f"Script directory: {SCRIPT_DIR}")
QUEUE_DIR = SCRIPT_DIR / "queue"
COMPLETED_DIR = QUEUE_DIR / "completed"
FAILED_DIR = QUEUE_DIR / "failed"
VENV_PYTHON = SCRIPT_DIR / ".venv" / "bin" / "python3"
ACCELERATE = SCRIPT_DIR / ".venv" / "bin" / "accelerate"

# Ensure output directories exist
COMPLETED_DIR.mkdir(parents=True, exist_ok=True)
FAILED_DIR.mkdir(parents=True, exist_ok=True)

# Import pushover notification helper
sys.path.insert(0, str(SCRIPT_DIR))
from src.utils.pushover import send_notification

# ANSI Colors for terminal output

HEADER = "\033[95m"
BLUE = "\033[94m"
CYAN = "\033[96m"
GREEN = "\033[92m"
WARNING = "\033[93m"
FAIL = "\033[91m"
ENDC = "\033[0m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"

def print_warning(message: str):
    print(f"{WARNING}[QUEUE] Warning: {message}{ENDC}")
def print_error(message: str):
    print(f"{FAIL}[QUEUE] Error: {message}{ENDC}")
def print_info(message: str):
    print(f"{CYAN}[QUEUE] {message}{ENDC}")
def print_header(message: str):
    print(f"{CYAN}{'='*60}{ENDC}")
    print(f"{CYAN}{BOLD}[QUEUE] {message}{ENDC}")
    print(f"{CYAN}{'='*60}{ENDC}")

def load_config(config_path: Path) -> dict:
    """Load and return a JSON config file."""
    with open(config_path, "r") as f:
        return json.load(f)


def build_command(execution_str: str, config_path: Path) -> str:
    """
    Build the shell command from the execution string.
    - Replaces $CONFIG with the absolute path of the config file.
    - Replaces '$PYTHON' with the venv python3 if available.
    """
    command = execution_str.replace("$CONFIG", str(config_path.resolve()))

    # If the command starts with $PYTHON, use the venv interpreter
    if VENV_PYTHON.exists() and command.strip().startswith("$PYTHON"):
        command = command.replace("$PYTHON", str(VENV_PYTHON), 1)

    return command


def execute_config(config_path: Path):
    """
    Execute a single queue config:
    1. Parse the config and extract queue settings.
    2. Send start notification if configured.
    3. Run the command, capturing all stdout/stderr to a log.
    4. On success: move config to completed/, optionally copy to output_dir.
    5. On failure: move config + log to failed/, send failure notification.
    """
    config_name = config_path.name
    timestamp = datetime.now().strftime("%Y-%m-%dT%H_%M_%S")
    print()  # blank line before header
    print_header(f"Processing: {config_name}")

    try:
        config = load_config(config_path)
    except Exception as e:
        print_error(f"Failed to load config {config_name}: {e}")
        # Move broken config to failed
        _move_to_dir(config_path, FAILED_DIR, timestamp)
        return

    queue_settings = config.get("queue", {})
    bash_command_str = queue_settings.get("bash_command")

    if not bash_command_str:
        print_error(f"No 'bash_command' field found under 'queue' in {config_name}")
        _move_to_dir(config_path, FAILED_DIR, timestamp)
        return

    command = build_command(bash_command_str, config_path)
    print_info(f"Command: {ENDC}{BOLD}{command}")

    # Notification flags
    notify_start = queue_settings.get("send_notification_on_start", False)
    notify_completion = queue_settings.get("send_notification_on_completion", False)
    notify_failure = queue_settings.get("send_notification_on_failure", False)
    copy_config = queue_settings.get("copy_config_to_output_dir", False)

    # Output dir from logging section
    output_dir = config.get("logging", {}).get("output_dir")

    # --- Send start notification ---
    if notify_start:
        comment = config.get("logging", {}).get("comment", "")
        send_notification(
            title=f"Queue started: {config_name}",
            message=f"Command: {bash_command_str}\n{comment}",
        )

    # --- Execute the command ---
    log_lines = []
    success = False
    start_time = time.time()

    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(SCRIPT_DIR),
        )

        # Stream output line by line, capturing to log
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_lines.append(line)

        process.wait()
        elapsed = time.time() - start_time
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))

        if process.returncode == 0:
            success = True
            print_info(f"{ENDC}{BOLD}{config_name}{ENDC} completed in {ENDC}{BOLD}{elapsed_str}")
        else:
            print_error(f"{config_name} exited with code {process.returncode} after {elapsed_str}")

    except Exception as e:
        elapsed = time.time() - start_time
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        log_lines.append(f"\n[QUEUE] Exception during execution: {e}\n")
        print_error(f"EXCEPTION: {config_name} failed with: {e}")

    # --- Post-execution ---
    log_content = "".join(log_lines)

    if success:
        # Copy config to output dir if requested
        if copy_config and output_dir:
            _copy_config_to_output_dir(config_path, output_dir)

        # Move config to completed
        _move_to_dir(config_path, COMPLETED_DIR, timestamp)

        # Send completion notification
        if notify_completion:
            send_notification(
                title=f"Queue completed: {config_name}",
                message=f"Finished in {elapsed_str}.\nCommand: {bash_command_str}",
            )
    else:
        # Save log to failed directory
        log_filename = f"{timestamp}_{config_path.stem}.log"
        log_path = FAILED_DIR / log_filename
        with open(log_path, "w") as f:
            f.write(log_content)
        print_warning(f"Log saved to: {log_path}")

        # Move config to failed
        _move_to_dir(config_path, FAILED_DIR, timestamp)

        # Send failure notification
        if notify_failure:
            # Include last few lines of log in notification
            tail = "".join(log_lines[-20:]) if log_lines else "No output captured."
            send_notification(
                title=f"Queue FAILED: {config_name}",
                message=f"Failed after {elapsed_str}.\nCommand: {bash_command_str}\n\nLast output:\n{tail[:500]}",
            )
    return success


def _copy_config_to_output_dir(config_path: Path, output_dir: str):
    """Copy the config file to the output directory."""
    dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / config_path.name
    shutil.copy2(config_path, dest)
    print_info(f"Config copied to: {ENDC}{BOLD}{dest}")
    


def _move_to_dir(config_path: Path, target_dir: Path, timestamp: str):
    """Move config file to target directory with timestamp prefix to avoid collisions."""
    # Shorten timestamp for better readability to mm-ddThh-mm
    timestamp = datetime.now().strftime("%m-%dT%H-%M")
    # Check if the config_path has a timestamp pattern in the filename already, if so, replace it instead of adding a new one
    existing_timestamp_match = re.search(r"^\d{2}-\d{2}T\d{2}-\d{2}_", config_path.name)
    if existing_timestamp_match:
        # If there's an existing timestamp, replace it with the new one
        new_name = f"{timestamp}_{config_path.name[len(existing_timestamp_match.group()):]}"
    else:
        new_name = f"{timestamp}_{config_path.name}"
    dest = target_dir / new_name
    shutil.move(str(config_path), str(dest))
    print_info(f"Moved {ENDC}{BOLD}{config_path.name}{ENDC}{CYAN} -> {dest}")

def re_run_recent_config():
    """Find the most recent completed or failed config and ask the user if they want to re-run it."""
    print_warning("No config files found in queue/. Nothing to do.")
    recent_completed = list(COMPLETED_DIR.glob("*.json"))
    recent_failed = list(FAILED_DIR.glob("*.json"))
    recent_configs = sorted(recent_completed + recent_failed, key=os.path.getmtime, reverse=True)
    if recent_configs:
        most_recent = recent_configs[0]
        print_info(f"Most recent config: {ENDC}{BOLD}{most_recent}{ENDC} (from {'completed' if most_recent in recent_completed else 'failed'})")
        # Catch KeyboardInterrupt to allow user to cancel if they don't want to be prompted
        try:
            response = input("Do you want to re-run this config? (Y/n): ").strip().lower()
        except KeyboardInterrupt:
            print_info("Interrupted by user.")
            return False
        if not response == "n":
            # Move the most recent config back to the queue directory
            shutil.copy2(most_recent, QUEUE_DIR / most_recent.name)
            print_info(f"Copied {most_recent.name} back to queue/. Please run the script again to execute it.")
            return True
        else:
            print_info("No config will be re-run. Exiting.")
            # system exit with code 0 to indicate normal exit, even though nothing was done
            return False
    else:
        print_info("No recent configs found in completed/ or failed/. Exiting.")
        return False
    

def run_queue():
    """Discover and execute all .json config files in the queue directory."""
    config_files = sorted(QUEUE_DIR.glob("*.json"))
    

    
    if not config_files:
        if re_run_recent_config():
            config_files = sorted(QUEUE_DIR.glob("*.json"))
        else:
            return
    completed_count = 0
    failed_count = 0
    while config_files:
        config_path = config_files[0]  # Always take the first one (sorted by name)
        print_info(f"Found {ENDC}{len(config_files)}{CYAN} config(s) in queue.")

        try:
            success = execute_config(config_path)
            if success:
                completed_count += 1
            else:
                failed_count += 1
        except Exception as e:
            # Catch-all so one bad config never kills the entire queue
            print_error(f"UNHANDLED ERROR processing {config_path.name}: {e}")
            timestamp = datetime.now().strftime("%Y-%m-%dT%H_%M_%S")
            try:
                _move_to_dir(config_path, FAILED_DIR, timestamp)
            except Exception:
                pass
            failed_count += 1
        config_files = sorted(QUEUE_DIR.glob("*.json"))

    print()
    print_header("All configs processed.")
    # Print queue summary
    if failed_count == 0:
        print_info(f"Summary: {completed_count} completed, {failed_count} failed.")
    else:
        print_warning(f"Summary: {completed_count} completed, {failed_count} failed.")


if __name__ == "__main__":
    run_queue()
