# Queue Documentation

## How to setup a config for the queue:
First, add the following fields to your `.json` file:

```json
{
...
  "queue": {
    "bash_command": "<BASH_COMMAND>",
    "send_notification_on_start": true/false,
    "send_notification_on_completion": true/false,
    "send_notification_on_failure": true/false,
    "copy_config_to_output_dir": true/false
  },
...
}
```
The `bash_command` field should contain the bash command that you want to execute. You can use `$CONFIG` in the command as a placeholder for the path to the current config file. Additionally, when executing python scripts, use `$PYTHON` as this will make it use the virtual environment's python interpreter if it exists. `$ACCELERATE` is also available as a placeholder for the accelerate command if you want to run your script with accelerate.

### Example:
```json
{
...
  "queue": {
    "bash_command": "$PYTHON src/testing/test_multiple_splits.py --config $CONFIG --train --evaluate",
    "send_notification_on_start": false,
    "send_notification_on_completion": true,
    "send_notification_on_failure": true,
    "copy_config_to_output_dir": true
  },
...
}
```

## Start the queue:
To start the queue, run the following command in your terminal:
```bash
python3 scripts/queue_manager.py
```
Or use the handy™ alias:
```bash
ms_queue
```

## Quick copilot overview of how the queue manager works:
The queue manager automatically executes machine learning experiments in sequence:

1. **Discovery**: Scans the `queue/` folder for all `.json` config files (sorted alphabetically)

2. **Execution**: For each config file:
   - Loads the JSON and extracts the `queue.bash_command` field
   - Replaces `$CONFIG` with the absolute path to the config file
   - Replaces `$PYTHON` with the virtual environment's Python interpreter (`.venv/bin/python3`)
   - Sends start notification (if enabled)
   - Executes the command while streaming output to the terminal
   - Captures all stdout/stderr to a log buffer

3. **Post-execution**:
   - **On Success**:
     - Moves config to `queue/completed/` with timestamp prefix
     - Optionally copies config to the output directory (if `copy_config_to_output_dir: true`)
     - Sends completion notification (if enabled)
   - **On Failure**:
     - Saves execution log to `queue/failed/<timestamp>_<config_name>.log`
     - Moves config to `queue/failed/` with timestamp prefix
     - Sends failure notification with last 20 lines of output (if enabled)

4. **Repeat**: Continues with the next config file until the queue is empty

5. **Empty Queue Handling**: If no configs are found, offers to re-run the most recent completed or failed config

**Key Features**:
- Sequential execution ensures one experiment at a time (no resource conflicts)
- Colored terminal output for easy monitoring
- Automatic timestamping prevents filename collisions
- Persistent logs for debugging failed runs
- Optional mobile notifications via Pushover
