import subprocess
import sys
import os

def run_command(cmd, capture_output=False, text=False, check=False, cwd=None, debug=False, timeout=None):
    """
    Runs a subprocess command and optionally prints it if debug is enabled.

    Args:
        cmd (list): The command and its arguments.
        capture_output (bool): Whether to capture stdout/stderr.
        text (bool): Whether to decode stdout/stderr as text.
        check (bool): If True, raise CalledProcessError on non-zero exit code.
        cwd (str, optional): The working directory for the command.
        debug (bool): If True, print the command being executed.
        timeout (int, optional): If set, the command will be killed if it runs longer than timeout seconds.

    Returns:
        subprocess.CompletedProcess: The result of the subprocess run.
    """
    if debug:
        cmd_str = ' '.join(cmd)
        cwd_str = f" (cwd: {cwd})" if cwd else ""
        timeout_str = f" (timeout: {timeout}s)" if timeout else ""
        print(f"Executing command: {cmd_str}{cwd_str}{timeout_str}", file=sys.stderr)

    try:
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=text,
            check=check,
            cwd=cwd,
            timeout=timeout # Pass timeout here
        )
        if debug and capture_output:
             print(f"Command stdout:\n{result.stdout}", file=sys.stderr)
             print(f"Command stderr:\n{result.stderr}", file=sys.stderr)
        return result
    except FileNotFoundError:
        print(f"Error: Command not found: {cmd[0]}", file=sys.stderr)
        raise # Re-raise the exception
    except subprocess.TimeoutExpired:
        print(f"Error: Command '{cmd[0]}' timed out after {timeout} seconds.", file=sys.stderr)
        # The process is killed by subprocess.run, but we might want to clean up
        # or re-raise a specific exception for the caller to handle.
        # For now, re-raise to be caught by the caller (e.g., repo_formatter).
        raise
    except subprocess.CalledProcessError as e:
        if debug:
             print(f"Command failed with exit code {e.returncode}", file=sys.stderr)
             print(f"Stderr: {e.stderr}", file=sys.stderr)
        raise # Re-raise the exception

