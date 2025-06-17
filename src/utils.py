import subprocess
import sys
import os

def run_command(cmd, capture_output=False, text=False, check=False, cwd=None, debug=False):
    """
    Runs a subprocess command and optionally prints it if debug is enabled.

    Args:
        cmd (list): The command and its arguments.
        capture_output (bool): Whether to capture stdout/stderr.
        text (bool): Whether to decode stdout/stderr as text.
        check (bool): If True, raise CalledProcessError on non-zero exit code.
        cwd (str, optional): The working directory for the command.
        debug (bool): If True, print the command being executed.

    Returns:
        subprocess.CompletedProcess: The result of the subprocess run.
    """
    if debug:
        cmd_str = ' '.join(cmd)
        cwd_str = f" (cwd: {cwd})" if cwd else ""
        print(f"Executing command: {cmd_str}{cwd_str}", file=sys.stderr)

    try:
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=text,
            check=check,
            cwd=cwd
        )
        if debug and capture_output:
             print(f"Command stdout:\n{result.stdout}", file=sys.stderr)
             print(f"Command stderr:\n{result.stderr}", file=sys.stderr)
        return result
    except FileNotFoundError:
        print(f"Error: Command not found: {cmd[0]}", file=sys.stderr)
        raise # Re-raise the exception
    except subprocess.CalledProcessError as e:
        if debug:
             print(f"Command failed with exit code {e.returncode}", file=sys.stderr)
             print(f"Stderr: {e.stderr}", file=sys.stderr)
        raise # Re-raise the exception

