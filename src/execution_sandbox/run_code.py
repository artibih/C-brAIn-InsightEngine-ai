import asyncio
import tempfile
import subprocess
import textwrap
import os
from typing import Tuple
import sys
import io
import traceback


def run_python_code(code: str, globals_dict: dict, return_globals: bool = False) -> Tuple[str, str, dict]:
    stdout_buffer = io.StringIO()
    old_stdout = sys.stdout
    stderr = None
    try:
        sys.stdout = stdout_buffer
        try:
            import matplotlib
            matplotlib.use("Agg")
            exec(code, globals_dict)

        except SystemExit as e:
            stderr = f"SystemExit: {e.code}\n"

        except Exception:
            stderr = traceback.format_exc()

                
        stdout = stdout_buffer.getvalue()
        attempt_artifact_dir = globals_dict.get("attempt_artifact_dir")

        if stdout and attempt_artifact_dir:
            _save_stdout_structured(stdout, attempt_artifact_dir)

        if return_globals:
            return stdout_buffer.getvalue(), stderr, globals_dict

        return stdout_buffer.getvalue(), stderr

    except Exception:
        err = traceback.format_exc()

        if return_globals:
            return "", err, globals_dict

        return "", err, None

    finally:
        sys.stdout = old_stdout

def summarize_error(stderr: str) -> str:
    lines = stderr.strip().splitlines()
    return "\n".join(lines[-25:]) 


def _save_stdout_structured(stdout: str, artifact_dir: str):
    import pandas as pd
    import os

    lines = [line.strip() for line in stdout.splitlines() if line.strip()]

    parsed_rows = []

    for line in lines:
        if ":" not in line or line.startswith("PLOT_SAVED"):
            continue

        row_dict = {}
        parts = line.split(",")

        for part in parts:
            if ":" not in part:
                continue

            key, value = part.split(":", 1)
            key = key.strip()
            value = value.strip()

            try:
                if value.lower() == "false":
                    value = False
                elif value.lower() == "true":
                    value = True
                else:
                    value = float(value)
            except:
                pass

            row_dict[key] = value

        if row_dict:
            parsed_rows.append(row_dict)

