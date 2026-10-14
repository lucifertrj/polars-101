import polars as pl
from pathlib import Path
from typing import Annotated, Optional
import json
import os
import resource
import subprocess
import sys
import traceback

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_CSV = DATA_DIR / "sales_data.csv"


def get_csv_context(csv_path: Optional[str] = None) -> str:
    """
    Read CSV data using Polars and return context about the data.
    This includes schema, sample rows, and basic statistics.

    Args:
        csv_path: Path to the CSV file. If None, uses default sales_data.csv

    Returns:
        A string containing data context for the LLM to generate code.
    """
    file_path = Path(csv_path) if csv_path else DEFAULT_CSV

    if not file_path.exists():
        return f"Error: CSV file not found at {file_path}"

    try:
        df = pl.read_csv(file_path)

        context_parts = []

        # File path info
        context_parts.append(f"CSV File: {file_path}")
        context_parts.append(f"CSV Path for code: '{file_path}'")

        context_parts.append("\n=== SCHEMA ===")
        context_parts.append(f"Columns ({len(df.columns)}):")
        for col_name, dtype in zip(df.columns, df.dtypes):
            context_parts.append(f"  - {col_name}: {dtype}")

        context_parts.append(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")

        context_parts.append("\n=== SAMPLE DATA (first 5 rows) ===")
        context_parts.append(str(df.head(5)))

        # Basic statistics for numeric columns
        context_parts.append("\n=== NUMERIC STATISTICS ===")
        numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
        if numeric_cols:
            stats_df = df.select(numeric_cols).describe()
            context_parts.append(str(stats_df))
        else:
            context_parts.append("No numeric columns found.")

        context_parts.append("\n=== CATEGORICAL COLUMNS (unique values) ===")
        string_cols = df.select(pl.col(pl.Utf8)).columns
        for col in string_cols:
            unique_vals = df[col].unique().to_list()
            if len(unique_vals) <= 10:
                context_parts.append(f"  - {col}: {unique_vals}")
            else:
                context_parts.append(f"  - {col}: {unique_vals[:10]} ... ({len(unique_vals)} unique values)")

        return "\n".join(context_parts)

    except Exception as e:
        return f"Error reading CSV: {str(e)}\n{traceback.format_exc()}"



_RUNNER = r"""
import json, sys, traceback
from pathlib import Path
import polars as pl

payload = json.load(sys.stdin)
namespace = {
    "pl": pl,
    "polars": pl,
    "Path": Path,
    "csv_path": payload["csv_path"],
    "__name__": "__main__",
}
try:
    exec(compile(payload["code"], "<generated_code>", "exec"), namespace)
except Exception:
    traceback.print_exc()
    sys.exit(1)
"""

_CPU_SECONDS = 15          # hard cap on CPU time
_FSIZE_BYTES = 64 * 1024 * 1024  # max bytes the child may write to any file
_WALL_TIMEOUT = 20         # wall-clock timeout; kills hangs / sleeps


def _apply_limits() -> None:
    """preexec_fn for the child: cap CPU time and file size.

    Note: RLIMIT_AS / RLIMIT_DATA are deliberately NOT set here. Polars and its
    pyarrow backend reserve large *virtual* address space up front, so a memory
    cap causes spurious allocation failures on legitimate queries. The wall-clock
    timeout is the primary guard against resource exhaustion instead.
    """
    resource.setrlimit(resource.RLIMIT_CPU, (_CPU_SECONDS, _CPU_SECONDS))
    resource.setrlimit(resource.RLIMIT_FSIZE, (_FSIZE_BYTES, _FSIZE_BYTES))


def run_python_code(code: str, csv_path: Optional[str] = None) -> str:
    """
    Execute Polars/Python code in an isolated subprocess and return its output.

    The code runs in a fresh `python -I` interpreter with `pl` (polars) and
    `csv_path` predefined. Anything the code `print()`s is returned.

    Args:
        code: Python code to execute. Should use Polars (imported as pl).
        csv_path: Path to the CSV file to be used in the code.

    Returns:
        String containing the execution stdout or an error message.
    """
    file_path = Path(csv_path) if csv_path else DEFAULT_CSV
    payload = json.dumps({"code": code, "csv_path": str(file_path)})

    preexec = _apply_limits if os.name == "posix" else None

    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", _RUNNER],
            input=payload,
            capture_output=True,
            text=True,
            timeout=_WALL_TIMEOUT,
            preexec_fn=preexec,
        )
    except subprocess.TimeoutExpired:
        return (
            f"Error: code execution exceeded the {_WALL_TIMEOUT}s time limit "
            "and was terminated. Simplify the query or avoid long loops."
        )
    except Exception as e:  # pragma: no cover - spawn failures
        return f"Error launching execution subprocess:\n{e}\n{traceback.format_exc()}"

    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()

    if proc.returncode != 0:
        detail = stderr or stdout or f"process exited with code {proc.returncode}"
        return f"Error executing code:\n{detail}"

    result = stdout
    if stderr:
        result += f"\n\nStderr:\n{stderr}" if result else f"Stderr:\n{stderr}"
    if not result:
        result = "Code executed successfully (no output produced). Remember to print() results."
    return result


def create_polars_tools():
    """
    Create the Polars data-analytics tools as Haystack ``Tool`` objects.

    Uses the ``@tool`` decorator, which infers the tool's name, description and
    JSON schema from the function's docstring and ``Annotated`` type hints. The
    returned list can be passed straight to ``haystack.components.agents.Agent``.

    A single tool: the agent discovers the schema by running
    ``collect_schema()`` inside the executor, so no separate context tool is
    needed. (``get_csv_context`` remains as a plain helper for the Streamlit UI.)
    """
    from haystack.tools import tool

    @tool
    def execute_polars_code(
        code: Annotated[
            str,
            "Python code to execute using Polars. `pl` and `csv_path` are predefined; "
            "use print() to output results.",
        ],
        csv_path: Annotated[
            str, "Path to the CSV file. Leave empty to use the default sales data."
        ] = "",
    ) -> str:
        """
        Execute Python code using Polars for data analysis in an isolated subprocess.

        The code has access to `pl` (polars) and `csv_path` (path to the CSV file).
        Use Polars (NOT pandas) and always print() results, e.g.:
            df = pl.read_csv(csv_path)
            print(df.group_by("category").agg(pl.col("total_revenue").sum()))
        """
        return run_python_code(code, csv_path if csv_path else None)

    return [execute_polars_code]


if __name__ == "__main__":
    # Test the tools
    print("Testing get_csv_context:")
    print(get_csv_context())
    print("\n" + "="*50 + "\n")

    print("Testing run_python_code (isolated subprocess):")
    test_code = """
df = pl.read_csv(csv_path)
print("Total revenue by category:")
result = df.group_by("category").agg(pl.col("total_revenue").sum().alias("total"))
print(result)
"""
    print(run_python_code(test_code))
