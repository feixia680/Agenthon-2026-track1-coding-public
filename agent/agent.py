from __future__ import annotations

import argparse
import os
import pathlib
import re
import shutil
import subprocess
import sys

from openai import OpenAI


def _input_files(task_dir: pathlib.Path) -> str:
    """Return a compact, deterministic inventory for the model prompt."""
    rows: list[str] = []
    for path in sorted(task_dir.rglob("*")):
        if (
            path.is_file()
            and ".git" not in path.parts
            and "checks" not in path.parts
            and path.name not in {"card.toml", "manifest.json"}
        ):
            try:
                size = path.stat().st_size
            except OSError:
                size = -1
            rows.append(f"- /input/{path.relative_to(task_dir)} ({size} bytes)")
    return "\n".join(rows) or "(no files found)"


def _expected_output_names(instruction: str) -> list[str]:
    """Find explicitly named output files without reading grader-owned checks."""
    names = re.findall(r"(?:/app)?/output/([A-Za-z0-9_.-]+)", instruction)
    output_section = re.search(
        r"(?:save|write).*?(?:outputs?|results?).*?(?:/output/|output directory).*?:?(.*)",
        instruction,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if output_section:
        names.extend(
            re.findall(
                r"`([A-Za-z0-9_.-]+\.(?:json|csv|parquet|png|txt|xlsx))`",
                output_section.group(1),
                flags=re.IGNORECASE,
            )
        )
    return sorted(set(names))


def _strip_code_fence(text: str) -> str:
    """Extract source from raw or fenced model output, including truncation."""
    text = text.strip()
    match = re.search(r"```(?:python|py)?\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"```(?:python|py)?\s*\n?(.*)$", text, re.DOTALL | re.IGNORECASE)
    if match:
        return re.sub(r"\n?```+\s*$", "", match.group(1)).strip()
    return text


def _run_solution(script: pathlib.Path, output_dir: pathlib.Path) -> int:
    timeout = int(os.environ.get("AGENT_SOLUTION_TIMEOUT_SEC", "1500"))
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=output_dir,
            check=False,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        print(f"solution timed out after {timeout}s", file=sys.stderr)
        return 124
    if result.stdout:
        print(result.stdout[-2000:])
    if result.stderr:
        print(result.stderr[-4000:], file=sys.stderr)
    return result.returncode


def _run_local_checks(task_path: pathlib.Path, output_path: pathlib.Path) -> tuple[bool, str]:
    """Run public checks only when explicitly enabled for local development."""
    if os.environ.get("AGENT_LOCAL_CHECKS") != "1":
        return True, ""
    checks = task_path / "checks" / "test_outputs.py"
    if not checks.is_file():
        return True, ""
    env = {**os.environ, "OUTPUT_DIR": str(output_path), "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(checks), "-q", "--no-header", "-p", "no:cacheprovider"],
            cwd=task_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=int(os.environ.get("AGENT_CHECK_TIMEOUT_SEC", "900")),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "local verifier timed out"
    report = (result.stdout or "") + (result.stderr or "")
    return result.returncode == 0, report[-12000:]


def solve(task_dir: str, out_dir: str) -> None:
    task_path = pathlib.Path(task_dir).resolve()
    output_path = pathlib.Path(out_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    instruction_path = task_path / "instruction.md"
    if not instruction_path.exists():
        raise FileNotFoundError(f"missing task instruction: {instruction_path}")
    instruction = instruction_path.read_text(encoding="utf-8")
    inventory = _input_files(task_path)

    endpoint = os.environ.get("MODEL_ENDPOINT")
    model = os.environ.get("MODEL_NAME")
    if not endpoint or not model:
        raise RuntimeError("MODEL_ENDPOINT and MODEL_NAME must be set")

    client = OpenAI(
        base_url=endpoint,
        api_key=os.environ.get("MODEL_API_KEY", "unused"),
    )
    prompt = f"""You are a quantitative-finance coding agent.

Read the task specification below and produce a complete executable Python program.
Before writing the final program, inspect the actual input files and their columns/keys;
never guess a column name or JSON key from the task title.
The program must read the task inputs and write exactly the deliverables requested by
the task. Do not hardcode expected answers. Use only local task files and the paths
specified by the task; do not use the network. Write deliverables under the requested
output directory. The current output directory is {output_path}.

The task files visible to the program are:
{inventory}

Serialization requirement: every JSON deliverable must be valid strict JSON. Convert NumPy
scalars and booleans to built-in Python float, int, or bool before json.dump, and never emit
NaN or Infinity.

Task specification:
---
{instruction}
---

Return only the Python source code, without Markdown fences or explanations.
"""

    request = {
        "model": model,
        "temperature": 0,
        "max_tokens": int(os.environ.get("MODEL_MAX_TOKENS", "8000")),
        "messages": [
            {
                "role": "system",
                "content": "Return robust, executable Python code for the task.",
            },
            {"role": "user", "content": prompt},
        ],
    }
    reasoning_effort = os.environ.get("MODEL_REASONING_EFFORT")
    if reasoning_effort:
        request["reasoning_effort"] = reasoning_effort
    script_path = output_path / "solution.py"
    expected_outputs = _expected_output_names(instruction)
    repairs_left = int(os.environ.get("AGENT_MAX_REPAIRS", "1"))
    code = ""
    return_code = 1
    last_error = ""
    for attempt in range(repairs_left + 1):
        if attempt == 0:
            response = client.chat.completions.create(**request)
        else:
            repair_request = {
                **request,
                "messages": [
                    request["messages"][0],
                    request["messages"][1],
                    {
                        "role": "user",
                        "content": (
                            "The previous program failed during execution or output checks. Re-read the "
                            "full task specification above, preserve every required output filename and "
                            "schema, then fix the program. Return only complete Python source.\n\n"
                            f"Previous source:\n{code}\n\nRuntime output:\n{last_error}"
                        ),
                    },
                ],
            }
            response = client.chat.completions.create(**repair_request)
        code = _strip_code_fence(response.choices[0].message.content or "")
        if not code.strip():
            raise RuntimeError("model returned empty code")
        script_path.write_text(code + "\n", encoding="utf-8")
        # A failed program can leave partial artifacts behind.  Repairs must be
        # judged only on files produced by their own execution.
        for name in expected_outputs:
            artifact = output_path / name
            if artifact.is_dir():
                shutil.rmtree(artifact)
            elif artifact.exists():
                artifact.unlink()
        return_code = _run_solution(script_path, output_path)
        missing = [name for name in expected_outputs if not (output_path / name).is_file()]
        checks_ok, checks_report = _run_local_checks(task_path, output_path)
        if return_code == 0 and not missing and checks_ok:
            return
        if return_code == 0 and missing:
            return_code = 125
            last_error = "program exited 0 but missing required output files: " + ", ".join(missing)
        elif return_code == 0 and not checks_ok:
            return_code = 125
            last_error = "local verifier failures:\n" + checks_report
        else:
            last_error = "solution exited with code " + str(return_code)
        if attempt < repairs_left:
            print("solution attempt failed; requesting a repair", file=sys.stderr)
    print(f"solution exited with code {return_code}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("verb")
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.verb != "solve":
        raise SystemExit("only the solve verb is supported")
    solve(args.task_dir, args.out)


if __name__ == "__main__":
    main()
