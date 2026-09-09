from __future__ import annotations

import argparse
import os
import pathlib
import re
import shutil
import subprocess
import sys

from openai import OpenAI
from profiler import profile_inputs


RESERVED_OUTPUTS = {"reward.json", "pytest_report.json", "reward.txt"}


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:python|py)?\s*\n?(.*?)```", text, re.DOTALL | re.I)
    if m:
        return m.group(1).strip()
    return text


def _expected_output_names(instruction: str) -> list[str]:
    names = re.findall(r"(?:/app)?/output/([A-Za-z0-9_.-]+)", instruction)
    return sorted({x for x in names if x not in RESERVED_OUTPUTS})


def _run_solution(script: pathlib.Path, output_dir: pathlib.Path):
    timeout = int(os.environ.get("AGENT_SOLUTION_TIMEOUT_SEC", "1500"))
    try:
        r = subprocess.run(
            [sys.executable, str(script)],
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return r.returncode, r.stdout[-4000:], r.stderr[-8000:]
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout}s"


def solve(task_dir: str, out_dir: str):
    task = pathlib.Path(task_dir).resolve()
    output = pathlib.Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    instruction = (task / "instruction.md").read_text(encoding="utf-8")
    data_profile = profile_inputs(task)

    endpoint = os.environ.get("MODEL_ENDPOINT")
    model = os.environ.get("MODEL_NAME")
    if not endpoint or not model:
        raise RuntimeError("MODEL_ENDPOINT and MODEL_NAME must be set")

    client = OpenAI(base_url=endpoint, api_key=os.environ.get("MODEL_API_KEY", "unused"))

    base_prompt = f"""
You are a quantitative finance coding agent.

You must produce a complete executable Python program.
Use the observed input profile below. Never guess columns or keys.
Do not access checks, card.toml, manifest.json, or hidden verifier data.
Do not create reward.json, pytest_report.json or reward.txt.
Only write requested deliverables.

INPUT PROFILE:
{data_profile}

TASK:
{instruction}

Return only Python source code.
"""

    expected = _expected_output_names(instruction)
    script = pathlib.Path("/tmp/qf_agent_solution.py")
    max_repairs = int(os.environ.get("AGENT_MAX_REPAIRS", "2"))
    code = ""
    last_error = ""

    for attempt in range(max_repairs + 1):
        if attempt == 0:
            messages = [
                {"role": "system", "content": "Write robust executable Python."},
                {"role": "user", "content": base_prompt},
            ]
        else:
            messages = [
                {"role": "system", "content": "Repair the program with minimum necessary changes."},
                {"role": "user", "content": base_prompt + "\nPrevious code:\n" + code + "\nExecution feedback:\n" + last_error},
            ]

        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=int(os.environ.get("MODEL_MAX_TOKENS", "12000")),
            messages=messages,
        )
        code = _strip_code_fence(resp.choices[0].message.content or "")
        script.write_text(code + "\n", encoding="utf-8")

        for name in expected:
            p = output / name
            if p.exists():
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()

        rc, stdout, stderr = _run_solution(script, output)
        missing = [x for x in expected if not (output / x).is_file()]

        if rc == 0 and not missing:
            return

        last_error = (
            f"return_code={rc}\n"
            f"missing={missing}\n"
            f"stdout={stdout}\n"
            f"stderr={stderr}"
        )

    print(last_error, file=sys.stderr)
    raise RuntimeError("agent failed after repairs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("verb")
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.verb != "solve":
        raise SystemExit("only solve supported")
    solve(args.task_dir, args.out)


if __name__ == "__main__":
    main()
