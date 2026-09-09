from __future__ import annotations


def verify_artifacts(task_text: str, artifact_report: str) -> dict:
    """Lightweight specification-derived verifier.

    This intentionally avoids hidden checks. It only detects generic issues.
    """
    failures = []
    lower = artifact_report.lower()

    if "no_artifacts_found" in lower:
        failures.append("no output artifacts produced")
    if "inspection_error" in lower:
        failures.append("artifact inspection failed")
    if "nan" in lower or "inf" in lower:
        failures.append("possible invalid numerical values")

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "artifact_report": artifact_report,
    }
