"""Generate llms.txt — structured metadata for agent consumption.

Follows the llms.txt convention: machine-readable key-value pairs
describing the project, stack, and cognitive memory hints.

Regenerated whenever the profile is rebuilt via `cmm sync`.
"""
from __future__ import annotations

from pathlib import Path

from ..schemas.memory import CognitiveProfile


def generate_llms_txt(
    project_name: str,
    project_description: str,
    profile: CognitiveProfile | None,
    project_dir: Path | None = None,
) -> str:
    """Generate llms.txt content from a cognitive profile.

    Returns the full text content ready to write to .cognitive/llms.txt.
    """
    lines = [
        f"# {project_name}",
        "",
        f"> {project_description}" if project_description else "> No description available",
        "",
        "## Project Metadata",
        "",
    ]

    if profile:
        lines.append(f"- sessions_analyzed: {profile.session_count}")
        lines.append(f"- last_updated: {profile.last_updated.strftime('%Y-%m-%d')}")
        lines.append(f"- insights_count: {len(profile.architectural_insights)}")
        lines.append(f"- pitfalls_count: {len(profile.pitfalls)}")
        lines.append(f"- strategies_count: {len(profile.diagnostic_strategies)}")
    else:
        lines.append("- sessions_analyzed: 0")
        lines.append("- status: no profile built yet")

    # Stack detection from project files
    if project_dir:
        stack = _detect_stack(project_dir)
        if stack:
            lines.append("")
            lines.append("## Stack")
            lines.append("")
            for item in stack:
                lines.append(f"- {item}")

    # Pitfalls summary
    if profile and profile.pitfalls:
        lines.append("")
        lines.append("## Known Pitfalls")
        lines.append("")
        for p in profile.pitfalls:
            sev = p.severity.upper()
            lines.append(f"- [{sev}] {p.description}")
            if p.resolution_strategy:
                lines.append(f"  Resolution: {p.resolution_strategy}")

    # Architectural insights
    if profile and profile.architectural_insights:
        lines.append("")
        lines.append("## Architectural Insights")
        lines.append("")
        for ins in profile.architectural_insights:
            lines.append(f"- [{ins.component}] {ins.insight}")

    # Diagnostic strategies
    if profile and profile.diagnostic_strategies:
        lines.append("")
        lines.append("## Diagnostic Strategies")
        lines.append("")
        for s in profile.diagnostic_strategies:
            lines.append(f"- Trigger: {s.trigger}")
            for step in s.steps:
                lines.append(f"  - {step}")

    # Key patterns
    if profile and profile.key_patterns:
        lines.append("")
        lines.append("## Key Patterns")
        lines.append("")
        for p in profile.key_patterns:
            lines.append(f"- {p}")

    # Anti-patterns
    if profile and profile.anti_patterns:
        lines.append("")
        lines.append("## Anti-Patterns")
        lines.append("")
        for p in profile.anti_patterns:
            lines.append(f"- {p}")

    # Retrieval hints for agents
    lines.append("")
    lines.append("## Retrieval Hints")
    lines.append("")
    lines.append("- Use `/search-memory <query>` to find relevant past reasoning")
    lines.append("- Use `/diagnose <problem>` to find proven debugging strategies")
    lines.append("- Read .cognitive/cached_profile.md for full architectural context")
    lines.append("- Memory covers ALL past sessions, not just the most recent one")

    lines.append("")
    return "\n".join(lines)


def _detect_stack(project_dir: Path) -> list[str]:
    """Detect the project's tech stack from common config files."""
    stack = []

    indicators = {
        "pyproject.toml": "Python",
        "setup.py": "Python",
        "requirements.txt": "Python",
        "package.json": "Node.js/JavaScript",
        "tsconfig.json": "TypeScript",
        "Cargo.toml": "Rust",
        "go.mod": "Go",
        "pom.xml": "Java (Maven)",
        "build.gradle": "Java (Gradle)",
        "Gemfile": "Ruby",
        "mix.exs": "Elixir",
        "Dockerfile": "Docker",
        "docker-compose.yml": "Docker Compose",
        "docker-compose.yaml": "Docker Compose",
        ".github/workflows": "GitHub Actions",
    }

    for file_or_dir, tech in indicators.items():
        if (project_dir / file_or_dir).exists():
            if tech not in stack:
                stack.append(tech)

    # Check for frameworks in Python projects
    pyproject = project_dir / "pyproject.toml"
    if pyproject.exists():
        try:
            content = pyproject.read_text()
            if "fastapi" in content.lower():
                stack.append("FastAPI")
            if "django" in content.lower():
                stack.append("Django")
            if "flask" in content.lower():
                stack.append("Flask")
            if "pytest" in content.lower():
                stack.append("pytest")
        except OSError:
            pass

    # Check for frameworks in JS projects
    pkg_json = project_dir / "package.json"
    if pkg_json.exists():
        try:
            content = pkg_json.read_text()
            if "react" in content.lower():
                stack.append("React")
            if "next" in content.lower():
                stack.append("Next.js")
            if "vue" in content.lower():
                stack.append("Vue.js")
        except OSError:
            pass

    return stack
