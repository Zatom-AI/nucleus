#!/usr/bin/env python3
"""Offline audit of public release files, inputs, and project-page resources."""

import ast
import json
import re
import subprocess
import sys
import tomllib
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def release_files():
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
    )
    return sorted({ROOT / p.decode() for p in output.split(b"\0") if p})


class Resources(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in {"src", "href", "data-zoom-src"} and value:
                self.paths.append(value)


def audit():
    errors = []
    files = release_files()
    for p in files:
        relative = p.relative_to(ROOT)
        if not p.is_file():
            continue
        if p.is_symlink():
            errors.append(f"Symlink in release: {relative}")
        if p.stat().st_size >= 100 * 1024 * 1024:
            errors.append(f"File exceeds GitHub limit: {relative}")
        if any(
            x in relative.parts
            for x in [
                "zatom2p",
                "zatom2n",
                "zatomfast",
                "perlmutter",
                ".env",
                "AGENTS.md",
                "_scratch",
            ]
        ):
            errors.append(f"Development-only file: {relative}")
        if p.suffix == ".py":
            try:
                ast.parse(p.read_text(), filename=str(relative))
            except SyntaxError as error:
                errors.append(str(error))
        if p.suffix in {
            ".py",
            ".yaml",
            ".toml",
            ".md",
            ".json",
            ".html",
            ".sh",
            ".tex",
            ".bib",
            ".csv",
            ".txt",
        }:
            text = p.read_text(errors="replace")
            if re.search(r"/(?:pscratch)/sd/|/(?:global)/(?:cfs|homes|u2)/", text):
                errors.append(f"Machine-specific path: {relative}")
            if re.search(r"\b(?:ghp_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{30,})\b", text):
                errors.append(f"Possible access token: {relative}")
    examples = ROOT / "models/zatom2/docs/examples"
    for p in examples.glob("*.json"):
        data = json.loads(p.read_text())
        if not isinstance(data, dict):
            continue
        for key, spec in data.items():
            if isinstance(spec, dict) and spec.get("input"):
                if not (p.parent / spec["input"]).is_file():
                    errors.append(f'Missing input: {p.name}:{key}: {spec["input"]}')
    page = ROOT / "models/zatom2/docs/index.html"
    parser = Resources()
    parser.feed(page.read_text())
    for value in parser.paths:
        url = urlsplit(value)
        if not url.scheme and not url.netloc and url.path:
            if not (page.parent / unquote(url.path)).is_file():
                errors.append(f"Missing page resource: {value}")
    for stylesheet in (page.parent / "static").rglob("*.css"):
        for value in re.findall(r"url\(([^)]+)\)", stylesheet.read_text()):
            url = urlsplit(value.strip("\"' "))
            if not url.scheme and not url.netloc and url.path:
                if not (stylesheet.parent / unquote(url.path)).is_file():
                    errors.append(
                        f"Missing stylesheet resource: {stylesheet.name}: {value}"
                    )
    for p in files:
        if p.suffix != ".md":
            continue
        for value in re.findall(r"\]\(([^)]+)\)", p.read_text()):
            value = value.split(' "', 1)[0]
            url = urlsplit(value)
            if url.scheme or url.netloc or not url.path or url.path.startswith("/"):
                continue
            if not (p.parent / unquote(url.path)).exists():
                errors.append(f"Broken document link: {p.relative_to(ROOT)} -> {value}")
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    if set(metadata["project"]["scripts"]) != {"nucleus", "zatom2", "zevals"}:
        errors.append("Unexpected public entry points")
    for source in metadata["tool"]["hatch"]["build"]["targets"]["wheel"][
        "force-include"
    ]:
        if not (ROOT / source).exists():
            errors.append(f"Missing packaged resource: {source}")
    return errors, len(files)


if __name__ == "__main__":
    errors, count = audit()
    if errors:
        print("\n".join(errors))
        sys.exit(1)
    print(
        f"Public release audit passed ({count} files; syntax, references, assets, packaging, paths)."
    )
