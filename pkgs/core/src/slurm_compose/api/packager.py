from pathlib import Path

from pathspec import PathSpec


def gitignore_filter(
    root: str | Path, ignore_files: list[str | Path] | None = None, ignore_patterns: list[str] | None = None
):
    root = Path(root).resolve()
    ignore_files = [Path(ifile) for ifile in (ignore_files or [])]
    ignore_patterns = ignore_patterns or []

    ignore_lines = sum([f.read_text().splitlines() for f in ignore_files if f.exists()], []) + ignore_patterns
    spec = PathSpec.from_lines("gitignore", ignore_lines)

    def ignore(dirpath, names):
        d = Path(dirpath).resolve()
        skipped = []
        for name in names:
            rel = (d / name).relative_to(root).as_posix()
            if (d / name).is_dir():
                rel += "/"
            if spec.match_file(rel):
                skipped.append(name)
        return skipped

    return ignore
