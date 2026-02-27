from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List


def _natural_key(name: str) -> list[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", name)
    ]


class ImageSession:
    def __init__(self, *, suffixes: Iterable[str]) -> None:
        self._suffixes = {str(s).lower() for s in suffixes}
        self.paths: List[Path] = []
        self.index = -1

    @property
    def total(self) -> int:
        return len(self.paths)

    @property
    def has_prev(self) -> bool:
        return self.total > 1 and self.index > 0

    @property
    def has_next(self) -> bool:
        return self.total > 1 and self.index >= 0 and self.index < self.total - 1

    @property
    def current_path(self) -> Path | None:
        if self.index < 0 or self.index >= self.total:
            return None
        return self.paths[self.index]

    def build_around(self, selected_path: Path) -> None:
        folder = selected_path.parent
        candidates = [
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in self._suffixes
        ]
        candidates.sort(key=lambda p: _natural_key(p.name))
        self.paths = candidates
        self.index = -1

        selected_abs = str(selected_path.resolve())
        for i, p in enumerate(self.paths):
            if str(p.resolve()) == selected_abs:
                self.index = i
                break

        if self.index >= 0:
            return

        self.paths.append(selected_path)
        self.paths.sort(key=lambda p: _natural_key(p.name))
        self.index = next(
            i for i, p in enumerate(self.paths) if str(p.resolve()) == selected_abs
        )

    def can_index(self, index: int) -> bool:
        return 0 <= index < self.total

    def set_index(self, index: int) -> bool:
        if not self.can_index(index):
            return False
        self.index = int(index)
        return True

    def restore_index(self, index: int) -> None:
        if self.can_index(index):
            self.index = int(index)
            return
        self.index = -1
