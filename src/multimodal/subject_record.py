from dataclasses import dataclass
from pathlib import Path


@dataclass
class SubjectRecord:
    id:    str
    sofa:  Path
    img_L: Path
    img_R: Path

    def __repr__(self) -> str:
        return f"SubjectRecord(id={self.id!r}, sofa={self.sofa.name!r})"
