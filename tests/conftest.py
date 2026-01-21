from pathlib import Path
import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def data_dir(project_root: Path) -> Path:
    return project_root / "data"
