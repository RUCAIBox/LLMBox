import os
from logging import getLogger
from pathlib import Path
from typing import Optional, Union

logger = getLogger(__name__)


def get_script_path(repo_path: Union[Path, str]) -> Path:
    if isinstance(repo_path, Path):
        repo_path = str(repo_path)

    load_script_name = repo_path.split("/")[-1].split("--")[-1] + ".py"
    return Path(repo_path) / load_script_name


def huggingface_download(
    path: str,
    mirror: bool = True,
    old: str = "https://huggingface.co",
    new: str = "https://hf-mirror.com",
) -> Optional[str]:
    """Download a dataset from Hugging Face Hub to a local directory using hfd.sh."""

    hub_cache_path = Path.home() / ".cache" / "huggingface" / "datasets"
    repo_name = "datasets--" + path.replace("/", "--")
    repo_path = hub_cache_path / repo_name
    load_script_path = get_script_path(repo_path)

    if repo_path.exists() and load_script_path.exists():
        logger.debug(f"Found {repo_path}, skipping download")
        return str(repo_path)

    if os.name != "posix":
        logger.warning("hfd.sh is only supported on Unix-like systems.")
        return None

    hfd_cli = Path(__file__).parent.parent / "hfd.sh"
    logger.debug(f"Downloading {path} to {repo_path}")

    mirror_flag = " --mirror" if mirror else ""
    command = f"bash {hfd_cli.as_posix()} {path} --dataset --local-dir {repo_path.as_posix()}"
    os.system(command + mirror_flag)

    if mirror:
        script = load_script_path.read_text().replace(old, new)
        load_script_path.write_text(script)

    return str(repo_path)
