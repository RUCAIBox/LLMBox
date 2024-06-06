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


def update_script(load_script_path: Path, mirror: bool, old: str, new: str) -> bool:
    if not load_script_path.exists():
        return False

    # if HF_ENDPOINT is set in the environment, update the load script
    hf_endpoint = os.environ.get("HF_ENDPOINT", "").rstrip("/")
    new_endpoint = new.rstrip("/")

    if mirror or hf_endpoint == new_endpoint:
        script = load_script_path.read_text()
        updated_script = script.replace(old, new)
        if script != updated_script:
            load_script_path.with_suffix(".py.bak").write_text(script)
            load_script_path.write_text(updated_script)
            return True

    return False


def huggingface_download(
    path: str,
    hfd_cache_path: str,
    mirror: bool = True,
    old: str = "https://huggingface.co",
    new: str = "https://hf-mirror.com",
) -> Optional[str]:
    """Download a dataset from Hugging Face Hub to a local directory using hfd.sh."""

    hub_cache_path = Path(hfd_cache_path).expanduser()
    repo_name = "datasets--" + path.replace("/", "--")
    repo_path = hub_cache_path / repo_name
    load_script_path = get_script_path(repo_path)

    if repo_path.exists() and load_script_path.exists():
        update_script(load_script_path, mirror, old, new)
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

    update_script(load_script_path, mirror, old, new)

    return str(repo_path)
