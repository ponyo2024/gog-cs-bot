"""Entrypoint wrapper for files/scraper_gamesguideinfo.py."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    target = Path(__file__).parent / "files" / "scraper_gamesguideinfo.py"
    runpy.run_path(str(target), run_name="__main__")
