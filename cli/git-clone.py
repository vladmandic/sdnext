#!/usr/bin/env python
import os
import logging
import git
from rich import console, progress


class GitRemoteProgress(git.RemoteProgress):
    OP_CODES = ["BEGIN", "CHECKING_OUT", "COMPRESSING", "COUNTING", "END", "FINDING_SOURCES", "RECEIVING", "RESOLVING", "WRITING"]
    OP_CODE_MAP = { getattr(git.RemoteProgress, _op_code): _op_code for _op_code in OP_CODES }

    def __init__(self, url, folder) -> None:
        super().__init__()
        self.url = url
        self.folder = folder
        self.progressbar = progress.Progress(
            progress.SpinnerColumn(),
            progress.TextColumn("[cyan][progress.description]{task.description}"),
            progress.BarColumn(),
            progress.TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            progress.TimeRemainingColumn(),
            progress.TextColumn("[yellow]<{task.fields[url]}>"),
            progress.TextColumn("{task.fields[message]}"),
            console=console.Console(),
            transient=False,
        )
        self.progressbar.start()
        self.active_task = None

    def __del__(self) -> None:
        self.progressbar.stop()

    @classmethod
    def get_curr_op(cls, op_code: int) -> str:
        op_code_masked = op_code & cls.OP_MASK
        return cls.OP_CODE_MAP.get(op_code_masked, "?").title()

    def update(self, op_code: int, cur_count: str | float, max_count: str | float | None = None, message: str | None = "") -> None:
        if op_code & self.BEGIN:
            self.curr_op = self.get_curr_op(op_code) # pylint: disable=attribute-defined-outside-init
            self.active_task = self.progressbar.add_task(description=self.curr_op, total=max_count, message=message, url=self.url)
        self.progressbar.update(task_id=self.active_task, completed=cur_count, message=message)
        if op_code & self.END:
            self.progressbar.update(task_id=self.active_task, message=f"[bright_black]{message}")


def clone(url: str, folder: str):
    git.Repo.clone_from(
        url=url,
        to_path=folder,
        progress=GitRemoteProgress(url=url, folder=folder),
        multi_options=['--config core.compression=0', '--config core.loosecompression=0', '--config pack.window=0'],
        allow_unsafe_options=True,
        depth=1,
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description = 'downloader')
    parser.add_argument('--url', required=True, help="download url, required")
    parser.add_argument('--folder', required=False, help="output folder, default: autodetect")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    log = logging.getLogger(__name__)
    try:
        if not args.url.startswith('http'):
            raise ValueError(f'invalid url: {args.url}')
        f = args.url.split('/')[-1].split('.')[0] if args.folder is None else args.folder
        if os.path.exists(f):
            raise FileExistsError(f'folder already exists: {f}')
        log.info(f'Clone start: url={args.url} folder={f}')
        clone(url=args.url, folder=f)
        log.info(f'Clone complete: url={args.url} folder={f}')
    except KeyboardInterrupt:
        log.warning(f'Clone cancelled: url={args.url} folder={f}')
    except Exception as e:
        log.error(f'Clone: url={args.url} {e}')
