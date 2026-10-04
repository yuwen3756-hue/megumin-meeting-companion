"""One application per data directory; Windows releases the lock on a crash."""
from pathlib import Path


class InstanceLock:
    def __init__(self, data_dir: Path):
        import msvcrt
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        self.file = (Path(data_dir) / "app.lock").open("a+b")
        if self.file.tell() == 0:
            self.file.write(b"0")
            self.file.flush()
        self.file.seek(0)
        try:
            msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            self.file.close()
            raise RuntimeError("MeguminMeet is already open for this data folder. Use the existing window.") from exc

    def close(self):
        if not self.file.closed:
            import msvcrt
            self.file.seek(0)
            msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
            self.file.close()
