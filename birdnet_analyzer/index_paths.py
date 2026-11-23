"""Lets an embeddings database lie below any folder on Windows.

usearch hands the path of its index file to the C runtime as UTF-8 bytes, which
Windows reads in its own code page: below a folder with a character outside ASCII
(``C:\\Users\\Jörg``) it finds no such file, and the database can neither be created nor
opened. For such a path the index is read and written by Python instead, through
usearch's buffer functions. Every other path keeps usearch's own file handling.
"""

from __future__ import annotations

import mmap
import os
import sys

_MAPPED = "_birdnet_mapped_file"


def _needs_python_io(path) -> bool:
    if sys.platform != "win32" or path is None:
        return False

    try:
        return not os.fspath(path).isascii()
    except TypeError:
        # A buffer, not a path.
        return False


def _release(index) -> None:
    mapped = index.__dict__.pop(_MAPPED, None)

    if mapped is not None:
        mapped.close()


def _write(path, data) -> None:
    partial = f"{os.fspath(path)}.{os.getpid()}.tmp"

    try:
        with open(partial, "wb") as file:
            file.write(data)

        os.replace(partial, path)
    finally:
        if os.path.exists(partial):
            os.remove(partial)


def reach_any_path() -> None:
    """Makes ``usearch.index.Index`` save, load and view below any path.

    Call it before an index is created. Calling it again changes nothing.
    """
    from usearch.index import Index

    if getattr(Index, "_birdnet_reaches_any_path", False):
        return

    save, load, view, reset = Index.save, Index.load, Index.view, Index.reset

    def save_anywhere(self, path_or_buffer=None, progress=None):
        target = path_or_buffer if path_or_buffer is not None else self.path

        if not _needs_python_io(target):
            return save(self, path_or_buffer, progress)

        _write(target, self._compiled.save_index_to_buffer(progress))

        return None

    def load_anywhere(self, path_or_buffer=None, progress=None):
        source = path_or_buffer if path_or_buffer is not None else self.path

        if not _needs_python_io(source):
            return load(self, path_or_buffer, progress)

        with open(source, "rb") as file:
            self._compiled.load_index_from_buffer(file.read(), progress)

        # Windows replaces no file that is still mapped, and a loaded index is saved
        # over the file it was viewed from.
        _release(self)

        return None

    def view_anywhere(self, path_or_buffer=None, progress=None):
        source = path_or_buffer if path_or_buffer is not None else self.path

        if not _needs_python_io(source):
            return view(self, path_or_buffer, progress)

        with open(source, "rb") as file:
            mapped = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)

        self._compiled.view_index_from_buffer(mapped, progress)
        _release(self)
        # The index reads from the mapping for as long as it views the file.
        setattr(self, _MAPPED, mapped)

        return None

    def reset_anywhere(self):
        reset(self)
        _release(self)

    Index.save = save_anywhere
    Index.load = load_anywhere
    Index.view = view_anywhere
    Index.reset = reset_anywhere
    Index._birdnet_reaches_any_path = True
