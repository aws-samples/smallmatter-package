"""This module provides ambidextrous path classes that deal with local vs S3 paths.

Any operation to S3 is done via s3fs, hence this module shares the same scope and limitations as s3fs.

This project is meant for convenience, rather than feature completeness. Users are assume to be fully aware that only a
subset of filesystem operations are supported on S3.

This module aims to support only the most frequent pattern, i.e., read/write from a single file. By design, it
deliberately makes no attempt to emulate unsupported operations or to provide a unified abstraction on local-and-s3
filesystem. In fact, exceptions should be expected on some operations.

20200512:
- Placeholders for file operations, subject to s3fs's support.
"""

from os import PathLike
from pathlib import Path, _PathParents, _PosixFlavour  # type: ignore
from typing import Union
from urllib.parse import quote_from_bytes as urlquote_from_bytes

import s3fs


def is_s3(s: str) -> bool:
    """Check whether a string represents an S3 path."""
    return s[:5] == "s3://"


def is_s3_schema(s: str) -> bool:
    """Check whether a string represents an S3 schema."""
    return s == "s3://"


class _S3Flavour(_PosixFlavour):
    sep = "/"
    altsep = ""
    has_drv = False
    is_supported = True

    def parse_parts(self, parts):
        """Make sure to parts do not start with '/'.

        Internally, will use relative representation, but make sure is_absolute() always returns True.
        """
        _, _, parsed = super().parse_parts(parts)
        if len(parsed) > 0 and parsed[0] == "/":
            parsed = parsed[1:]
        return "", "", parsed

    def make_uri(self, path: "S3Path"):
        path_noschema: str = str(path)[len("s3://") :]  # Need to chop-off s3://
        bpath: bytes = path_noschema.encode("utf-8")
        return "s3://" + urlquote_from_bytes(bpath)  # type: ignore


_s3_flavour = _S3Flavour()

# Default S3FileSystem. Caller can set this to another S3FileSystem instance.
fs = s3fs.S3FileSystem(anon=False)


class Path2(Path):
    """An ambidextrous container of either a local path or an S3 path."""

    def __new__(cls, *args, fs=fs):
        """Create a new instance of ``Path2``."""
        # Override

        # Figure out which class to use
        if cls is Path2:
            if len(args) > 0 and is_s3(str(args[0])):
                cls = S3Path
                # Discard s3:// from parts
                args = args[1:] if is_s3_schema((args[0])) else (args[0][5:], *args[1:])
            else:
                # Defer local files to parent.
                return super().__new__(Path, *args)

        # Instantiate the object
        self = cls._from_parts(args)
        if self._flavour == _s3_flavour:
            if not fs:
                raise ValueError("S3FileSystem not defined")
            self._accessor = fs  # NOTE: deviate from pathlib, where isinstance(fs, Accessor) is False.

        return self


def pathify(path: Union[str, Path, PathLike], *args, **kwargs) -> Path2:
    """Ensure that path object will end-up as ``Path2``, if it hasn't been so."""
    return path if isinstance(path, Path2) else Path2(path, *args, **kwargs)


class S3Path(Path2):
    """A path to S3 object."""

    _flavour = _s3_flavour
    __slots__ = ()

    def _make_child(self, args):
        new_path = super()._make_child(args)
        new_path._accessor = self._accessor
        return new_path

    def is_absolute(self):
        """Return True always."""
        return True

    def _try_raise_closed(self):
        # Maintain consistent behavior with pathlib.Path, but only for python<3.9.
        # See https://bugs.python.org/issue39682 -- python 3.9 removed self._closed.
        try:
            if self._closed:
                self._raise_closed()
        except Exception:
            pass

    def open(self, *args, **kwargs):
        """[summary].

        args & kwargs: for S3FileSystem.open()
        """
        self._try_raise_closed()
        return self._accessor.open(self.__str__(), *args, **kwargs)

    def chmod(self, *args, **kwargs):
        self._try_raise_closed()
        raise NotImplementedError()

    def mkdir(self, *args, **kwargs):
        self._try_raise_closed()
        raise NotImplementedError()

    def read_bytes(self, *args, **kwargs):
        """Read bytes from file."""
        raise NotImplementedError()

    def read_text(self, *args, **kwargs):
        """Read strings from file."""
        raise NotImplementedError()

    def touch(self, *args, **kwargs):
        """Create a 0-byte file."""
        raise NotImplementedError()

    def unlink(self, *args, **kwargs):
        """Remove this file."""
        self._try_raise_closed()
        raise NotImplementedError()

    def rmdir(self, *args, **kwargs):
        """Remove this directory (should it be empty? Check s3fs/core.py)."""
        self._try_raise_closed()
        raise NotImplementedError()

    def exists(self, *args, **kwargs):
        """Check whether this path exists."""
        raise NotImplementedError()

    def write_bytes(self, *args, **kwargs):
        """Write bytes to file."""
        raise NotImplementedError()

    def write_text(self, *args, **kwargs):
        """Write strings to file."""
        raise NotImplementedError()

    def iterdir(self):
        """Iterate over the files in this directory.

        Does not yield any result for the special paths '.' and '..'.
        """
        self._try_raise_closed()
        raise NotImplementedError()

    def glob(self, pattern):
        """Yield all children (including directories) matching the given relative pattern."""
        raise NotImplementedError()

    def __str__(self):
        """Return the string representation of the path with the s3:// schema.

        Equivalent to as_uri().
        """
        try:
            return self._str
        except AttributeError:
            self._str = "s3://" + super()._format_parsed_parts("", "", self._parts) or "."
            return self._str

    def __rtruediv__(self, key):
        # super() ends-up calling PurePath._parse_args() which works directly on
        # parts: List[str], hence we must chop-off the s3:// for correct result.
        if not isinstance(key, Path) and key.startswith("s3://"):
            key = key[len("s3://") :]
        return super().__rtruediv__(key)

    @property
    def parent(self):
        # super() ends-up returning self or PurePath._from_parsed_parts(), where
        # both does not have _accessor information. Hence, propagate _accessor here.
        new_path = super().parent
        new_path._accessor = self._accessor
        return new_path

    @property
    def parents(self):
        """Return a sequence of this path's logical parents."""
        return _S3PathParents(self, self._accessor)


class _S3PathParents(_PathParents):
    def __init__(self, path, fs):
        super().__init__(path)
        self._accessor = fs

    def __len__(self):
        return len(self._parts)

    def __getitem__(self, idx):
        # Unlike the super() version, we don't want to return the parent of the
        # first path, which is 's3://' because s3 object ops always need bucket.
        if idx < 0 or idx >= len(self) - 1:
            raise IndexError(idx)
        parent = super().__getitem__(idx)
        parent._accessor = self._accessor
        return parent
