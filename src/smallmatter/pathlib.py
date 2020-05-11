"""This module provides ambidextrous path classes that deal with local vs S3 paths."""

from os import PathLike
from pathlib import Path, _PosixFlavour  # type: ignore
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
        """Make sure to parts are absolute."""
        drv, root, parsed = super().parse_parts(parts)
        if parsed[0] != "/":
            parsed.insert(0, "/")
        return drv, "s3://", parsed

    def make_uri(self, path: "S3Path"):
        path_noschema: str = str(path)[len("s3://") :]  # Need to chop-off s3://
        bpath: bytes = path_noschema.encode("utf-8")
        return path._root + urlquote_from_bytes(bpath)  # type: ignore


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
            self._fs = fs

        return self


def pathify(path: "PathLike", *args, **kwargs) -> Path2:
    """Ensure that path object will end-up as ``Path2``, if it hasn't been so."""
    return path if isinstance(path, Path2) else Path2(path, *args, **kwargs)


class S3Path(Path2):
    """A path to S3 object."""

    _flavour = _s3_flavour
    __slots__ = ()

    def _init(self):
        # Override PurePath._init().
        #
        # Called by _from_parts() and _from_parsed_parts() to initialize a new
        # instance of S3Path.
        pass
        # FIXME: propagate _fs

    def open(self, *args, **kwargs):
        """[summary].

        args & kwargs: for S3FileSystem.open()
        """
        return self._fs.open(self.__str__(), *args, **kwargs)

    def iterdir(self):
        """Iterate over the files in this directory.

        Does not yield any result for the special paths '.' and '..'.
        """
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
            self._str = super()._format_parsed_parts("", self._root, self._parts) or "."
            return self._str

    # Override a few pathlib's, to propagate _fs to the new S3Path constructed from me.
    # FIXME: all tehse should be handled inside _init?? OTherwise there's simply too many to override :(
    def __truediv__(self, key):
        new_path = super().__truediv__(key)
        new_path._fs = self._fs
        return new_path

    def __rtruediv__(self, key):
        new_path = super().__rtruediv__(key)
        new_path._fs = self._fs
        return new_path

    @property
    def parent(self):
        new_path = super().parent
        new_path._fs = self._fs
        return new_path
