# from pathlib import Path

from typing import Generator

import s3fs

from smallmatter.pathlib import Path2 as Path
from smallmatter.pathlib import S3Path


def test_glob():
    """
    Test data directory structure:

    data
    |-- 1.txt
    |-- 2.txt
    `-- folder
        `-- 3.txt

    """
    # print("\n\n")
    # print("*" * 80)
    # path_local = Path("data/")
    # files_local = list(path_local.glob("*"))
    # print(files_local)

    path_s3 = Path("s3://wy-test-sg/data/")
    for pattern in ["*.txt", "*"]:
        gen = path_s3.glob(pattern)
        assert isinstance(gen, Generator)
        file_s3 = list(gen)
        assert isinstance(file_s3[0], S3Path)
        print(file_s3)

        fs = s3fs.S3FileSystem(anon=False)
        s3fs_list = fs.glob(str(path_s3 / pattern))
        print(s3fs_list)
        assert len(s3fs_list) == len(file_s3)
        print("\n")


test_glob()
