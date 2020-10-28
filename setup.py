# noqa: D100
import os
from typing import List

from setuptools import find_packages, setup

import versioneer

_pkg: str = "smallmatter"
_ghrepo: str = f"{_pkg}-package"


def read(fname):
    """Read content on file name."""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


# Declare minimal set for installation
required_packages: List[str] = ["s3fs"]

# Specific use case dependencies
extras = {
    "sm": ["sagemaker"],
    "ds": ["pandas", "matplotlib"],
    "lambda.s3": ["boto3-stubs[s3]"],
}
all_deps = required_packages
for extra in extras.values():
    all_deps.extend(extra)
extras["all"] = all_deps

setup(
    name=_pkg,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Collection of fine-grained utilities for data science works on AWS.",
    long_description=read("README.md"),
    author="Verdi March",
    author_email="first.last@email.com",
    url=f"https://github.com/aws-samples/{_ghrepo}/",
    download_url="",
    project_urls={
        "Bug Tracker": f"https://github.com/aws-samples/{_ghrepo}/issues/",
        "Source Code": f"https://github.com/aws-samples/{_ghrepo}/",
    },
    license="MIT",
    keywords="data science AWS Amazon SageMaker S3 pandas matplotlib",
    platforms=["any"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.6.0",
    install_requires=required_packages,
    extras_require=extras,
    include_package_data=True,
)
