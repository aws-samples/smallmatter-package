## smallmatter Python package

Wrappers to streamline data-science tasks using Python toolkits (such as pandas,
matplotlib, etc.).

Capabilities (non-exhaustive list):

- `smallmatter.pathlib.Path2`: `pathlib`-compatible interface to abstract
certain single-file operations on local filesystem and Amazon S3.

- `smallmatter.pathlib.S3Path`: `pathlib`-compatible interface to abstract
certain single-file operations on Amazon S3.

- `smallmatter.sm`: utilities for Amazon SageMaker
  * `get_sm_execution_role()`: an opinionated function to unify fetching the
    execution role inside and outside of a notebook instance.
  * `PyTestHelpers`: intended to run pytest tests on a git repo with multiple
    SageMaker sourcedirs.

- `smallmatter.ds`: for rapid prototyping of some data science tasks.
  * `PyExec`: typical use-case: to implement data dictionary or config files
    that can mix-in Python code to construct certain variables.
  * `DFBuilder`: a helper class to build a Pandas dataframe incrementally,
    row-by-row.
  * `json_np_converter`: convert numpy values to JSON-compliant data type.
  * `SimpleMatrixPlotter`: a simpler helper class to fill-in supplot one after
    another.
  * `MontagePager`: a pager to group and save subplots into multiple montage
    image files.

- `bin/pp.sh`: standard template to run `pandas-profiling`.


## Installation

```bash
pip install \
    'git+https://github.com/aws-samples/smallmatter-package@v0.1.0#egg=smallmatter'

# Install extras capability; need the single quotes.
pip install \
    'git+https://github.com/aws-samples/smallmatter-package@v0.1.0#egg=smallmatter[all]'
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
