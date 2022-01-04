# smallmatter Python package

Wrappers to streamline data-science tasks using Python toolkits (such as pandas,
matplotlib, etc.).

The intent is to improve the ergonomics of data science exploration and
implementation, by simplifying repetitive tasks such as checking data types
(i.e., data quality), procedural subplots, consistent interface to local
filesystem and Amazon S3, etc.

Capabilities (non-exhaustive list):

- `smallmatter.pathlib.Path2`: `pathlib`-compatible interface to abstract
certain single-file operations on local filesystem and Amazon S3.

- `smallmatter.pathlib.S3Path`: `pathlib`-compatible interface to abstract
certain single-file operations on Amazon S3.

- `smallmatter.sm`: utilities for Amazon SageMaker
  - `smallmatter.sm.FrameworkProcessor`: a prototype to support SageMaker
    Python processing jobs that accept multiple files. This is done through
    `source_dir`, `depedencies`, `requirements.txt`, and `git_config`, similar
    to SageMaker estimator APIs. Furthermore, this `FrameworkProcessor` supports
    the SageMaker managed framework containers (i.e., MXNet, PyTorch, TensorFlow,
    Scikit-learn, and XGBoost).

    It aims to give you the familiar workflow of (1) instantiate a processor, then
    immediately (2) call the `run(...)` method.

    Here's an [example](`https://github.com/aws-samples/smallmatter-package/blob/main/notebooks/smproc-stopgap/try-smproc-stopgap.py#L14-L53`)
    on how to use this `FrameworkProcessor` class -- right now, the example is
    a `.py` file as opposed to `.ipynb`. Run the Python example using
    [this shell script](https://github.com/aws-samples/smallmatter-package/blob/main/notebooks/smproc-stopgap/try-smproc-stopgap.sh).
    You need to update the shell script S3 prefix and SageMaker execution role,
    and optionally choose your preferred framework container.

    ```Python
    --s3-prefix <s3://bucket/prefix/sagemaker>
    --role <arn:aws:iam::111122223333:role/service-role/my-amazon-sagemaker-execution-role-1234>

    # Optional: Update with your preferred container, it is Pytorch here
    --framework_version 1.6.0
    sagemaker.pytorch.estimator.PyTorch
    ```

    It slightly changes the processing API by adding a SageMaker Framework
    estimator, which was done for two purposes: (1) auto-detect container uri,
    and (2) re-use the packaging mechanism in the estimator to upload to
    `s3://.../sourcedir.tar.gz`.

  - `PyTestHelpers`: intended to run pytest tests on a git repo with multiple
    SageMaker sourcedirs.
  - `get_sm_execution_role()`: an opinionated function to unify fetching the
    execution role inside and outside of a notebook instance.

- `smallmatter.typecheck`: check possible dtypes of a csv file.
  - For each column, list the auto-detected dtypes. Useful to check data
    for data qualities (e.g., mixed-in strings and numbers in the same column).
  - Generate html reports of the auto-detected dtypes.
  - CLI interface to generate those html reports.

- `smallmatter.ds`: for rapid prototyping of some data science tasks.
  - `SimpleMatrixPlotter`: a simpler helper class to fill-in subplots one after
    another.
  - `MontagePager`: a pager to group and save subplots into multiple montage
    image files.
  - `DFBuilder`: a helper class to build a Pandas dataframe incrementally,
    row-by-row.
  - `json_np_converter`: convert numpy values to JSON-compliant data type.
  - `PyExec`: typical use-case: to implement a data dictionary or config files
    that can mix-in Python code to construct certain variables.

- `smallmatter.ts`: for rapid EDA on timeseries data.
  - `fill_dt_all` and `fill_dt`: utilities to convert a long pandas dataframe
    that consists of multiple sparse timeseries to another long pandas dataframe
    that consists of multiple dense timeseries. In a long dataframe format, each
    timeseries is denoted by multiple rows, with each row denotes a timestamp.
    A sparse timeseries is a collection of rows with "missing" timestamp, which
    may represent "non event" or "missing measurement". A dense timeseries will
    have a row for those "non-event" timestamp with specific value (e.g., 0).

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
