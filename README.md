# abtesting-public
Open source core of A/B testing calculator

## What is this?
This is the core code powering my [A/B testing
calculator](https://abtesting.convexanalytics.com). I wanted to open-source the
core logic so that everyone could scrutinize the methods to their heart's
content. Reviewing the test cases is a good starting point.

### Architecture
We use [poetry](https://python-poetry.org/) to manage dependencies. Test cases
utilize [pytest](https://docs.pytest.org/en/latest/). We use
[black](https://github.com/python/black) and
[flake8](https://flake8.pycqa.org/en/latest/) for formatting.


## How to run the test cases
After ensuring poetry is installed (see Architecture section), run:

```
$ poetry shell
$ poetry install --no-root
$ python -m pytest
```
