# UETAI
![workflow](https://github.com/UETAILab/uetai/actions/workflows/lint-test.yaml/badge.svg)
[![codecov](https://codecov.io/gh/UETAILab/uetai/branch/main/graph/badge.svg?token=9KY7UU1QNB)](https://codecov.io/gh/UETAILab/uetai)

This is library for multiple utilities and templates for project in UET-AILAB

## Installation

```bash
pip install -e .
```

## Getting started

```python
import uetai
from uetai.logger import SummaryWriter
writer = SummaryWriter("some_experiment")
```