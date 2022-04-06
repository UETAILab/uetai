<div align="center">

<p align="center">
  <img src="docs\_static\images\logo.png" width="220px" alt="logo">
</p>

**Machine Learning tracking experiment and debugging tools.**

______________________________________________________________________


<!-- Badge start -->
| Branch 	| Build 	| Coverage 	| Linting 	| Release 	| License 	|
|--------	|-------	|----------	|---------	|---------	|---------	|
| main   	|       	| [![cov-test](https://codecov.io/gh/UETAILab/uetai/branch/main/graph/badge.svg?token=9KY7UU1QNB)](https://codecov.io/gh/UETAILab/uetai) | [![linting](https://github.com/UETAILab/uetai/actions/workflows/lint-test.yml/badge.svg)](https://github.com/UETAILab/uetai/actions/workflows/lint-test.yml)	| [![release](https://img.shields.io/pypi/v/uetai)](https://pypi.org/project/uetai/) [![pyversion](https://img.shields.io/pypi/pyversions/uetai)](https://pypi.org/project/uetai/)| [![license](https://img.shields.io/github/license/UETAILab/uetai)](https://github.com/UETAILab/uetai/blob/main/LICENSE.txt) |

<!-- Badge end -->
</div>

______________________________________________________________________
UETAI is a customize PyTorch logger which will able to help users track machine learning experiment, and esily debug raw datasets and trained models.

UETAI provided tools for helping user tracking their experiment, visualizing the dataset, results, and debuging the model (and the raw dataset also) with little effort by integrated the tools into the dashboards which users are using for logging.

*In this beta version, we will only focus on integrated Comet ML, which is amazing dashboard with well-writen API and customable panel*

<!--One of common problem is performance of model remains poorly, even though researcher applied quality control and monitoring process. In our experiment, the quaility of raw dataset are often underestimated, which leads to poor performance of model.-->

<!--However, visualizing and debugging it are not easy and time consuming, we believe a good solution to handle this problem can be integrated into the tools which users are using to monitor their experiments.-->

## Getting started
Firstly, you must sign up for an account from one of these supported MLTE (Machine Learning tracking experiment) tools, each dashboard will give you a unique API key to log in dashboard from any terminal or code:

| Dashboard        	| Status 	|
|------------------	|--------	|
| Comet ML         	|    ✅ 	 |
| Weights & Biases 	|    ❌   |
| MLFlow           	|    ❌ 	 |

### Install `uetai`
You install `uetai` with `pip` by running:
```bash
pip install uetai
```

Or install from source repository:
```bash
git clone git@github.com:UETAILab/uetai.git; cd uetai
pip install -e .
```

### Basic usage
Importing and initialize your supported dashboard logger (for example: Comet ML) and start logging your experiment:

```python
from src import CometLogger

logger = CometLogger(project_name="Uetai project")

# training process
logger.log({"loss": loss, "acc": acc})

```

<!-- Analysis your dataset:

```python

```

Logging model:
```python
```
 -->


## Examples

*Coming soon...*

## The team
UETAI is a non-profit project hosted by AI Laboratory of University of Engineering and Technology.

UETAI is currently maintained by [manhdung20112000](https://github.com/manhdung20112000) with the support from BS. Phi Nguyen Van - [gungui98](https://github.com/gungui98/) as an advisor.

## License
UETAI has a MIT license, as found in the [LICENSE](https://github.com/UETAILab/uetai/blob/main/LICENSE.txt) file.
