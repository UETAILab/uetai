# Dataset Registry

The fact that UETAI has several datasets, some of them is very large, So it is not feasible to download the dataset everytime the runner is trigger. 
We create this submodule as an initiative to manage all those datasets at once.

# Usage
The modules will support some function and class to handle dataset management on multiple machine at UETAI.
We create a shared NFS upon our cluster. Every machine with GPU will access and read data from that. The Runner, there for, have to mount to the shared NFS in order to acess the data.

The developer will use `data_path` function from logger to access to certain dataset given `dataset_name`
and `alias`.

```python
from uetai.logger import SummaryWriter
logger = SummaryWriter("my_experiment")
data_path = logger.data_path(path="./my/local/path", dataset_name="echo", alias="latest")
```
The data_path actually follow the 2 environment scenario:
1. If developer run the code on local machine. It will return the original `path`.
2. If developer run the code on github runner. It will return the full data depend on `dataset_name` and `alias`.

# Registry
We create a registry to mapping and validate whether dataset path is correct.

```python


```