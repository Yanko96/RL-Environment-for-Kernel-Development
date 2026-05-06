In this tutorial, we’ll create a simple RL task in which an agent is tasked with training a model to maximize performance on MNIST using PyTorch.
The purpose of this tutorial is to teach you how to build an RL task inside of `pm_env_slim`. 

Training an MNIST classifier is, of course, a fairly trivial task for current state-of-the-art LLMs. 
Thus, this isn't a very interesting task to create.
For your take-home assignment, you should implement something much more interesting and challenging.

By the way: `pm_env_slim` stands for "Preference Model Environment, slim version". It's a much simpler version of the infrastructure we use to build RL environments at Preference Model.

## Installing environment dependencies

First, run the command

```bash
uv sync
```

This creates a virtual environment based on the dependencies in the `pyproject.toml` file. If you want to use the packages in `pyproject.toml` for local testing, you can activate the venv with

```bash
source .venv/bin/activate
```

**Installing torch[cpu]**

We’ll use `torchvision` to download the MNIST dataset. The judge will also need to use `torch`. Since `torch` and `torchvision` aren’t in the `pyproject.toml` dependencies, we’ll need to add them to `pyproject.toml`.

Add the following to pyproject.toml:

```bash
[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]
torchvision = [{ index = "pytorch-cpu" }]

[[tool.uv.index]]   
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

Then, `torch` and `torchvision` to the `pyproject.toml` dependencies manually or using the command

```bash
uv add torch torchvision
```

**Installing torch[gpu]**

If you need to use a GPU, use the command

```bash
uv add torch torchvision
```

without adding anything else to the `pyproject.toml`.

## Installing model dependencies

The packages in `pyproject.toml` have not yet been made available to the agent, because the agent runs in a sandbox inside the container. 

To make python packages available to the agent, we want to change `env_requirements.txt` as shown below:

```python
# Add any Python dependencies the agent needs for solving tasks
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.11.0+cpu
torchvision==0.26.0+cpu
```

Note that there are two different virtual environments, one for the judge (`pyproject.toml`) and one for the agent during the run (`env_requirements.txt`).

## Downloading MNIST train and test datasets

We will need to put the MNIST train dataset in the `env_data` folder, and the full MNIST dataset in the `scoring_data` folder. This makes sure that the agent doesn’t have access to the test dataset in the `env_data` folder.

To do this, edit the file `setup_data.py` . This file will download the data into the appropriate folders and delete the testing data from `env_data`. `setup_data.py` should look like:

```python
# This script gets executed on the build machine that builds your environment,
# but before the environment is created. That means you cannot depend on system
# packages that are not installed on the build machine, so keep it simple.
# Add any dependencies required for data setup in the section below.

# /// script
# requires-python = "==3.12.*"
# dependencies = ["torch==2.10.0+cpu", "torchvision==0.25.0+cpu"]
#
# [tool.uv]
# extra-index-url = ["https://download.pytorch.org/whl/cpu"]
# index-strategy = "unsafe-best-match"
# ///

import glob
import os

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

def main():
    train_ds = MNIST("env_data", train=True, transform=ToTensor(), download=True)
    test_ds = MNIST("scoring_data", train=False, transform=ToTensor(), download=True)

    # delete test files from env_data to prevent reward hacking
    for file in glob.glob("env_data/MNIST/raw/t10k*"):
        os.remove(file)

if __name__ == "__main__":
    main()
```

We can then run `setup_data.py` locally to download the MNIST data into the appropriate folders. This should be done with the bash command

```bash
uv run setup_data.py
```

These files should not be committed to GitHub. 

## Implement task

Now we need to edit the `get_tasks` function in `tasks.py`. This is where we explain to the model what it needs to do, and create the `Judge` we will use to evaluate it.In this task, there is only one `Step`. We modify the `instructions` and create a `Judge`.

```python
import sys
from pathlib import Path
from textwrap import dedent

from pm_env.get_data_dir import get_env_data_dir, get_scoring_data_dir
from pm_env.judges.executable_judge import ExecutableJudge
from pm_env.judges.regex_judge import RegexJudge
from pm_env.schemas.evaluation_run_config import EvaluationRunConfig
from pm_env.task import Step, Task

def get_tasks(config: EvaluationRunConfig) -> list[Task]:
    """Create tasks for this environment."""
    module_name = "classifier.py"
    checkpoint_name = "mnist-classifier.pt"
    score_script = (Path(__file__).parent / "score_mnist_classifier.py").as_posix()

    return [
        Task(
            id="train-mnist",
            # Check out tool implementations in the `tools` directory
            tools=["bash"],
            steps=[
                Step(
                    instructions=dedent(f"""
                    Your task is to train a machine learning model that classifies
                    MNIST digits. Your goal is to achieve an accuracy above 95%.
                    
                    You have access to the MNIST training data in {get_env_data_dir()}.
                    However, you do not have access to the MNIST testing data. This means
                    that torchvision.datasets.MNIST will not work. You will need to write
                    your own custom Dataset.
                    
                    You should save two files. The first file should be named
                    {module_name} and contain a torch class MNISTClassifier.
                    The second file {checkpoint_name} should contain the state_dict.
                    """),
                    judge=ExecutableJudge(
                        [
                            sys.executable,
                            score_script,
                            f"{get_env_data_dir()}/{module_name}",
                            f"{get_env_data_dir()}/{checkpoint_name}",
                            "/tmp/mnist_classifier_results.txt",
                        ]
                    ),
                ),
            ],
        ),
    ]


```

The instructions tell the model to train a PyTorch model and save it as `mnist-classifier.pt` in the `env_data` dir. To ensure the model can be loaded by the scoring script, it also saves the subclass of `nn.Module` in a file `classifier.py`.

We use an `ExecutableJudge`. This judge is intended to allow easy evaluation of model-written code or other files that the model saves into its workdir. For more information, see the docstring in `executable_judge.py`.

## Write scoring script

Our `ExecutableJudge` allows us to run any scoring script to evaluate the model, so long as the script outputs a txt file with a numerical score.

Our scoring function, as mentioned above, is called `score_mnist_classifier.py`. We put this inside of `src/pm_env`.

```python
import importlib.util
import json
import sys
from pathlib import Path

import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from pm_env.get_data_dir import get_scoring_data_dir

def _import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def get_eval_accuracy(model):
    test_ds = MNIST(get_scoring_data_dir(), train=False, transform=ToTensor(), download=False)
    test_dl = DataLoader(test_ds, batch_size=256)

    num_accurate = 0

    for X, y in test_dl:
        y_hat = model(X)
        num_accurate += (torch.argmax(y_hat, dim=-1) == y).sum()
    
    return (num_accurate / len(test_ds)).item()

if __name__ == '__main__':
    try:
        classifier_module = _import_from_path('MNISTClassifier', sys.argv[1])
    except Exception as e:
        score = 0
        metadata = {
            'error': 'Cannot import module',
            'exception': str(e),
        }
    else:
        try:
            model = classifier_module.MNISTClassifier()
        except Exception as e:
            score = 0
            metadata = {
                'error': 'Cannot create model from module',
                'exception': str(e),
                'module': classifier_module.__name__,
            }
        else:
            try:
                model.load_state_dict(torch.load(sys.argv[2], weights_only=True))
            except Exception as e:
                score = 0
                metadata = {
                    'error': 'Failed to load model',
                    'exception': str(e),
                }
            else:
                try:
                    score = get_eval_accuracy(model)
                    metadata = {'error': 'None'}
                except Exception as e:
                    score = 0
                    metadata = {
                    'error': 'Failed to evaluate dataset',
                    'dataset dir': str(get_scoring_data_dir()),
                    'exception': str(e),
                }

    Path(sys.argv[-1]).write_text(json.dumps({"score": score, "metadata": metadata}))
```

## Prepare run config file

You will use a configuration file `run_config.json` for evaluating the RL environment. This file can be created using the command

```bash
uv run pm_env create-run-config --model claude-haiku-4-5-20251001 --model-api-key $ANTHROPIC_API_KEY
```

You should now see that `run_config.json` has been created. You should make sure that `task-id` is set to the current task ID, in this case `"train-mnist"`.

## Running environment

We can run our environment using 

```python
uv run pm_env run --config run_config.json
```

This will launch the environment. If everything was configured successfully, the agent should run for a few minutes then output a successful answer.
