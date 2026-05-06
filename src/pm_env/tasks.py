import re
import sys
from textwrap import dedent

from pm_env.get_data_dir import get_env_data_dir, get_scoring_data_dir
from pm_env.judges.executable_judge import ExecutableJudge
from pm_env.judges.regex_judge import RegexJudge
from pm_env.schemas.evaluation_run_config import EvaluationRunConfig
from pm_env.task import Step, Task


def get_tasks(config: EvaluationRunConfig) -> list[Task]:
    """Create tasks for this environment."""

    return [
        Task(
            id="get-python-version",
            tools=["bash", "submit_answers"],
            steps=[
                # A task can consist of multiple steps, each with its own instructions and judge.
                # The evaluation run gets terminated early if the scoring for a step fails.
                Step(
                    instructions=dedent("""
                    Use the `submit_answers` tool to submit your answers.

                    Use the `bash` tool to determine the path to your Python executable.

                    Submit your answer in the following format:
                    path: /path/to/python3
                    """),
                    judge=RegexJudge(
                        {"path": re.compile(r".*/workdir/\.venv/bin/python.*")}
                    ),
                ),
                Step(
                    instructions=dedent(f"""
                    Use the `bash` tool to determine the version of Python you are using.

                    Write you answer to {get_env_data_dir()}/python_version.txt
                    """),
                    judge=ExecutableJudge(
                        [
                            sys.executable,
                            "-m",
                            "pm_env.scoring_script",
                            "/tmp/score_output.txt",
                        ]
                    ),
                ),
            ],
        ),
    ]
