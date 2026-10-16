# my-agent

A Python 3.13+ starter project for building an agent with LangChain and
LangChain's OpenAI integration.

## Prerequisites

- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/) for dependency and environment management
- Git, if you are cloning the repository

## Get started

Clone the repository and enter its directory, then sync the project and its
development dependencies:

```sh
git clone <repository-url>
cd my_agent
uv sync --group dev
```

`uv` creates the local `.venv` and installs dependencies from `uv.lock`.
Activate the environment if you prefer to run Python commands directly:

```sh
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows PowerShell
```

## Usage

Run the installed command:

```sh
uv run my-agent
```

Or call the package entry point through Python:

```sh
uv run python -c 'from my_agent import main; main()'
```

Both currently print `Hello from my-agent!`.

## Configuration

## Development

Install the development dependency group (included in the setup command above):

```sh
uv sync --group dev
```

Available project tools include [pytest](https://docs.pytest.org/) and
[Ruff](https://docs.astral.sh/ruff/). Run them with:

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

To format code after making changes:

```sh
uv run ruff format .
```

There are no tests in the repository yet, so `pytest` currently reports that
no tests were collected until tests are added.

## Project layout

```text
src/my_agent/
├── __init__.py   # Package entry point
├── agent.py      # Current CLI behavior
└── tools.py      # Agent tools (implementation to come)
```

Project metadata and dependencies are defined in `pyproject.toml`; `uv.lock`
records the resolved dependency versions.
