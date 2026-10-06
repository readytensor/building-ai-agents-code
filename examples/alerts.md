# Installing the agent

> [!NOTE]
> The agent runs on any OpenAI-compatible endpoint. Pick the provider in `.env`.

Clone the repository and install the requirements:

```sh
pip install -r requirements.txt
```

> [!TIP]
> A virtual environment keeps the agent's packages apart from the rest of
> your Python installation.

> [!IMPORTANT]
> Copy `.env.example` to `.env` and add your API key before the first run.

> [!WARNING]
> The `bash` tool is not sandboxed. It starts in the working copy, but it runs
> with your permissions.

> [!CAUTION]
> Pointing the agent at a repository you care about, outside a container or a
> virtual machine, risks changes you did not ask for.

A quote that is not an alert still renders as one:

> Make it work, make it right, make it fast.
