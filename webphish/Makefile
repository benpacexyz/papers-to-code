# Build the agent's copy of the repo.
#
#   make agent        build ../webphish-auto and set up its venv
#   make agent-clean  delete the agent's copy between attempts

.PHONY: agent agent-clean

agent:
	uv run --active python make_agent_copy.py
	uv sync --directory ../webphish-auto --all-extras --dev

agent-clean:
	uv run --active python -c "import shutil; shutil.rmtree('../webphish-auto', ignore_errors=True)"
