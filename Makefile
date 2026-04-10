install-uv:
	curl -LsSf https://astral.sh/uv/install.sh | sh
install:
	uv sync
install-dev:
	uv sync --dev
test:
	uv run pytest
build:
	rm -rf dist/
	uv build
