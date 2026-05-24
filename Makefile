.PHONY: install install-all dev lint format typecheck test test-unit test-integration clean build help

# Setup 

install:
	uv pip install -e ".[dev]"

install-all:
	uv pip install -e ".[all,dev]"

dev: install
	pre-commit install
	@echo "Dev environment ready"

# Quality

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck:
	mypy src/nexrag

# Tests

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v -m unit

test-integration:
	pytest tests/integration/ -v -m integration

test-cov:
	pytest tests/unit/ --cov=nexrag --cov-report=term-missing --cov-report=html

# Build 

build:
	uv build

clean:
	rm -rf dist/ build/ .eggs/ *.egg-info
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +

# Help

help:
	@echo ""
	@echo "NexRAG — Developer Commands"
	@echo "────────────────────────────"
	@echo "  make install          Install core + dev dependencies"
	@echo "  make install-all      Install all optional extras + dev"
	@echo "  make dev              Install + set up pre-commit hooks"
	@echo "  make lint             Run ruff linter"
	@echo "  make format           Auto-format with ruff"
	@echo "  make typecheck        Run mypy type checker"
	@echo "  make test             Run all tests"
	@echo "  make test-unit        Run unit tests only"
	@echo "  make test-integration Run integration tests only"
	@echo "  make test-cov         Run unit tests with coverage report"
	@echo "  make build            Build distribution packages"
	@echo "  make clean            Remove build artifacts and caches"
	@echo ""