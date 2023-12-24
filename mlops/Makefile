# Define targets
.PHONY: tests docs

# Default target
default: env

dependencies:
	@echo "Installing dependencies..."
	poetry install
	poetry run pre-commit install

env:
	@echo "Activating virtual environment..."
	poetry shell

tests:
	pytest

docs:
	@echo Save documentation to docs...
	pdoc src -o docs --force
	@echo View API documentation...
	pdoc src --http localhost:8080
