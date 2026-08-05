.PHONY: help install test lint fmt app bench figures clean

help:  ## Show this help
	@grep -E "^[a-z]+:.*##" $(MAKEFILE_LIST) | sed -e "s/:.*## / -- /"

install:  ## Create .venv and install the project with all extras
	python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/pip install -e ".[app,dev]"

test:  ## Run the test suite
	pytest

lint:  ## Check style
	ruff check .

fmt:  ## Auto-fix style issues
	ruff check . --fix

app:  ## Launch the Streamlit app
	streamlit run app.py

bench:  ## Print the benchmark tables
	python benchmark.py

figures:  ## Regenerate the README figures
	python scripts/make_figures.py

clean:  ## Remove caches and build artefacts
	rm -rf __pycache__ */__pycache__ .pytest_cache .ruff_cache build dist *.egg-info
