.PHONY: lint

lint:
	flake8 .
	mypy .
	black --check .

test:
	pytest tests
