SHELL=/bin/bash

develop:
	python setup.py develop

test:
	python test.py

.PHONY: develop test
