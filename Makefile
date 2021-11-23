SHELL=/bin/bash

clean:
	rm -r build *.so *.egg-info

develop:
	python setup.py develop

test:
	python test.py

all: clean develop test

.PHONY: clean develop test
