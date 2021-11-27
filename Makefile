SHELL=/bin/bash

all: develop test

clean:
	rm -r build *.so *.egg-info

develop:
	python setup.py develop

test:
	python test.py

.PHONY: clean develop test
