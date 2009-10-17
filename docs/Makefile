#
# Makefile for PyBLAW documentation
#

SPHINXBUILD = sphinx-build
SPHINXOPTS  = -d _doctrees
SPHINXGHPAGES = ~/bin/sphinx-to-github

html:
	$(SPHINXBUILD) -b html $(SPHINXOPTS) . _build
	rm -rf _ghpages/*
	cp -R _build/* _ghpages/
	cd _ghpages && $(SPHINXGHPAGES)

all: html

.PHONY: html
