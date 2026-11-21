.PHONY: all clean view help
.DEFAULT_GOAL := all
# A failed build deletes main.pdf, so a rerun cannot pass with the broken PDF
# that latexmk wrote before it failed (for example, one with an undefined citation).
.DELETE_ON_ERROR:

# Everything main.tex reads. reported_values.tex is generated
# (`make reported-values` from the repository root) but not regenerated here.
FIGURES := $(wildcard figures/figure0*/figure0*.pdf) $(wildcard figures/supp_*/supp_*.pdf)
TABLES := $(wildcard tables/*.tex)

all: main.pdf

main.pdf: main.tex reported_values.tex references.bib $(FIGURES) $(TABLES)
	latexmk -pdf main.tex

clean:
	latexmk -C

view: main.pdf
	open main.pdf

help:
	@echo "make        Build main.pdf"
	@echo "make clean  Remove main.pdf and the LaTeX build files"
	@echo "make view   Open main.pdf"
