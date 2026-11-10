# latexmk settings for main.tex: pdflatex, and BibTeX run when needed.
$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -halt-on-error -synctex=1 %O %S';
$clean_ext = "synctex.gz run.xml bbl";
$max_repeat = 5;
# Fail the build on undefined citations or references and on BibTeX warnings,
# so a mistyped key cannot ship a PDF that shows "[?]".
$warnings_as_errors = 1;
