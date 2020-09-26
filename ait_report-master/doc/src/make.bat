@pushd \
@%~d0
@cd "%~dp0"

set MAIN_DOC=main

::needed for tikz grafics (allows to call commands from tex)
set pdflatex_args=-shell-escape

pdflatex %pdflatex_args% %MAIN_DOC%.tex
bibtex %MAIN_DOC%
pdflatex %pdflatex_args% %MAIN_DOC%.tex
pdflatex %pdflatex_args% %MAIN_DOC%.tex
@popd