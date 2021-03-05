""" Adds the Universtiät Wien Master Thesis Title Page to the documente main.tex 

Date: 05.03.2021
Author: j. weber

"""

from PyPDF2 import PdfFileReader, PdfFileWriter

thesis = PdfFileReader("main.pdf", "rb")
title_page = PdfFileReader("Deckblatt/title_page.pdf", "rb")

output = PdfFileWriter()
output.addPage(title_page.getPage(0))

for i in range(1, thesis.getNumPages()):
    output.addPage(thesis.getPage(i))

try:
    with open("thesis.pdf", "wb") as f:
        output.write(f)
except PermissionError:
    print("File is opened in some program!")


