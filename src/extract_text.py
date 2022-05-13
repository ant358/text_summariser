# TODO extract text from pdf
# TODO extract text from docx
# TODO extract text from doc


# %%
import os
import zipfile
import re
# to pretty print the xml:
import xml.dom.minidom

document = zipfile.ZipFile('C:Users\\AnthonyWynne\\OneDrive - SVGC Limited'
                           '\\Dev\\text_summariser\\text_data'
                           '\\docxs\\Mobile Working Policy.docx')

# %%
# show the document parts
# document.namelist()
# %%
# first to turn the xml content into a string:
xml_content = document.read('word/document.xml')
# document.close()
xml_str = str(xml_content)
# %%
# extract the text from the xml string
link_list = re.findall('preserve">*?\<',xml_str)[1:]
link_list = [x[:-1] for x in link_list]

# %%
import textract
text = textract.process("path/to/file.extension")