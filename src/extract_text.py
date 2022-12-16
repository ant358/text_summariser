# TODO extract text from pdf
# TODO extract text from docx
# TODO extract text from doc
# TODO extract text from json
# TODO extract text from XML


# %%
import os
import zipfile
import re
# to pretty print the xml:
import xml.dom.minidom
import textract

test_docx_path = ('C:Users\\AnthonyWynne\\OneDrive - SVGC Limited'
                  '\\Dev\\text_summariser\\text_data'
                  '\\docxs\\Mobile Working Policy.docx')
# document = zipfile.ZipFile
# # show the document parts
# # document.namelist()
# # first to turn the xml content into a string:
# xml_content = document.read('word/document.xml')
# # document.close()
# xml_str = str(xml_content)
# # extract the text from the xml string
# link_list = re.findall('preserve">*?\<',xml_str)[1:]
# link_list = [x[:-1] for x in link_list]

# %%
# extract the text from the docx
text = textract.process(test_docx_path)
# %%
