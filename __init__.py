import os
import requests

GLOVE_DIR = "word2vec"
os.mkdir( GLOVE_DIR)
GLOVE_File_Name = os.path.join(GLOVE_DIR,"wiki.en.vec")
URL = "https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec"

response = requests.get(URL, stream=True)
handle = open(GLOVE_File_Name, "wb")
for chunk in response.iter_content(chunk_size=512):
    if chunk:  # filter out keep-alive new chunks
        handle.write(chunk)