
# Word2Vec Model Tensorboard Visualization

Tutorial | by Qixiang Fang

This is a tutorial on how to easily visualize and thus interpret your word2vec
models on the __[tensorboard projector](https://projector.tensorflow.org/)__.
We also demonstrate via such visualizations that __our approach of training a
data set on a model already built on the FastText wikipedia pre-trained word
vector data__ achieves better performance, in comparison to either models
trained on just a new data set or the wikepedia pre-trained data alone.

## Import Packages


```python

import os
import datetime

import gensim 
from gensim.models import Word2Vec
from gensim.scripts import word2vec2tensor
import pandas as pd
```

## Define Useful Functions


```python
# A simple function to pre-process data usable for gensim
def process_data(file):
    for line in file:
        yield gensim.utils.simple_preprocess(line)
```


```python
# A re-written gensim function to convert word2vec files to tensor formates, 
# due to decoding procedures incompatible with python 35
def word2vec2tensor2(word2vec_model_path, tensor_filename, binary=False):

    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model_path, binary=False)
    outfiletsv = tensor_filename + '_tensor.tsv'
    outfiletsvmeta = tensor_filename + '_metadata.tsv'
    
    with open(outfiletsv, 'w+') as file_vector:
        with open(outfiletsvmeta, 'w+') as file_metadata:
            for word in model.index2word:
                try:
                    file_metadata.write(gensim.utils.to_utf8(word).decode("utf-8") + gensim.utils.to_utf8('\n').decode("utf-8"))
                    vector_row = '\t'.join(str(x) for x in model[word])
                    file_vector.write(vector_row + '\n')
                except UnicodeEncodeError:
                    pass
```


```python
# A simple function to get current time in string formats, used later for file names
def get_time():
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H-%M")
    return time
```

## Data Preparation


```python
# Location of the training data
fp_ptsd = os.path.join("data", "datasets", "ptsd_review", "schoot-lgmm-ptsd.csv")

# Import data (the ptsd data set)
data = pd.read_csv(fp_ptsd)

# Merge individual titles and abstracts into single strings
usedata = data['title'].fillna('') + ' ' + data['abstract'].fillna('')

# Convert from panda data frame to lists
usedata_list = usedata.values.tolist() 

# Preprocess data into a format workable with gensim
usedata_gen = process_data(usedata_list)
usedata_clean = list(usedata_gen) #This is the data we will be working with
```

## Word2Vec Models

### Preparing Facebook FastText files

This section details how you can convert FastText word2vec files into gensim word2vec files which gensim can work with. 

You can download the FastText word2vec files from https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
Please use the _'text'_ file instead of _'bin + text'_. 


```python
from gensim.models import KeyedVectors

# Load the original Facebook FastText files. This can take a while!!!
en_model = KeyedVectors.load_word2vec_format('wiki.en.vec')
# Convert and save as gensim word2vec models
en_model.save_word2vec_format("wiki"+".bin", binary=True)
del en_model
```

## Model 1: PTSD dataset only


```python
# Train a word2vec model based on only the PTST data set
# Only includes words with a frequency of at least 10
model1 = Word2Vec(usedata_clean, size = 300, min_count=10)

# Check words similar to 'ptsd'
model1.most_similar(positive = "ptsd")

# Save model as word2vec format
model1_name = "word2vec_ptsd" + get_time()
model1.wv.save_word2vec_format(model1_name + ".txt", binary=False)

# Convert word2vec file to tensorboard format
word2vec2tensor2(model1_name + ".txt", model1_name, binary = False)
```

## Model 2: FastText/Wikipedia pre-trained word embeddings only


```python
# Prepare a word2vec model with a vocabulary identital to the one in Model 1
model2 = Word2Vec(size=300, min_count=10)
model2.build_vocab(usedata_clean)
len(model2.wv.vocab) #Check vocabulary size

# Load the FastText data file into the model, train it, but only for words already
#defined in the vocabulary, and set lockf = 1 so that word vectors can be updated
model2.intersect_word2vec_format(fname = "wiki.bin", binary = True, lockf = 1)

# Check words similar to 'ptsd'
model2.most_similar(positive = "ptsd")

# Save model as word2vec format
model2_name = "word2vec_wiki" + get_time()
model2.wv.save_word2vec_format(model2_name + ".txt", binary=False)

# Convert word2vec file to tensorboard format
word2vec2tensor2(model2_name + ".txt", model2_name, binary = False)
```

## Model 3: PTST + FastText/Wikipedia


```python
# Reuse model 2 for model 3
model3 = model2

# Continue training model 2 with the PTSD data set
model3.train(usedata_clean, total_examples=len(usedata_clean), epochs = 10)

# Check words similar to 'ptsd'
model3.most_similar(positive = "ptsd")

# Save model as word2vec format
model3_name = "word2vec_both" + get_time()
model3.wv.save_word2vec_format(model3_name + ".txt", binary=False)

# Convert word2vec file to tensorboard format
word2vec2tensor2(model3_name + ".txt", model3_name, binary = False)
```

## Visualize word embeddings

You can observe how these three models relate to each other by checking word associations. For instance, we have done so by finding words similar to '__ptsd__'. You can see that the results become more meaningful and interpretable over the three models. 

But what is more interesting is to visualize the results using the tensorboard files (tsv. formats) we generated for each model.

1. Use this website: https://projector.tensorflow.org/

2. Upload your data: both __metadata__ and __vector__ files

3. Click anywhere outside the box where you upload your data to proceed

4. Activate regular expression (__.\*__) for the search bar on the lefthand panel

5. Search keywords: Use '__\b__' as a container for an exact match of keywords. For instance, __\bptsd\b__, where you can also see semantically closest words for '__ptsd__'.

6. Use "__|__" (OR operator) to show multiple keywords at the same time, such as 'ptsd' and 'stress': __\bptsd\b|\bstress\b__. Don't include space.

Across models (i.e. FastTest vs. PTSD Data Set vs. FastTest+PTSD), you can see for instance the relationship between such a word pair improves or worsens. For the word pair '__ptsd vs. stress__', we can see that these two words become spatially closer to each other over the three models, suggesting that using the FastText wikipedia data set as the first embedding layer in a word2vec model improves performance of the model.
