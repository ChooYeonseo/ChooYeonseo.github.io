---
layout: single
title:  "word2vec"
permalink: /paper-implementation/ml-methology/initialization-xavier/
toc: true
author_profile: false
sidebar:
    nav: "Gen-Mthd"
---
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


# COSE461 Assignment 1: Word Vectors

### **Due 9:00 AM, Tue Mar 21**



Welcome to COSE461.



## About using Colab

Our assignment will be in the form of Jupyter notebooks to be able to be run in Google Colab. Colab is an online editor that provides free access to a GPU, so you don't have to worry about the computing resources when doing the assignements. GPU can be accessed by `Edit->Notebook settings` or `수정->노트 설정` and choosing GPU in the Hardware accelerator dropdown (We will not be using a GPU in the current assignment, though). To get started, make a copy of the assignment by clicking `File->Save a copy in drive...` or `파일->드라이브에 사본 저장`.  You will need to be logged into a Google account, such as your @korea.ac.kr mail account.



 The code cells in Colab are executed on a `preemtible` VM instance. This means:

 - Every time you open the notebook, you will be assigned a different machine; run cells and files will be lost.

 - Even if you keep using a machine attached to notebook, it can be lost and altered with other machine after 24 hours.



Therefore, if you have some output that you don't want to lose, you should download them to your local machine or save them to your Google drive by mounting it to the VM instance. The former can be done by `File->Download` or `파일->다운로드`, where the latter can be done by using something similar to the following code snippet:





```

from google.colab import drive



# mount Google drive

drive.mount('/mnt/')



# You can run a linux command by putting ! at the start

# The following commands shows the number of files in your drive

!echo -e "\n# of files in /mnt/My Drive/:"

!ls -l "/mnt/My Drive/" | wc -l



```


One important limitation to Colab is that it disconnects from VM when is inactive for a few tens of minutes. So you should either check Colab periodically, or use something like [this](https://stackoverflow.com/questions/54057011/google-colab-session-timeout). If you have available computing resources and if you do not want to go through such inconvenience, feel free to use yours by recreating a similar environment to Colab.



---





# Word vectors

This assignment is based on the Stanford CS224n lecture by Christopher Manning. The followings are the import, data downloading, and unzipping statements. Do not modify, and simply run it once before going down.



```python
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import pprint
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]
import nltk
nltk.download('reuters')
from nltk.corpus import reuters
import numpy as np
import random
import scipy as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

np.random.seed(0)
random.seed(0)
!unzip -nq /root/nltk_data/corpora/reuters.zip -d /root/nltk_data/corpora
```

<pre>
[nltk_data] Downloading package reuters to /root/nltk_data...
</pre>
Word Vectors are often used as a fundamental component for downstream NLP tasks, e.g. question answering, text generation, translation, etc., so it is important to build some intuitions as to their strengths and weaknesses. Here, you will explore two types of word vectors: those derived from co-occurrence matrices, and those derived via GloVe.



**Note on Terminology**: The terms "word vectors" and "word embeddings" are often used interchangeably. The term "embedding" refers to the fact that we are encoding aspects of a word's meaning in a lower dimensional space. As Wikipedia states, "*conceptually it involves a mathematical embedding from a space with one dimension per word to a continuous vector space with a much lower dimension*".


# Part 1: Count-based word vectors

#### You Shall know a word by the company it keeps (Firth, J. R. 1957:11)

Many word vector implementations are driven by the idea that **similar words**, i.e., (near) synonyms, **will be used in similar contexts**. As a result, similar words will often be spoken or written along with a shared subset of words, i.e., contexts. By examining these contexts, we can try to develop embeddings for our words. With this intuition in mind, many "old school" approaches to constructing word vectors relied on word counts. Here we elaborate upon one of those strategies, *co-occurrence matrices* (for more information, see [here](https://medium.com/data-science-group-iitr/word-embedding-2d05d270b285)).



## Co-occurence

A co-occurrence matrix counts how often things co-occur in some environment. Given some word  $w_i$  occurring in the document, we consider the context window surrounding  $w_i$ . Supposing our fixed window size is  $n$ , then this is the  $n$  preceding and  $n$  subsequent words in that document, i.e. words  $w_{i−n}…w_{i−1}$  and  $w_{i+1}…w_{i+n}$ . We build a co-occurrence matrix  $M$ , which is a symmetric word-by-word matrix in which  $M_{ij}$  is the number of times  $w_j$  appears inside  $w_i$ 's window among all documents.



**Example: Co-occurence with fixed window of $n=1$**



Document 1: *all that glitters is not gold*



Document 2: *all is well that ends well*





| *   |	`<START>`	| all	| that | glitters	| is	| not	| gold	| well	| ends	|`<END>`|

| --- | ---     | --- | ---  | ---      | --- | --- | ---   | ---   | ---   | --- |

|`<START>`	|0|	2|	0|	0|	0|	0|	0|	0|	0|	0| |

|all	|2	|0	|1	|0	|1	|0	|0	|0	|0	|0 |

|that	|0	|1	|0	|1	|0	|0	|0	|1	|1	|0 |

|glitters	|0	|0	|1	|0	|1	|0	|0	|0	|0	|0|

|is	|0	|1	|0	|1	|0	|1	|0	|1	|0	|0|

|not|	0|	0|	0|	0|	1|	0|	1|	0|	0|	0|

|gold|	0|	0|	0|	0|	0|	1|	0|	0|	0|	1|

|well|	0|	0|	1|	0|	1|	0|	0|	0|	1|	1|

|ends	|0	|0	|1	|0	|0	|0	|0	|1	|0	|0|

|`<END>`	|0	|0	|0	|0	|0	|0	|1	|1	|0	|0|



**Note:** In NLP, we often add `<START>` and `<END>` tokens to represent the beginning and end of sentences, paragraphs or documents. In thise case we imagine `<START>` and `<END>` tokens encapsulating each document, e.g., "`<START>` All that glitters is not gold `<END>`", and include these tokens in our co-occurrence counts.



The rows (or columns) of this matrix provide one type of word vectors (those based on word-word co-occurrence), but the vectors will be large in general (linear in the number of distinct words in a corpus). Thus, our next step is to run dimensionality reduction. In particular, we will run SVD (Singular Value Decomposition), which is a kind of generalized PCA (Principal Components Analysis) to select the top  $k$  principal components. Here's a visualization of dimensionality reduction with SVD. In this picture our co-occurrence matrix is  $A$  with  $n$  rows corresponding to  $n$  words. We obtain a full matrix decomposition, with the singular values ordered in the diagonal  $S$  matrix, and our new, shorter length- $k$  word vectors in  $U_k$.


<div align="center">



![svd.png](data:image/png;base64,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OgBbt261I4880v773/+6BT1at26dWrxH17k4H3/88da3b1/bvHlz3nOzpyqPwAf6nHN04pw5c+zUU0+1Z555JsczyR5XAlrM4ZNPPrHLLrvMBg8ebJMnT3ZfWskFXL7//nt3fPz48TZx4kTTikTaRyoegY8//tg6d+5supGSjw4++GC76KKLnKDoSY9UmcD69evtzTfftJNPPtl69+5td911lxPr9FwSYj0pX3zxxe4GR/8H1113XSUBT8+vz/rfCHrluap1erutValI2RM444wztIpXReJOOvuTyAmBBIGePXu6a2fs2LEZeSSWxKxIiHPGY+wMlsB5551XMWjQoJ0qSTw9V4waNWqn/ezYQeDKK69017Su7Uzp8ssvr3jyySczHapIiLU794ILLsh4PO47eXLO4bZJd4LqF1HSE7Re25AgAIHwE3j99ddNy9FWXZb06quvDn/jAmxB4sbFddfMmzfPFi1aVKmmTZs22SuvvGJDhgyptJ+N7Aggztlxcrnuu+8+u/baa61Pnz5ue8KECTmcTVYIQMBXAu3bt3c32+oz/d3vfmczZsxw60ofd9xx7lWsr3aX2q6DDjrITj/9dGfGHXfcUcmcSZMm2fnnn5/ql650kI0aCSDONSLakUH9g48//ridc845dskll7idf/vb30z9KCQIQCDcBMaMGeP6OhUMdtNNN9lJJ53kIolHjhzpgsPC3bpgrb/qqqtcBYnX16a+eyV9X+r7ccSIEW676q97773XrrnmGkt0JdiWLVvs0ksvdUGTVfPFeZto7Sy9ryhORRMqWOG0006z5s2bu8jDxx57zN0dZlkM2SAAAQ8JdO/e3XVZSTRmzpxp7777bkpg9tlnH0vECXhotR8mHXPMMdajRw9bsGCBCwy77bbbTELdv39/a9y4cUYjddND2jUBnpx3zSd19J577rGuXbua+lbUP9WrVy93TPtJEMiGgIaU1JSyyVNTGRzPnYCGBR166KH25z//2ZYtW2YbN260J554wpo0aWJ6PavhbaTqCVxxxRXuoLr+NBRK0dvqjyblTwBxzoKdhldoYLwCwmbPnu1+OnTo4MalKghC4yBJEKiJgN62KJWXl2fMqutLkzKQik9g6NChlSrVE9+ZZ55pv/zlL91rV43TJVVPQP3Obdu2NQWBaQhaixYtTP3RpAIIxD1cPZv2J/pUKhL9zTtlTQ6rSvSr7HSMHRCoSmDu3Llu6Mjhhx9e9VBFYtxoRZcuXSoSE2HsdIwdwRM45JBDKhJPyDtVpKFCide2O+1nx84EEgFh7vpOyFHF/Pnzd87AnpwIMH3nLm5sNLRCMzsp4lCTQyhyc7/99nNnKHDk6aeftuuvv97NjDN9+nRr1aqVKeqTBIHqCCioUE8W6hbRU5liGDQ8T7ELd999t/34xz+u7lT2B0igXbt2blYqPf1pOknN8Sy/3HLLLTZr1izXpRVg9ZEo+uuvv3bfgR07drSFCxdGok2lbATivAv6L774oi1dutS9vlb0oQLB1C+lJDHWOOf06et07JRTTtlFiRyCgLlpC9U9okUANBOYvsw0K9Uee+wBnhIR0I22Xs2uWLHCBTapi6Fly5aWeDtWbVBTiUz1utqpU6c6geYms3A3Ic6FM6QECEAAAhCAQK0SICCsVnFSGAQgAAEIQKBwAohz4QwpAQIQgAAEIFCrBBDnWsVJYRCAAAQgAIHCCSDOhTOkBAhAAAIQgECtEkCcaxUnhUEAAhCAAAQKJ4A4F86QEiAAAQhAAAK1SgBxrlWcFAYBCEAAAhAonADiXDhDSoAABCAAAQjUKgHEuVZxUhgEIAABCECgcAKIc+EMKQECEIAABCBQqwQQ51rFSWEQgAAEIACBwgn8H93cyehHB5yuAAAAAElFTkSuQmCC)



</div>



This reduced-dimensionality co-occurrence representation preserves semantic relationships between words, e.g. *doctor* and *hospital* will be closer than *doctor* and *dog*.



**If you are not friendly to SVD:** you can refer to [this comprehensive tutorial](https://davetang.org/file/Singular_Value_Decomposition_Tutorial.pdf) or these lecture notes ([1](https://web.stanford.edu/class/cs168/l/l7.pdf), [2](http://theory.stanford.edu/~tim/s15/l/l8.pdf), [3](https://web.stanford.edu/class/cs168/l/l9.pdf) ) of Stanford CS168. While these materials give a great introduction to PCA/SVD, you do not have to understand all of these materials for this assignment or this class.


##Plotting Co-Occurrence Word Embeddings

Here, we will be using the Reuters (business and financial news) corpus. If you haven't run the import cell above, please run it now (click it and press SHIFT-RETURN). The corpus consists of 10,788 news documents totaling 1.3 million words. These documents span 90 categories and are split into train and test. For more details, please see [this](https://www.nltk.org/book/ch02.html). We provide a `read_corpus` function below that pulls out only articles from the "crude" (i.e. news articles about oil, gas, etc.) category. The function also adds `<START>` and `<END>` tokens to each of the documents, and lowercases words. You do not have to perform any other kind of pre-processing.



```python
START_TOKEN = '<START>'
END_TOKEN = '<END>'

def read_corpus(category="crude"):
    """ Read files from the specified Reuter's category.
        Params:
            category (string): category name
        Return:
            list of lists, with words from each of the processed files
    """
    files = reuters.fileids(category)
    return [[START_TOKEN] + [w.lower() for w in list(reuters.words(f))] + [END_TOKEN] for f in files]
```

Let's have a look what these documents are like....



```python
reuters_corpus = read_corpus()
pprint.pprint(reuters_corpus[:3], compact=True, width=100)
```

<pre>
[['<START>', 'japan', 'to', 'revise', 'long', '-', 'term', 'energy', 'demand', 'downwards', 'the',
  'ministry', 'of', 'international', 'trade', 'and', 'industry', '(', 'miti', ')', 'will', 'revise',
  'its', 'long', '-', 'term', 'energy', 'supply', '/', 'demand', 'outlook', 'by', 'august', 'to',
  'meet', 'a', 'forecast', 'downtrend', 'in', 'japanese', 'energy', 'demand', ',', 'ministry',
  'officials', 'said', '.', 'miti', 'is', 'expected', 'to', 'lower', 'the', 'projection', 'for',
  'primary', 'energy', 'supplies', 'in', 'the', 'year', '2000', 'to', '550', 'mln', 'kilolitres',
  '(', 'kl', ')', 'from', '600', 'mln', ',', 'they', 'said', '.', 'the', 'decision', 'follows',
  'the', 'emergence', 'of', 'structural', 'changes', 'in', 'japanese', 'industry', 'following',
  'the', 'rise', 'in', 'the', 'value', 'of', 'the', 'yen', 'and', 'a', 'decline', 'in', 'domestic',
  'electric', 'power', 'demand', '.', 'miti', 'is', 'planning', 'to', 'work', 'out', 'a', 'revised',
  'energy', 'supply', '/', 'demand', 'outlook', 'through', 'deliberations', 'of', 'committee',
  'meetings', 'of', 'the', 'agency', 'of', 'natural', 'resources', 'and', 'energy', ',', 'the',
  'officials', 'said', '.', 'they', 'said', 'miti', 'will', 'also', 'review', 'the', 'breakdown',
  'of', 'energy', 'supply', 'sources', ',', 'including', 'oil', ',', 'nuclear', ',', 'coal', 'and',
  'natural', 'gas', '.', 'nuclear', 'energy', 'provided', 'the', 'bulk', 'of', 'japan', "'", 's',
  'electric', 'power', 'in', 'the', 'fiscal', 'year', 'ended', 'march', '31', ',', 'supplying',
  'an', 'estimated', '27', 'pct', 'on', 'a', 'kilowatt', '/', 'hour', 'basis', ',', 'followed',
  'by', 'oil', '(', '23', 'pct', ')', 'and', 'liquefied', 'natural', 'gas', '(', '21', 'pct', '),',
  'they', 'noted', '.', '<END>'],
 ['<START>', 'energy', '/', 'u', '.', 's', '.', 'petrochemical', 'industry', 'cheap', 'oil',
  'feedstocks', ',', 'the', 'weakened', 'u', '.', 's', '.', 'dollar', 'and', 'a', 'plant',
  'utilization', 'rate', 'approaching', '90', 'pct', 'will', 'propel', 'the', 'streamlined', 'u',
  '.', 's', '.', 'petrochemical', 'industry', 'to', 'record', 'profits', 'this', 'year', ',',
  'with', 'growth', 'expected', 'through', 'at', 'least', '1990', ',', 'major', 'company',
  'executives', 'predicted', '.', 'this', 'bullish', 'outlook', 'for', 'chemical', 'manufacturing',
  'and', 'an', 'industrywide', 'move', 'to', 'shed', 'unrelated', 'businesses', 'has', 'prompted',
  'gaf', 'corp', '&', 'lt', ';', 'gaf', '>,', 'privately', '-', 'held', 'cain', 'chemical', 'inc',
  ',', 'and', 'other', 'firms', 'to', 'aggressively', 'seek', 'acquisitions', 'of', 'petrochemical',
  'plants', '.', 'oil', 'companies', 'such', 'as', 'ashland', 'oil', 'inc', '&', 'lt', ';', 'ash',
  '>,', 'the', 'kentucky', '-', 'based', 'oil', 'refiner', 'and', 'marketer', ',', 'are', 'also',
  'shopping', 'for', 'money', '-', 'making', 'petrochemical', 'businesses', 'to', 'buy', '.', '"',
  'i', 'see', 'us', 'poised', 'at', 'the', 'threshold', 'of', 'a', 'golden', 'period', ',"', 'said',
  'paul', 'oreffice', ',', 'chairman', 'of', 'giant', 'dow', 'chemical', 'co', '&', 'lt', ';',
  'dow', '>,', 'adding', ',', '"', 'there', "'", 's', 'no', 'major', 'plant', 'capacity', 'being',
  'added', 'around', 'the', 'world', 'now', '.', 'the', 'whole', 'game', 'is', 'bringing', 'out',
  'new', 'products', 'and', 'improving', 'the', 'old', 'ones', '."', 'analysts', 'say', 'the',
  'chemical', 'industry', "'", 's', 'biggest', 'customers', ',', 'automobile', 'manufacturers',
  'and', 'home', 'builders', 'that', 'use', 'a', 'lot', 'of', 'paints', 'and', 'plastics', ',',
  'are', 'expected', 'to', 'buy', 'quantities', 'this', 'year', '.', 'u', '.', 's', '.',
  'petrochemical', 'plants', 'are', 'currently', 'operating', 'at', 'about', '90', 'pct',
  'capacity', ',', 'reflecting', 'tighter', 'supply', 'that', 'could', 'hike', 'product', 'prices',
  'by', '30', 'to', '40', 'pct', 'this', 'year', ',', 'said', 'john', 'dosher', ',', 'managing',
  'director', 'of', 'pace', 'consultants', 'inc', 'of', 'houston', '.', 'demand', 'for', 'some',
  'products', 'such', 'as', 'styrene', 'could', 'push', 'profit', 'margins', 'up', 'by', 'as',
  'much', 'as', '300', 'pct', ',', 'he', 'said', '.', 'oreffice', ',', 'speaking', 'at', 'a',
  'meeting', 'of', 'chemical', 'engineers', 'in', 'houston', ',', 'said', 'dow', 'would', 'easily',
  'top', 'the', '741', 'mln', 'dlrs', 'it', 'earned', 'last', 'year', 'and', 'predicted', 'it',
  'would', 'have', 'the', 'best', 'year', 'in', 'its', 'history', '.', 'in', '1985', ',', 'when',
  'oil', 'prices', 'were', 'still', 'above', '25', 'dlrs', 'a', 'barrel', 'and', 'chemical',
  'exports', 'were', 'adversely', 'affected', 'by', 'the', 'strong', 'u', '.', 's', '.', 'dollar',
  ',', 'dow', 'had', 'profits', 'of', '58', 'mln', 'dlrs', '.', '"', 'i', 'believe', 'the',
  'entire', 'chemical', 'industry', 'is', 'headed', 'for', 'a', 'record', 'year', 'or', 'close',
  'to', 'it', ',"', 'oreffice', 'said', '.', 'gaf', 'chairman', 'samuel', 'heyman', 'estimated',
  'that', 'the', 'u', '.', 's', '.', 'chemical', 'industry', 'would', 'report', 'a', '20', 'pct',
  'gain', 'in', 'profits', 'during', '1987', '.', 'last', 'year', ',', 'the', 'domestic',
  'industry', 'earned', 'a', 'total', 'of', '13', 'billion', 'dlrs', ',', 'a', '54', 'pct', 'leap',
  'from', '1985', '.', 'the', 'turn', 'in', 'the', 'fortunes', 'of', 'the', 'once', '-', 'sickly',
  'chemical', 'industry', 'has', 'been', 'brought', 'about', 'by', 'a', 'combination', 'of', 'luck',
  'and', 'planning', ',', 'said', 'pace', "'", 's', 'john', 'dosher', '.', 'dosher', 'said', 'last',
  'year', "'", 's', 'fall', 'in', 'oil', 'prices', 'made', 'feedstocks', 'dramatically', 'cheaper',
  'and', 'at', 'the', 'same', 'time', 'the', 'american', 'dollar', 'was', 'weakening', 'against',
  'foreign', 'currencies', '.', 'that', 'helped', 'boost', 'u', '.', 's', '.', 'chemical',
  'exports', '.', 'also', 'helping', 'to', 'bring', 'supply', 'and', 'demand', 'into', 'balance',
  'has', 'been', 'the', 'gradual', 'market', 'absorption', 'of', 'the', 'extra', 'chemical',
  'manufacturing', 'capacity', 'created', 'by', 'middle', 'eastern', 'oil', 'producers', 'in',
  'the', 'early', '1980s', '.', 'finally', ',', 'virtually', 'all', 'major', 'u', '.', 's', '.',
  'chemical', 'manufacturers', 'have', 'embarked', 'on', 'an', 'extensive', 'corporate',
  'restructuring', 'program', 'to', 'mothball', 'inefficient', 'plants', ',', 'trim', 'the',
  'payroll', 'and', 'eliminate', 'unrelated', 'businesses', '.', 'the', 'restructuring', 'touched',
  'off', 'a', 'flurry', 'of', 'friendly', 'and', 'hostile', 'takeover', 'attempts', '.', 'gaf', ',',
  'which', 'made', 'an', 'unsuccessful', 'attempt', 'in', '1985', 'to', 'acquire', 'union',
  'carbide', 'corp', '&', 'lt', ';', 'uk', '>,', 'recently', 'offered', 'three', 'billion', 'dlrs',
  'for', 'borg', 'warner', 'corp', '&', 'lt', ';', 'bor', '>,', 'a', 'chicago', 'manufacturer',
  'of', 'plastics', 'and', 'chemicals', '.', 'another', 'industry', 'powerhouse', ',', 'w', '.',
  'r', '.', 'grace', '&', 'lt', ';', 'gra', '>', 'has', 'divested', 'its', 'retailing', ',',
  'restaurant', 'and', 'fertilizer', 'businesses', 'to', 'raise', 'cash', 'for', 'chemical',
  'acquisitions', '.', 'but', 'some', 'experts', 'worry', 'that', 'the', 'chemical', 'industry',
  'may', 'be', 'headed', 'for', 'trouble', 'if', 'companies', 'continue', 'turning', 'their',
  'back', 'on', 'the', 'manufacturing', 'of', 'staple', 'petrochemical', 'commodities', ',', 'such',
  'as', 'ethylene', ',', 'in', 'favor', 'of', 'more', 'profitable', 'specialty', 'chemicals',
  'that', 'are', 'custom', '-', 'designed', 'for', 'a', 'small', 'group', 'of', 'buyers', '.', '"',
  'companies', 'like', 'dupont', '&', 'lt', ';', 'dd', '>', 'and', 'monsanto', 'co', '&', 'lt', ';',
  'mtc', '>', 'spent', 'the', 'past', 'two', 'or', 'three', 'years', 'trying', 'to', 'get', 'out',
  'of', 'the', 'commodity', 'chemical', 'business', 'in', 'reaction', 'to', 'how', 'badly', 'the',
  'market', 'had', 'deteriorated', ',"', 'dosher', 'said', '.', '"', 'but', 'i', 'think', 'they',
  'will', 'eventually', 'kill', 'the', 'margins', 'on', 'the', 'profitable', 'chemicals', 'in',
  'the', 'niche', 'market', '."', 'some', 'top', 'chemical', 'executives', 'share', 'the',
  'concern', '.', '"', 'the', 'challenge', 'for', 'our', 'industry', 'is', 'to', 'keep', 'from',
  'getting', 'carried', 'away', 'and', 'repeating', 'past', 'mistakes', ',"', 'gaf', "'", 's',
  'heyman', 'cautioned', '.', '"', 'the', 'shift', 'from', 'commodity', 'chemicals', 'may', 'be',
  'ill', '-', 'advised', '.', 'specialty', 'businesses', 'do', 'not', 'stay', 'special', 'long',
  '."', 'houston', '-', 'based', 'cain', 'chemical', ',', 'created', 'this', 'month', 'by', 'the',
  'sterling', 'investment', 'banking', 'group', ',', 'believes', 'it', 'can', 'generate', '700',
  'mln', 'dlrs', 'in', 'annual', 'sales', 'by', 'bucking', 'the', 'industry', 'trend', '.',
  'chairman', 'gordon', 'cain', ',', 'who', 'previously', 'led', 'a', 'leveraged', 'buyout', 'of',
  'dupont', "'", 's', 'conoco', 'inc', "'", 's', 'chemical', 'business', ',', 'has', 'spent', '1',
  '.', '1', 'billion', 'dlrs', 'since', 'january', 'to', 'buy', 'seven', 'petrochemical', 'plants',
  'along', 'the', 'texas', 'gulf', 'coast', '.', 'the', 'plants', 'produce', 'only', 'basic',
  'commodity', 'petrochemicals', 'that', 'are', 'the', 'building', 'blocks', 'of', 'specialty',
  'products', '.', '"', 'this', 'kind', 'of', 'commodity', 'chemical', 'business', 'will', 'never',
  'be', 'a', 'glamorous', ',', 'high', '-', 'margin', 'business', ',"', 'cain', 'said', ',',
  'adding', 'that', 'demand', 'is', 'expected', 'to', 'grow', 'by', 'about', 'three', 'pct',
  'annually', '.', 'garo', 'armen', ',', 'an', 'analyst', 'with', 'dean', 'witter', 'reynolds', ',',
  'said', 'chemical', 'makers', 'have', 'also', 'benefitted', 'by', 'increasing', 'demand', 'for',
  'plastics', 'as', 'prices', 'become', 'more', 'competitive', 'with', 'aluminum', ',', 'wood',
  'and', 'steel', 'products', '.', 'armen', 'estimated', 'the', 'upturn', 'in', 'the', 'chemical',
  'business', 'could', 'last', 'as', 'long', 'as', 'four', 'or', 'five', 'years', ',', 'provided',
  'the', 'u', '.', 's', '.', 'economy', 'continues', 'its', 'modest', 'rate', 'of', 'growth', '.',
  '<END>'],
 ['<START>', 'turkey', 'calls', 'for', 'dialogue', 'to', 'solve', 'dispute', 'turkey', 'said',
  'today', 'its', 'disputes', 'with', 'greece', ',', 'including', 'rights', 'on', 'the',
  'continental', 'shelf', 'in', 'the', 'aegean', 'sea', ',', 'should', 'be', 'solved', 'through',
  'negotiations', '.', 'a', 'foreign', 'ministry', 'statement', 'said', 'the', 'latest', 'crisis',
  'between', 'the', 'two', 'nato', 'members', 'stemmed', 'from', 'the', 'continental', 'shelf',
  'dispute', 'and', 'an', 'agreement', 'on', 'this', 'issue', 'would', 'effect', 'the', 'security',
  ',', 'economy', 'and', 'other', 'rights', 'of', 'both', 'countries', '.', '"', 'as', 'the',
  'issue', 'is', 'basicly', 'political', ',', 'a', 'solution', 'can', 'only', 'be', 'found', 'by',
  'bilateral', 'negotiations', ',"', 'the', 'statement', 'said', '.', 'greece', 'has', 'repeatedly',
  'said', 'the', 'issue', 'was', 'legal', 'and', 'could', 'be', 'solved', 'at', 'the',
  'international', 'court', 'of', 'justice', '.', 'the', 'two', 'countries', 'approached', 'armed',
  'confrontation', 'last', 'month', 'after', 'greece', 'announced', 'it', 'planned', 'oil',
  'exploration', 'work', 'in', 'the', 'aegean', 'and', 'turkey', 'said', 'it', 'would', 'also',
  'search', 'for', 'oil', '.', 'a', 'face', '-', 'off', 'was', 'averted', 'when', 'turkey',
  'confined', 'its', 'research', 'to', 'territorrial', 'waters', '.', '"', 'the', 'latest',
  'crises', 'created', 'an', 'historic', 'opportunity', 'to', 'solve', 'the', 'disputes', 'between',
  'the', 'two', 'countries', ',"', 'the', 'foreign', 'ministry', 'statement', 'said', '.', 'turkey',
  "'", 's', 'ambassador', 'in', 'athens', ',', 'nazmi', 'akiman', ',', 'was', 'due', 'to', 'meet',
  'prime', 'minister', 'andreas', 'papandreou', 'today', 'for', 'the', 'greek', 'reply', 'to', 'a',
  'message', 'sent', 'last', 'week', 'by', 'turkish', 'prime', 'minister', 'turgut', 'ozal', '.',
  'the', 'contents', 'of', 'the', 'message', 'were', 'not', 'disclosed', '.', '<END>']]
</pre>
## Question 1.1: Implement `distinct_words` [code] (2 points)



Write a method to work out the distinct words (word types) that occur in the corpus. You can do this with for loops, but it's more efficient to do it with Python list comprehensions. In particular, [this](https://coderwall.com/p/rcmaea/flatten-a-list-of-lists-in-one-line-in-python) may be useful to flatten a list of lists. If you're not familiar with Python list comprehensions in general, here's [more information](https://python-3-patterns-idioms-test.readthedocs.io/en/latest/Comprehensions.html).



Your returned corpus_words should be sorted. You can use python's sorted function for this.



You may find it useful to use [Python sets](https://www.w3schools.com/python/python_sets.asp) to remove duplicate words.



```python
def distinct_words(corpus):
    """ Determine a list of distinct words for the corpus.
        Params:
            corpus (list of list of strings): corpus of documents
        Return:
            corpus_words (list of strings): sorted list of distinct words across the corpus
            num_corpus_words (integer): number of distinct words across the corpus
    """
    corpus_words = []
    num_corpus_words = -1

    # ------------------
    # Write your implementation here.

    for i in (corpus):
        corpus_words += i
    corpus_words = list(set(corpus_words))
    corpus_words.sort()
    num_corpus_words = len(corpus_words)

    # ------------------

    return corpus_words, num_corpus_words
```

Run below for sanity check - note that this is not an exhaustive check for correctness.



```python
# Define toy corpus
test_corpus = ["{} All that glitters isn't gold {}".format(START_TOKEN, END_TOKEN).split(" "), "{} All's well that ends well {}".format(START_TOKEN, END_TOKEN).split(" ")]
test_corpus_words, num_corpus_words = distinct_words(test_corpus)

# Correct answers
ans_test_corpus_words = sorted([START_TOKEN, "All", "ends", "that", "gold", "All's", "glitters", "isn't", "well", END_TOKEN])
ans_num_corpus_words = len(ans_test_corpus_words)

# Test correct number of words
assert(num_corpus_words == ans_num_corpus_words), "Incorrect number of distinct words. Correct: {}. Yours: {}".format(ans_num_corpus_words, num_corpus_words)

# Test correct words
assert (test_corpus_words == ans_test_corpus_words), "Incorrect corpus_words.\nCorrect: {}\nYours:   {}".format(str(ans_test_corpus_words), str(test_corpus_words))

# Print Success
print ("-" * 80)
print("Passed All Tests!")
print ("-" * 80)
```

<pre>
--------------------------------------------------------------------------------
Passed All Tests!
--------------------------------------------------------------------------------
</pre>
## Question 1.2: Implement `compute_co_occurrence_matrix` [code] (3 points)

Write a method that constructs a co-occurrence matrix for a certain window-size  $n$  (with a default of 4), considering words  $n$  before and  $n$  after the word in the center of the window. Here, we start to use `numpy (np)` to represent vectors, matrices, and tensors. If you're not familiar with NumPy, there's a NumPy tutorial in the second half of this [Stanford cs231n Python NumPy tutorial](https://cs231n.github.io/python-numpy-tutorial/).



```python
def compute_co_occurrence_matrix(corpus, window_size=4):
    """ Compute co-occurrence matrix for the given corpus and window_size (default of 4).

        Note: Each word in a document should be at the center of a window. Words near edges will have a smaller
              number of co-occurring words.

              For example, if we take the document "<START> All that glitters is not gold <END>" with window size of 4,
              "All" will co-occur with "<START>", "that", "glitters", "is", and "not".

        Params:
            corpus (list of list of strings): corpus of documents
            window_size (int): size of context window
        Return:
            M (a symmetric numpy matrix of shape (number of unique words in the corpus , number of unique words in the corpus)):
                Co-occurence matrix of word counts.
                The ordering of the words in the rows/columns should be the same as the ordering of the words given by the distinct_words function.
            word2ind (dict): dictionary that maps word to index (i.e. row/column number) for matrix M.
    """
    words, num_words = distinct_words(corpus)
    M = None
    word2ind = {}

    # ------------------
    # Write your implementation here.
    M = np.zeros((num_words, num_words))

    for idx, word in enumerate(words):
        word2ind[word] = idx

    for doc in (corpus):
        for i, main_word in enumerate(doc):
            mw_idx = word2ind[main_word]
            temp_list = doc[max(0, i - window_size) : i] + doc[i+1 : i+1+window_size]
            for text in temp_list:
                temp_idx = word2ind[text]
                M[mw_idx, temp_idx] += 1

    # ------------------

    return M, word2ind
```

Run below for sanity check - note that this is not an exhaustive check for correctness.



```python
# Define toy corpus and get student's co-occurrence matrix
test_corpus = ["{} All that glitters isn't gold {}".format(START_TOKEN, END_TOKEN).split(" "), "{} All's well that ends well {}".format(START_TOKEN, END_TOKEN).split(" ")]
M_test, word2ind_test = compute_co_occurrence_matrix(test_corpus, window_size=1)

# Correct M and word2ind
M_test_ans = np.array(
    [[0., 0., 0., 0., 0., 0., 1., 0., 0., 1.,],
     [0., 0., 1., 1., 0., 0., 0., 0., 0., 0.,],
     [0., 1., 0., 0., 0., 0., 0., 0., 1., 0.,],
     [0., 1., 0., 0., 0., 0., 0., 0., 0., 1.,],
     [0., 0., 0., 0., 0., 0., 0., 0., 1., 1.,],
     [0., 0., 0., 0., 0., 0., 0., 1., 1., 0.,],
     [1., 0., 0., 0., 0., 0., 0., 1., 0., 0.,],
     [0., 0., 0., 0., 0., 1., 1., 0., 0., 0.,],
     [0., 0., 1., 0., 1., 1., 0., 0., 0., 1.,],
     [1., 0., 0., 1., 1., 0., 0., 0., 1., 0.,]]
)
ans_test_corpus_words = sorted([START_TOKEN, "All", "ends", "that", "gold", "All's", "glitters", "isn't", "well", END_TOKEN])
word2ind_ans = dict(zip(ans_test_corpus_words, range(len(ans_test_corpus_words))))

# Test correct word2ind
assert (word2ind_ans == word2ind_test), "Your word2ind is incorrect:\nCorrect: {}\nYours: {}".format(word2ind_ans, word2ind_test)

# Test correct M shape
assert (M_test.shape == M_test_ans.shape), "M matrix has incorrect shape.\nCorrect: {}\nYours: {}".format(M_test.shape, M_test_ans.shape)

# Test correct M values
for w1 in word2ind_ans.keys():
    idx1 = word2ind_ans[w1]
    for w2 in word2ind_ans.keys():
        idx2 = word2ind_ans[w2]
        student = M_test[idx1, idx2]
        correct = M_test_ans[idx1, idx2]
        if student != correct:
            print("Correct M:")
            print(M_test_ans)
            print("Your M: ")
            print(M_test)
            raise AssertionError("Incorrect count at index ({}, {})=({}, {}) in matrix M. Yours has {} but should have {}.".format(idx1, idx2, w1, w2, student, correct))

# Print Success
print ("-" * 80)
print("Passed All Tests!")
print ("-" * 80)
```

<pre>
--------------------------------------------------------------------------------
Passed All Tests!
--------------------------------------------------------------------------------
</pre>
## Question 1.3: Implement `reduce_to_k_dim` [code] (1 point)



Construct a method that performs dimensionality reduction on the matrix to produce k-dimensional embeddings. Use SVD to take the top k components and produce a new matrix of k-dimensional embeddings.



**Note:** Please use [sklearn.decomposition.TruncatedSVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) for SVD. While there are numbers of implementation of SVD (e.g. numpy, scipy, ...), it is challenging to apply full SVD to large corpora due to its high asymptotic complexity. Here we only need top $k$ vector components for relative small $k$, which is the case called Truncated SVD, and some iterative algorithms for this scale reasonably. The above implementation by sklean is an efficient randomized algorithm for calculating large-scale Truncated SVD.



```python
def reduce_to_k_dim(M, k=2):
    """ Reduce a co-occurence count matrix of dimensionality (num_corpus_words, num_corpus_words)
        to a matrix of dimensionality (num_corpus_words, k) using the following SVD function from Scikit-Learn:
            - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html

        Params:
            M (numpy matrix of shape (number of unique words in the corpus , number of unique words in the corpus)): co-occurence matrix of word counts
            k (int): embedding size of each word after dimension reduction
        Return:
            M_reduced (numpy matrix of shape (number of corpus words, k)): matrix of k-dimensioal word embeddings.
                    In terms of the SVD from math class, this actually returns U * S
    """
    n_iters = 10     # Use this parameter in your call to `TruncatedSVD`
    M_reduced = None
    print("Running Truncated SVD over %i words..." % (M.shape[0]))

    # ------------------
    # Write your implementation here.
    M_reduced = TruncatedSVD(k, n_iter=n_iters).fit_transform(M)
    # ------------------

    print("Done.")
    return M_reduced
```

Run below for sanity check - note that the code below only check shapes.



```python
# Define toy corpus and run student code
test_corpus = ["{} All that glitters isn't gold {}".format(START_TOKEN, END_TOKEN).split(" "), "{} All's well that ends well {}".format(START_TOKEN, END_TOKEN).split(" ")]
M_test, word2ind_test = compute_co_occurrence_matrix(test_corpus, window_size=1)
M_test_reduced = reduce_to_k_dim(M_test, k=2)

# Test proper dimensions
assert (M_test_reduced.shape[0] == 10), "M_reduced has {} rows; should have {}".format(M_test_reduced.shape[0], 10)
assert (M_test_reduced.shape[1] == 2), "M_reduced has {} columns; should have {}".format(M_test_reduced.shape[1], 2)

# Print Success
print ("-" * 80)
print("Passed All Tests!")
print ("-" * 80)
```

<pre>
Running Truncated SVD over 10 words...
Done.
--------------------------------------------------------------------------------
Passed All Tests!
--------------------------------------------------------------------------------
</pre>
## Question 1.4: Implement `plot_embeddings` [code] (1 point)

Here you will write a function to plot a set of 2D vectors in 2D space. For graphs, we will use Matplotlib (`plt`).



For this example, you may find it useful to adapt [this code](http://web.archive.org/web/20190924160434/https://www.pythonmembers.club/2018/05/08/matplotlib-scatter-plot-annotate-set-text-at-label-each-point/). In the future, a good way to make a plot is to look at the [Matplotlib gallery](https://matplotlib.org/stable/gallery/index.html), find a plot that looks somewhat like what you want, and adapt the code they give.



```python
def plot_embeddings(M_reduced, word2ind, words):
    """ Plot in a scatterplot the embeddings of the words specified in the list "words".
        NOTE: do not plot all the words listed in M_reduced / word2ind.
        Include a label next to each point.

        Params:
            M_reduced (numpy matrix of shape (number of unique words in the corpus , 2)): matrix of 2-dimensioal word embeddings
            word2ind (dict): dictionary that maps word to indices for matrix M
            words (list of strings): words whose embeddings we want to visualize
    """

    # ------------------
    # Write your implementation here.

    for word in words:
        idx = word2ind[word]

        x = M_reduced[idx, 0]
        y = M_reduced[idx, 1]

        plt.scatter(x, y, marker='x', color='red')
        plt.text(x, y, word, fontsize=9)
    plt.show()
    # ------------------
```

Run below for sanity check - the plot produced should look like the "test solution plot" depicted below.




```python
print ("-" * 80)
print ("Outputted Plot:")

M_reduced_plot_test = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1], [0, 0]])
word2ind_plot_test = {'test1': 0, 'test2': 1, 'test3': 2, 'test4': 3, 'test5': 4}
words = ['test1', 'test2', 'test3', 'test4', 'test5']
plot_embeddings(M_reduced_plot_test, word2ind_plot_test, words)

print ("-" * 80)
```

<pre>
--------------------------------------------------------------------------------
Outputted Plot:
</pre>
<pre>
<Figure size 720x360 with 1 Axes>
</pre>
<pre>
--------------------------------------------------------------------------------
</pre>
**Test Plot Solution**



<div align="center">



![test_plot.png](data:image/png;base64,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LpEa2KkeRQgggAACCCCAAAKFCESKO0ksnE4VqYGd6lGEAAIIIIAAAgggUIhApLiTxMLpVJEa2KkeRQgggAACCCCAAAKFCESKO0ksnE4VqYGd6lGEAAIIIIAAAgggUIhApLiTxMLpVJEa2KkeRQgggAACCCCAAAKFCESKO0ksnE4VqYGd6lGEAAIIIIAAAgggUIhApLiTxMLpVJEa2KkeRQgggAACCCCAAAKFCESKO0ksnE4VqYGd6lGEAAIIIIAAAgggUIhApLiTxMLpVJEa2KkeRQgggAACCCCAAAKFCESKO0ksnE4VqYGd6lGEAAIIIIAAAgggUIhApLiTxMLpVJEa2KkeRQgggAACCCCAAAKFCESKO0ksnE4VqYGd6lGEAAIIIIAAAgggUIhApLiTxMLpVJEa2KkeRQgggAACCCCAAAKFCESKO0ksnE4VqYGd6lGEAAIIIIAAAgggUIhApLiTxMLpVJEa2KkeRQgggAACCCCAAAKFCESKO0ksnE4VqYGd6lGEAAIIIIAAAgggUIhApLiTxMLpVJEa2KkeRQgggAACCCCAAAKFCESKO0ksnE4VqYGd6lGEAAIIIIAAAgggUIhApLiTxMLpVJEa2KkeRQgggAACCCCAAAKFCESKO0ksnE4VqYGd6lGEAAIIIIAAAgggUIhApLiTxMLpVJEa2KkeRQgggAACCCCAAAKFCESKO0ksnE4VqYGd6lGEAAIIIIAAAgggUIhApLiTxMLpVJEa2KkeRQgggAACCCCAAAKFCESKO0ksnE4VqYGd6lGEAAIIIIAAAgggUIhApLiTxMLpVJEa2KkeRQgggAACCCCAAAKFCESKO0ksnE4VqYGd6lGEAAIIIIAAAgggUIhApLiTxMLpVJEa2KkeRQgggAACCCCAAAKFCESKO0ksnE4VqYGd6lGEAAIIIIAAAgggUIhApLiTxMLpVJEa2KkeRQgggAACCCCAAAKFCESKO0ksnE4VqYGd6lGEAAIIIIAAAgggUIhApLiTxMLpVJEa2KkeRQgggAACCCCAAAKFCESKO0ksnE4VqYGd6lGEAAIIIIAAAgggUIhApLiTxMLpVJEa2KkeRQgggAACCCCAAAKFCESKO0ksnE4VqYGd6lGEAAIIIIAAAgggUIhApLiTxMLpVJEa2KkeRQgggAACCCCAAAKFCESKO0ksnE4VqYGd6lGEAAIIIIAAAgggUIhApLiTxMLpVJEa2KkeRQgggAACCCCAAAKFCESKO0ksnE4VqYGd6lGEAAIIIIAAAgggUIhApLiTxMLpVJEa2KkeRQgggAACCCCAAAKFCESKO0ksnE4VqYGd6lGEAAIIIIAAAgggUIhApLiTxMLpVJEa2KkeRQgggAACCCCAAAKFCESKO0ksnE4VqYGd6lGEAAIIIIAAAgggUIhApLiTxMLpVJEa2KkeRQgggAACCCCAAAKFCESKO0ksnE4VqYGd6lGEAAIIIIAAAgggUIhApLiTxMLpVJEa2KkeRQgggAACCCCAAAKFCESKO0ksnE4VqYGd6lGEAAIIIIAAAgggUIhApLiTxMLpVJEa2KkeRQgggAACCCCAAAKFCESKO0ksnE4VqYGd6lGEAAIIIIAAAgggUIhApLiTxMLpVJEa2KkeRQgggAACCCCAAAKFCESKO0ksnE4VqYGd6lGEAAIIIIAAAgggUIhApLiTxMLpVJEa2KkeRQgggAACCCCAAAKFCESKO0ksnE4VqYGd6lGEAAIIIIAAAgggUIhApLiTxMLpVJEa2KkeRQgggAACCCCAAAKFCESKO0ksnE4VqYGd6lGEAAIIIIAAAgggUIhApLiTxMLpVJEa2KkeRQgggAACCCCAAAKFCESKO0ksnE4VqYGd6lGEAAIIIIAAAgggUIhApLiTxMLpVJEa2KkeRQgggAACCCCAAAKFCESKO0ksnE4VqYGd6lGEAAIIIIAAAgggUIhApLiTxMLpVJEa2KkeRQgggAACCCCAAAKFCESKO0ksnE4VqYGd6lGEAAIIIIAAAgggUIhApLiTxMLpVJEa2KkeRQgggAACCCCAAAKFCESKO0ksnE4VqYGd6lGEAAIIIIAAAgggUIhApLiTxMLpVJEa2KkeRQgggAACCCCAAAKFCESKO0ksnE4VqYGd6lGEAAIIIIAAAgggUIhApLiTxMLpVJEa2KkeRQgggAACCCCAAAKFCESKO0ksnE4VqYGd6lGEAAIIIIAAAgggUIhApLiTxMLpVJEa2KkeRQgggAACCCCAAAKFCESKO0ksnE4VqYGd6lGEAAIIIIAAAgggUIhApLiTxMLpVJEa2KkeRQgggAACCCCAAAKFCESKO0ksnE4VqYGd6lGEAAIIIIAAAgggUIhApLiTxMLpVJEa2KkeRQgggAACCCCAAAKFCESKO0ksnE4VqYGd6lGEAAIIIIAAAgggUIhApLiTxMLpVJEa2KkeRQgggAACCCCAAAKFCESKO0ksnE4VqYGd6lGEAAIIIIAAAgggUIhApLiTxMLpVJEa2KkeRQgggAACCCCAAAKFCESKO0ksnE4VqYGd6lGEAAIIIIAAAgggUIhApLiTxMLpVJEa2KkeRQgggAACCCCAAAKFCESKOxuaWOzevdu2bdtmGzdutK+++ip/vXfvXuvQoYN1797devToYT179rSOHTu6Tb1z507bvHmzbdiwwdatW2c7duywdu3aWdu2bfPj9Tydp/rQees5IjVwPfXnOQgggAACCCCAAAKNEYgUdzY0sdi0aZN9/vnn9uGHH9qMGTPy10osevXqZUOGDLGRI0faueeea3369HFbUsnIRx99ZPPnz7d33nnHVq9ebV26dLETTjghP/7kk0+2ESNG2LBhw/L5evfu7Z7ncIWRGvhwdeXnCCCAAAIIIIAAAq0nECnubFhioQRCScXMmTPt/ffft2XLluWRC402dOrUybp165YTgokTJ9rZZ5+dkwWNRtQeK1assDlz5tisWbPszTffzCMXAwYMyCMdehyJRa0WXyOAAAIIIIAAAgiULkBi0cQWUlKhj3nz5tmjjz5qa9assQsvvNDOPPPMPOLwxRdf2LRp0/IUqBtuuMFGjx5tJ554Yk44al+qSiw++OADW7x4sXXt2tUmTJhgSi50MBWqVouvEUAAAQQQQAABBEoXILFoYgtpbcX27dvzSMODDz5oe/bssdtuuy0nF5rKtGDBAnvkkUfyCMSVV15p5513Xk46tN6i9qgSCyUVSkb69etn1157bZ72VPu45n4dqYGba8HzEUAAAQQQQAABBI6eQKS4syFTobZs2WKrVq3K05iee+65PBKhxGLMmDHWvn17W7JkiT3//PP25Zdf2imnnGKDBw+2sWPH2qmnnnpAK5JYHMDBNwgggAACCCCAAALHuACJRRMbcP369bZ06dK86Hr69Ol5mtOtt95qo0aNymf67LPPTOV6jHZ6Ov300+2qq66ygQMHHvBKVWJROxXq0ksvzY/TAm5NjdJCcH2u3S3qgJPUfKNdprSgvPrQ9zo0gjJ16tS8u9Sdd96ZR1ZqnsaXCCCAAAIIIIAAAgi0iACJRRMZ165dawsXLsy7QWnhthZZT548Oe/gpFNppOLdd9+1jz/+OO/0pJGKSZMm5UXctS9VJRZavK1ERNvO9u/fP49yaKvas846y8aNG2eDBg3KyUW1W1TtOWq/1u+1aNGi/Lvps77XoURIoyhDhw61u+++2y6++OLap/E1AggggAACCCCAAAItIkBi0URGbQurUQYlF9ouVtOdtEhb28LqWLlypc2dOzcvyNbOUX379nXXTlQJipKT2bNn5ySkc+fOeQcp7S6lbWq1Za3Oq52lTjrppEP+piQWh+ThhwgggAACCCCAAAJHWYDEoonALZVYbN26NY8qaKRC05e+/vrrnFTohnsaYdDIh74+44wz7Lrrrjvsom6mQjWxIXk4AggggAACCCCAQIsKkFg0kbMaadCN8ZozFUq7S2kNhrau1TQnraPQh26cp9EQNcxrr72Wp1rdfvvteQF4E3/V/PBIDVxP/XkOAggggAACCCCAQGMEIsWdDdkVqqUWbyuh0CiFDk19atOmTf5QsqGRCq27eOCBB/L9LO69914bP358XT0iUgPXBcCTEEAAAQQQQAABBBoiECnubEhiUbvd7LPPPmtaF1HPdrO6/0WVWNTu+qQpTXqNt956y37/+99bhw4d7Oc//7lpx6h6jkgNXE/9eQ4CCCCAAAIIIIBAYwQixZ0NSSw0hWnbtm0H3CBvypQp7g3yrrjiiv03yNPdt2uPXbt25fOorFq0ra+15mLZsmX5/P/85z9NN9bTNrHaIaqeI1ID11N/noMAAggggAACCCDQGIFIcWdDEgtNYdLHvHnz8h2216xZk4N+3adCCYJ2gpo2bVqewlS7W5RGJTT60KlTJ9MdunX3bt1xW9OeqkPTofS97oWh8+jnp512mt100002fPjw6mFN+hypgZtUcR6MAAIIIIAAAggg0FCBSHFnQxILtY4SCwX+M2bMyAu4NcKghKBKHrp165a3ib366qtNX8+fP982b96cF2L369fPBgwYkL+fOXNmvoGdEgj9XGstdG6NivTu3TtvNztixIicVOh+GfUckRq4nvrzHAQQQAABBBBAAIHGCESKOxuWWKhpqpEF3dlaCYJGGZQU6G7ZgwcPzknB6NGjc5nuU6EpTkoOqsRC280KX89fvnx5Pp8Si/bt2+eRDY2AaMG2zqXpUB07dqyrR0Rq4LoAeBICCCCAAAIIIIBAQwQixZ0NTSyqNRK6B4W2iFWioEPTnTRKobtnKyHQoZ2k9PjaqVBauL1u3bqcUGjNhn5e7QylBKNr16551ELn0na0KqvniNTA9dSf5yCAAAIIIIAAAgg0RiBS3NnQxKIxzdP8V4nUwM3X4AwIIIAAAggggAACR0sgUtxJYuH0kkgN7FSPIgQQQAABBBBAAIFCBCLFnSQWTqeK1MBO9ShCAAEEEEAAAQQQKEQgUtxJYuF0qkgN7FSPIgQQQAABBBBAAIFCBCLFnSQWTqeK1MBO9ShCAAEEEEAAAQQQKEQgUtxJYuF0qkgN7FSPIgQQQAABBBBAAIFCBCLFnSQWTqeK1MBO9ShCAAEEEEAAAQQQKEQgUtxJYuF0qkgN7FSPIgQQQAABBBBAAIFCBCLFnSQWTqeK1MBO9ShCAAEEEEAAAQQQKEQgUtxJYuF0qkgN7FSPIgQQQAABBBBAAIFCBCLFnSQWTqeK1MBO9ShCAAEEEEAAAQQQKEQgUtxJYuF0qkgN7FSPIgQQQAABBBBAAIFCBCLFnSQWTqeK1MBO9ShCAAEEEEAAAQQQKEQgUtxJYuF0qkgN7FSPIgQQQAABBBBAAIFCBCLFnSQWTqeK1MBO9ShCAAEEEEAAAQQQKEQgUtxJYuF0qkgN7FSPIgQQQAABBBBAAIFCBCLFnSQWTqeK1MBO9ShCAAEEEEAAAQQQKEQgUtxJYuF0qkgN7FSPIgQQQAABBBBAAIFCBCLFnSQWTqeK1MBO9ShCAAEEEEAAAQQQKEQgUtxJYuF0qkgN7FSPIgQQQAABBBBAAIFCBCLFnSQWTqeK1MBO9ShCAAEEEEAAAQQQKEQgUtxJYuF0qkgN7FSPIgQQQAABBBBAAIFCBCLFnSQWTqeK1MBO9ShCAAEEEEAAAQQQKEQgUtxJYuF0qkgN7FSPIgQQQAABBBBAAIFCBCLFnSQWTqeK1MBO9ShCAAEEEEAAAQQQKEQgUtxJYuF0qkgN7FSPIgQQQAABBBBAAIFCBCLFnSQWTqeK1MBO9ShCAAEEEEAAAQQQKEQgUtxJYuF0qkgN7FSPIgQQQAABBBBAAIFCBCLFnSQWTqeK1MBO9ShCAAEEEEAAAQQQKEQgUtxJYuF0qkgN7FSPIgQQQAABBBBAAIFCBCLFnSQWTqeK1MBO9ShCAAEEEEAAAQQQKEQgUtxJYuF0qkgN7FSPIgQQQAABBBBAAIFCBCLFnSQWTqeK1MBO9ShCAAEEEEAAAQQQKEQgUtxJYuF0qkgN7FSPIgQQQAABBBBAAIFCBCLFnSQWTqeK1MBO9ShCAAEEEEAAAQQQKEQgUtxJYuF0qkgN7FSPIgQQQAABBBBAAIFCBCLFnSQWTqeK1MBO9ShCAAEEEEAAAQQQKEQgUtxJYuF0qkgN7FSPIgQQQAABBBBAAIFCBCLFnSQWTqeK1MBO9ShCAAEEEEAAAQQQKEQgUtxJYuF0qkgN7FSPIgQQQAABBBBAAIFCBCLFnSQWTqeK1MBO9ShCAAEEEEAAAQQQKEQgUtxJYuF0qkgN7FSPIgQQQAABBBBAAIFCBCLFnSQWTqeK1MBO9ShCAAEEEEAAAQQQKEQgUtxJYuF0qkgN7FSPIgQQQAABBBBAAIFCBCLFnSQWTqeK1MBO9ShCAAEEEEAAAQQQKEQgUtxJYuF0qkgN7FSPIgQQQAABBBBAAIFCBCLFnSQWTqeK1MBO9ShCAAEEEEAAAQQQKEQgUtxJYuF0qkgN7FSPIgQQQAABBBBAAIFCBCLFnSQWTqeK1MBO9ShCAAEEEEAAAQQQKEQgUtxJYuF0qkgN7FSPIgQQQAABBBBAAIFCBCLFnSQWTqeK1MBO9ShCAAEEEEAAAQQQKEQgUtxJYuF0qkgN7FSPIgQQQAABBBBAAIFCBCLFnSQWTqeK1MBO9ShCAAEEEEAAAQQQKEQgUtxJYuF0qkgN7FSPIgQQQAABBBBAAIFCBCLFnSQWTqeK1MBO9ShCAAEEEEAAAQQQKEQgUtxJYuF0qkgN7FSPIgQQQAABBBBAAIFCBCLFnSQWTqeK1MBO9ShCAAEEEEAAAQQQKEQgUtxJYuF0qkgN7FSPIgQQQAABBBBAAIFCBCLFnSQWTqeK1MBO9ShCAAEEEEAAAQQQKEQgUtxJYuF0qkgN7FSPIgQQQAABBBBAAIFCBCLFnSQWTqeK1MBO9ShCAAEEEEAAAQQQKEQgUtxJYuF0qkgN7FSPIgQQQAABBBBAAIFCBCLFnSQWTqeK1MBO9ShCAAEEEEAAAQQQKEQgUtxJYuF0qkgN7FSPIgQQQAABBBBAAIFCBCLFnSQWTqeK1MBO9ShCAAEEEEAAAQQQKEQgUtxJYuF0qkgN7FSPIgQQQAABBBBAAIFCBCLFnSQWTqeK1MBO9ShCAAEEEEAAAQQQKEQgUtxJYuF0qkgN7FSPIgQQ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eXjwwQfDggULwgMPPBBFBlWHmNh1113D0UcfHXbaaaew7rrrjgmF8ar29ttvD+edd1647777wiabbBJmzpwZD11nnXWimJg3b1447rjjwg477DDeKbrKr6mCuyqwB0lAAhKQgAQkIAEJDIVATX7nlAoLVioWLVoU7rrrrnDNNdeEhQsXjokHRMd2220XjjrqqDB//vyw5ZZbRrHRqUavuuqq8OUvfzkKi3322SdstdVW8XCFRSdqvicBCUhAAhKQgAQkUCoBhUWXNfPEE0+Em2++OTz00ENhyZIlYaONNgq77757/PQ999wTQ5pmz54ddtxxx3DQQQeFbbfdtuOZr7766vCVr3wlfu5Tn/pU2G+//eLxhkJ1xOabEpCABCQgAQlIQAKFElBYdFkxhCxdcsklcdWC1Ynddtst7LvvvvHTt912W1x5YM8EKw8nn3xyIJSpU2LV46tf/Wo85Etf+lJc7eh0fK/v1VTBvTLwcxKQgAQkIAEJSEACU0+gJr9zSkOhbr311vD9738/7qs48cQT4wrD1ltvHWvoySefDLx/2WWXxRCo0047bWwFYrwqVFiMR8Z8CUhAAhKQgAQkIIFRJKCw6LLWbrzxxvCtb30rvP766+H0008Phx56aFhvvfXip1999dXA++ecc07gDk+f/exn4/udTo2waA+F4k5SbOImpIoN4euvv348X6fzrO69mip4dWX1fQlIQAISkIAEJCCB4RGoye+c0hWL6667LnzjG98I3Hb2c5/7XDj88MNX2bzN+2eddVa8zeznP//5cMQRR3SsVfZYpM3b7K9g9QNhwcbvvffeO+yyyy5xQzgiYzKppgqeDAc/KwEJSEACEpCABCQwtQRq8ju7Ehbc3empp56Km6a5mxPPpuiUWEHgdrDcHvbss8+OKwhf/OIXw5FHHjn2McQGwuJrX/tazGt/f+zAlj8InTr33HPD/fffH59/weoE34dVEJ6Hwe1r2QQ+Z86ceE02dXdKr732WiwTz9fgh/9J9957b9wbwnM2zjjjjHDggQd2Oo3vSUACEpCABCQgAQlIoCcCa5yw4DkUV1xxRdxsTQgToU2dEisJ+++/f3j22WfD+eefH8OUcsLh2muvDV//+tfjqb7whS+sIjxy5+che+luUhtvvHF8qB6ih+9HWBWhUKecckrgVrSIAp6N0Sk999xzUaSwyRyx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FM8sCUhAAhKQgAQkIIFCCNTkdyosMkZVagWvXLkyvPnmm+G1114Lr7zySpg2bVrYYIMNwjrrrJMpRe9ZXGPFihXh9ddfj9daa621wvrrrx/WXnvteNJXX301PP/88+HFF18My5cvD2+88UbM532O4zvNnj07zJw5s/cv4SclIAEJSEACEpDAFBAoxZ/Ch1q6dGm4+eabw3/+539Gn+sjH/lImD9/fpgxY0b0p97xjndE3wqfbxSSwiJTS6UKCxx4RAVO/cKFC8P06dPDTjvtFDbddNNMKXrP4hovvfRSWLJkSXjhhReicJkzZ04UC5z1iSeeiI3g/vvvD88880wUOeQjJrbbbruwyy67hL322itsueWWZJskIAEJSEACEpBAMQRK8afwoe66665w3XXXhcsuuyz6XYiKd77znWHDDTcMO++8czjiiCOib8Uk7yiIC4VFxsxLFRasILBSgai49dZb4woCBofT38/ENZ599tmwaNGieC1WIfbbb7+w1VZbxctwfRrBAw88EIXHsmXLYj5Gz+oJYufd7353fGUVg3yTBCQgAQlIQAISKIFAKf7U008/PSYsLr/88jFhwYQxqxnbb799eM973hN22223MGvWrL5HqExFXSgsMlRLFRZpyeyee+4JV155ZVh33XXDhz/84bD77rtnStF7Fstyjz/+eHjooYfCvffeGzbaaKNw/PHHhx133DGedPHixeGRRx6Jqxp8B1ZOWFYkb8GCBdHwTzrppLhqgeLmGJMEJCABCUhAAhIogUAp/hQTs0SGEAr1wx/+MIahEwqF73TDDTfE8PcDDzww+nlMIuOPlZ4UFpkaKlVYEJr02GOPBb7fL37xi+jQf+ADHwh77rlnjMNjiezll1+O4VI4+iTyULkpRo/VA1YkUMkcy36KlFiZ2GyzzWL+3XffHRAw/PD5o48+Ohr2JptsEldKaJQICowc4cD1fve734Vzzz030FAQPKxycPx6662XLuGrBCQgAQlIQAISGCqBUvwp/Cv2Utx+++3he9/7XvTZTj/99Lin4qc//WlgInffffeN/tcOO+wQNt5446Fy6+biCosMpVKFRQpBQsWibnHuWa2YN29e3NeAaHj44Yej+mXzNQmDZSntyCOPDHPnzo3/I05+85vfxBUJjkNcIEBQw4cffnj8/9prrw133HFHXIXgvKxWEPd30EEH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)

</div>


## Question 1.5: Co-Occurrence Plot Analysis [written] (3 points)



Now we will put together all the parts you have written! We will compute the co-occurrence matrix with fixed window of 4 (the default window size), over the Reuters "crude" (oil) corpus. Then we will use `TruncatedSVD` to compute 2-dimensional embeddings of each word. `TruncatedSVD` returns $US$, so we need to normalize the returned vectors, so that all the vectors will appear around the unit circle (therefore closeness is directional closeness).



**Note:** The line of code below that does the normalizing uses the NumPy concept of broadcasting. If you don't know about broadcasting, check out [Computation on Arrays: Broadcasting by Jake VanderPlas](https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html).



Run the below cell to produce the plot. It'll probably take a few seconds to run. What clusters together in 2-dimensional embedding space? What doesn't cluster together that you might think should have? **Note:** "bpd" stands for "barrels per day" and is a commonly used abbreviation in crude oil topic articles.



```python
# -----------------------------
# Run This Cell to Produce Your Plot
# ------------------------------
reuters_corpus = read_corpus()
M_co_occurrence, word2ind_co_occurrence = compute_co_occurrence_matrix(reuters_corpus)
M_reduced_co_occurrence = reduce_to_k_dim(M_co_occurrence, k=2)

# Rescale (normalize) the rows to make them each of unit-length
M_lengths = np.linalg.norm(M_reduced_co_occurrence, axis=1)
M_normalized = M_reduced_co_occurrence / M_lengths[:, np.newaxis] # broadcasting

words = ['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'iraq']

plot_embeddings(M_normalized, word2ind_co_occurrence, words)
```

<pre>
Running Truncated SVD over 8185 words...
Done.
</pre>
<pre>
<Figure size 720x360 with 1 Axes>
</pre>




---





**Write your answer here. You can write in either Korean or English.**



매우 넓은 관점에서 보면 위의 단어들은 모두 촘촘하게 붙어있다. 이는 단어들이 모두 로이터의 원유 관련 기사에서 온 것이므로 관련성이 있는 것은 당연하다. 그외에 위 표를 보면 쿠웨이트, 에콰도르, 아리크와 같은 국가명들은 근접하여 붙어있는 것을 알 수 있다. BPD의 경우 단위이므로 다른 단어들과 연관성이 떨어지는 것은 사실이나, 단어에서 볼 수 있듯 Barrel이라는 단어 그리고 그 뜻이 하루당 생산된 Barrel의 개수라는 의미에서 부터 그 위치가 Output과 가깝다는 것을 설명할 수 있다.



---





# Part 2: Prediction-Based Word Vectors (15 points)

As discussed in class, more recently prediction-based word vectors have demonstrated better performance, such as word2vec and GloVe (which also utilizes the benefit of counts). Here, we shall explore the embeddings produced by GloVe. Please revisit the class notes and lecture slides for more details on the word2vec and GloVe algorithms. If you're feeling adventurous, challenge yourself and try reading [GloVe's original paper](https://nlp.stanford.edu/pubs/glove.pdf).



Then run the following cells to load the GloVe vectors into memory. **Note:** If this is your first time to run these cells, i.e. download the embedding model, it will take a couple minutes to run. If you've run these cells before, rerunning them will load the model without redownloading it, which will take about 1 to 2 minutes.



```python
def load_embedding_model():
    """ Load GloVe Vectors
        Return:
            wv_from_bin: All 400000 embeddings, each lengh 200
    """
    import gensim.downloader as api
    wv_from_bin = api.load("glove-wiki-gigaword-200")
    print("Loaded vocab size %i" % len(wv_from_bin.vocab.keys()))
    return wv_from_bin
wv_from_bin = load_embedding_model()
```

<pre>
[=================================================-] 99.8% 251.7/252.1MB downloadedLoaded vocab size 400000
</pre>
**Note: If you are receiving a "reset by peer" error, rerun the cell to restart the download.**


## Reducing dimensionality of Word Embeddings

Let's directly compare the GloVe embeddings to those of the co-occurrence matrix. In order to avoid running out of memory, we will work with a sample of 10000 GloVe vectors instead. Run the following cells to:



1.   Put 10000 Glove vectors into a matrix $M$

2.   Run `reduce_to_k_dim` (your Truncated SVD function) to reduce the vectors from 200-dimensional to 2-dimensional.



```python
def get_matrix_of_vectors(wv_from_bin, required_words=['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'iraq']):
    """ Put the GloVe vectors into a matrix M.
        Param:
            wv_from_bin: KeyedVectors object; the 400000 GloVe vectors loaded from file
        Return:
            M: numpy matrix shape (num words, 200) containing the vectors
            word2ind: dictionary mapping each word to its row number in M
    """
    import random
    words = list(wv_from_bin.vocab.keys())
    print("Shuffling words ...")
    random.seed(224)
    random.shuffle(words)
    words = words[:10000]
    print("Putting %i words into word2ind and matrix M..." % len(words))
    word2ind = {}
    M = []
    curInd = 0
    for w in words:
        try:
            M.append(wv_from_bin.word_vec(w))
            word2ind[w] = curInd
            curInd += 1
        except KeyError:
            continue
    for w in required_words:
        if w in words:
            continue
        try:
            M.append(wv_from_bin.word_vec(w))
            word2ind[w] = curInd
            curInd += 1
        except KeyError:
            continue
    M = np.stack(M)
    print("Done.")
    return M, word2ind
```

Run the cell below to reduce 200-dimensional word embeddings to $k$ dimensions. This should be quick to run.



```python
M, word2ind = get_matrix_of_vectors(wv_from_bin)
M_reduced = reduce_to_k_dim(M, k=2)

# Rescale (normalize) the rows to make them each of unit-length
M_lengths = np.linalg.norm(M_reduced, axis=1)
M_reduced_normalized = M_reduced / M_lengths[:, np.newaxis] # broadcasting
```

<pre>
Shuffling words ...
Putting 10000 words into word2ind and matrix M...
Done.
Running Truncated SVD over 10010 words...
Done.
</pre>
## Question 2.1: GloVe Plot Analysis [written] (3 points)

Run the cell below to plot the 2D GloVe embeddings for `['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'iraq']`.



What clusters together in 2-dimensional embedding space? What doesn't cluster together that you think should have? How is the plot different from the one generated earlier from the co-occurrence matrix? What is a possible cause for the difference?



```python
words = ['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'iraq']
plot_embeddings(M_reduced_normalized, word2ind, words)
```

<pre>
<Figure size 720x360 with 1 Axes>
</pre>




---





**Write your answer here. You can write in either Korean or English.**



전체적으로도 이전에 비해 매우 넓게 분포하고 있는 단어들은, 세세한 부분에서도 많은 차이를 가진다. 가장 대표적인 것은 쿠웨이트는 국가명임에도 불구하고 다른 국가들과 거리가 있다. 또한 bpd, barrels, output의 거리가 상대적으로 너무 멀리 떨어져 있는 것을 확인할 수 있다.



이처럼 각 단어들 사이의 연관성이 이전에 비해 GloVe를 사용하였을때 더 작게 나타나는 이유는 이는 이전 처럼 로이터 원유 관련 글을 학습 시킨 것이 아니라, 광범위한 글들을 학습시킨 모델을 사용하였기 떄문이라고 판단된다. 즉 Specification이 덜 되어 있고 조금 더 General하여 단어 간 연관성이 덜 나타나는 것으로 판단된다.



---





### Cosine Similarity

Now that we have word vectors, we need a way to quantify the similarity between individual words, according to these vectors. One such metric is cosine-similarity. We will be using this to find words that are "close" and "far" from one another.



We can think of $n$-dimensional vectors as points in $n$-dimensional space. If we take this perspective [$L^1$](https://mathworld.wolfram.com/L1-Norm.html) and [$L^2$](https://mathworld.wolfram.com/L2-Norm.html) Distances help quantify the amount of space "we must travel" to get between these two points. Another approach is to examine the angle between two vectors. From trigonometry we know that:




<div align="center">



![inner_product.png](data:image/png;base64,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)



</div>


Instead of computing the actual angle, we can leave the similarity in terms of  similarity$=\cos(\theta)$. Formally, [the Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)  $s$  between two vectors  $p$ and  $q$  is defined as:



$$s=\frac{p\cdot q}{\lVert p \rVert\lVert q \rVert},$$ where $s\in [-1, 1]$.


## Question 2.2: Words with Multiple Meanings [code + written] (1.5 points)

Polysemes and homonyms are words that have more than one meaning (see [this wiki](https://en.wikipedia.org/wiki/Polysemy) for details). Find a word with **at least two different meanings** such that the top-10 most similar words (according to cosine similarity) contain related words from **both** meanings. For example, "leaves" has both "go_away" and "a_structure_of_a_plant" meaning in the top 10, and "scoop" has both "handed_waffle_cone" and "lowdown". You will probably need to try several polysemous or homonymic words before you find one.



Please state the word you discover and the multiple meanings that occur in the top 10. Why do you think many of the polysemous or homonymic words you tried didn't work (i.e. the top-10 most similar words only contain one of the meanings of the words)?



Note: You should use the `wv_from_bin.most_similar(word)` function to get the top 10 similar words. This function ranks all other words in the vocabulary with respect to their cosine similarity to the given word. For further assistance, please check [the GenSim documentation](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.FastTextKeyedVectors.most_similar).



```python
print('10 Similar words to "leaves":', wv_from_bin.most_similar('leaves'))
# ------------------
# Write your implementation here.
word = ['play', 'head', 'bat', 'bank']
for i in word:
    print("Given word: {}".format(i))
    for j in wv_from_bin.most_similar(i):
        print(j)

# ------------------
```

<pre>
10 Similar words to "leaves": [('ends', 0.6128067970275879), ('leaf', 0.6027014851570129), ('stems', 0.5998532176017761), ('takes', 0.5902854800224304), ('leaving', 0.5761634111404419), ('grows', 0.5663397312164307), ('flowers', 0.5600922107696533), ('turns', 0.5536050796508789), ('leave', 0.5496848821640015), ('goes', 0.5434924960136414)]
Given word: play
('playing', 0.8513093590736389)
('played', 0.7978196144104004)
('plays', 0.7768365144729614)
('game', 0.752869188785553)
('players', 0.7354426383972168)
('player', 0.7004927396774292)
('match', 0.662858784198761)
('matches', 0.6463508009910583)
("n't", 0.6403259634971619)
('going', 0.6353694200515747)
Given word: head
('heads', 0.7668997645378113)
('headed', 0.6344294548034668)
('chief', 0.6314131021499634)
('body', 0.6098023653030396)
('assistant', 0.606410562992096)
('director', 0.6037707328796387)
('deputy', 0.583614706993103)
('hand', 0.573833703994751)
('left', 0.5574275255203247)
('arm', 0.5565925240516663)
Given word: bat
('bats', 0.691724419593811)
('batting', 0.6160587668418884)
('balls', 0.5692733526229858)
('batted', 0.5530908107757568)
('toss', 0.5506129264831543)
('wicket', 0.5495278835296631)
('pitch', 0.548936128616333)
('bowled', 0.5452010035514832)
('hitter', 0.5353438854217529)
('batsman', 0.5348091125488281)
Given word: bank
('banks', 0.7625691890716553)
('banking', 0.6818838119506836)
('central', 0.6283639669418335)
('financial', 0.6166563034057617)
('credit', 0.6049750447273254)
('lending', 0.5980608463287354)
('monetary', 0.5963002443313599)
('bankers', 0.5913101434707642)
('loans', 0.5802938938140869)
('investment', 0.5740202069282532)
</pre>




---





**Write your answer here. You can write in either Korean or English.**



다의어나 동음이의어를 word라는 리스트에 넣어 실행 시켰다. 이때 head의 경우에는 신체부위로써의 "머리"라는 의미 그리고 어떤 단체 또는 집단에서의 "우두머리"를 의미하는 단어가 모두 나타남을 확인할 수 있었다. 그러나 bank에서는 개울의 의미를 가지는 단어를 확인할 수 없었고, bat 역시 박쥐의 의미를 가지는 단어를 확인할 수 없었다. 이와 같은 이유는 bank를 개울의 의미로 사용하는 경우가 거의 없으며 대부분 river이라는 단어를 사용하기 때문이며, 이런 의미에 비해 압도적으로 은행의 의미가 크게 작용하기 때문이라고 판단된다. Bat는 한 편 유일하게 박쥐를 나타낼 수 있는 단어임에도 불구하고 박쥐라는 단어 자체의 활용도가 거의 없어서 관련 단어가 나타나지 않았음을 유추할 수 있다.



만약 상위 10개가 아닌, 더 많은 수의 단어를 나타낸다면 결국 다양한 의미를 가지는 단어들이 나타날 것으로 예상된다.



---





## Question 2.3: Synonyms & Antonyms [code + written] (2 points)



When considering Cosine Similarity, it's often more convenient to think of Cosine Distance, which is simply $1 - \text{Cosine Similarity}$.



Find three words  $(w_1,w_2,w_3)$  where  $w_1$  and  $w_2$  are synonyms and  $w_1$  and  $w_3$  are antonyms, but Cosine Distance  $(w_1,w_3)<  \text{Cosine Distance}  (w_1,w_2)$ .



As an example,  $w_1 =``\text{happy}"$ is closer to  $w_3 =``\text{sad}"$ than to  $w_2 =``\text{cheerful}"$. Please find a different example that satisfies the above. Once you have found your example, please give a possible explanation for why this counter-intuitive result may have happened.



You should use the the `wv_from_bin.distance(w1, w2)` function here in order to compute the cosine distance between two words. Please see [the GenSim documentation](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.FastTextKeyedVectors.distance) for further assistance.




```python
from sklearn.utils.validation import has_fit_parameter
print('Cosine distance between "happy" and "sad":', wv_from_bin.distance('happy', 'sad'))
print('Cosine distance between "happy" and "cheerful":', wv_from_bin.distance('happy', 'cheerful'))
# ------------------
# Write your implementation here.
w1 = "love"
w2 = "adore"
w3 = "hate"

print('Cosine distance between "{}" and "{}":'.format(w1, w2), wv_from_bin.distance(w1, w2))
print('Cosine distance between "{}" and "{}":'.format(w1, w3), wv_from_bin.distance(w1, w3))

# ------------------
```

<pre>
Cosine distance between "happy" and "sad": 0.4040136933326721
Cosine distance between "happy" and "cheerful": 0.5172466933727264
Cosine distance between "love" and "adore": 0.6329789161682129
Cosine distance between "love" and "hate": 0.49353712797164917
</pre>




---





**Write your answer here. You can write in either Korean or English.**



Love와 adore는 모두 사랑이라는 감정을 표현하다는 것에서는 같은 유의어다. 하자만 사랑이라는 단어와 사모라는 단어에 뉘앙스가 다르듯, love와 adore도 뉘앙스가 다르게 읽힌다. 반면 Love and hate라는 구절이 자주 사용되 듯 love와 hate는 의미 상으로는 정반대 일지는 모르겠지만, 문장에서 사용되는 형태가 매우 비슷하기에 둘의 Cosine distance는 작게 나타나는 것으로 보인다.



---





## Question 2.4: Analogies with Word Vectors [written] (1.5 points)

Word vectors have been shown to *sometimes* exhibit the ability to solve analogies.



As an example, for the analogy "man : king :: woman : $x$" (read: man is to king as woman is to $x$), what is $x$?



In the cell below, we show you how to use word vectors to find $x$ using the `most_similar` function from the [GenSim documentation](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.KeyedVectors.most_similar). The function finds words that are most similar to the words in the `positive` list and most dissimilar from the words in the `negative` list (while omitting the input words, which are often the most similar; see [this paper](https://aclanthology.org/N18-2039.pdf) for more details). The answer to the analogy will have the highest cosine similarity (largest returned numerical value).



```python
# Run this cell to answer the analogy -- man : king :: woman : x
pprint.pprint(wv_from_bin.most_similar(positive=['woman', 'king'], negative=['man']))
```

<pre>
[('queen', 0.6978678703308105),
 ('princess', 0.6081745028495789),
 ('monarch', 0.5889754891395569),
 ('throne', 0.5775108933448792),
 ('prince', 0.5750998854637146),
 ('elizabeth', 0.546359658241272),
 ('daughter', 0.5399125814437866),
 ('kingdom', 0.5318052768707275),
 ('mother', 0.5168544054031372),
 ('crown', 0.5164472460746765)]
</pre>
Let  $m$ ,  $k$ ,  $w$ , and  $x$  denote the word vectors for `man`, `king`, `woman`, and the answer, respectively. Using only vectors  $m$ ,  $k$ ,  $w$ , and the vector arithmetic operators  $+$  and  $−$  in your answer, what is the expression in which we are maximizing cosine similarity with  $x$ ?



**Hint:** Recall that word vectors are simply multi-dimensional vectors that represent a word. It might help to draw out a 2D example using arbitrary locations of each vector. Where would `man` and `woman` lie in the coordinate plane relative to `king` and the answer?






---





**Write your answer here. You can use \$ \$ signs around the letter to write mathematical symbols and equations.**



$ x = k + (w-m) $



---





## Question 2.5: Finding Analogies [code + written] (1.5 points)

Similar to the example in the cell above in question 2.4, find an example of analogy that holds according to these vectors (i.e. the intended word is ranked top). In your solution please state the full analogy in the form "$x:y :: a:b$". If you believe the analogy is complicated, explain why the analogy holds in one or two sentences.



**Note:** You may have to try many analogies to find one that works!



```python
# ------------------
# Write your implementation here.

pprint.pprint(wv_from_bin.most_similar(positive = ['radio', 'show'], negative = ['music']))


# ------------------
```

<pre>
[('television', 0.6668423414230347),
 ('tv', 0.6427779197692871),
 ('broadcast', 0.6086510419845581),
 ('abc', 0.6053410768508911),
 ('aired', 0.5917834043502808),
 ('shows', 0.5860623121261597),
 ('nbc', 0.5738277435302734),
 ('talk', 0.5514693260192871),
 ('cbs', 0.5453892946243286),
 ('interview', 0.5450031757354736)]
</pre>




---





**Write your answer here. You can write in either Korean or English.**



radio : music :: television : show



기본적인 틀은 "매체 : 미디어"



---





## Question 2.6: Incorrect Analogy [code + written] (1.5 points)

Find an example of analogy that does not hold according to these vectors. In your solution, state the intended analogy in the form "$x:y :: a:b$", and state the (incorrect) value of $b$ according to the word vectors.



```python
# ------------------
# Write your implementation here.

pprint.pprint(wv_from_bin.most_similar(positive = ['fruit', 'fat'], negative = ['vitamin']))


# ------------------
```

<pre>
[('fruits', 0.5519857406616211),
 ('vegetable', 0.5398288369178772),
 ('meat', 0.5263804793357849),
 ('chicken', 0.5009006857872009),
 ('eat', 0.4987374544143677),
 ('pie', 0.49465227127075195),
 ('cream', 0.49077239632606506),
 ('sour', 0.482103556394577),
 ('vegetables', 0.480699360370636),
 ('stuffed', 0.4803802967071533)]
</pre>




---





**Write your answer here. You can write in either Korean or English.**

위에서 주어진 상황은

vitamin : fruit :: fat : $x$

와 같은 상황이다. 이런 상황에서 $x$에 들어갈만 한 음식으로는 지방을 함유하고 있는 다양항 음식들을 생각할 수 있으나, 위 상황에서는 fruits라는 전혀 맞지 않는 단어를 선정하였다.



---





## Question 2.7: Guided Analysis of Bias in Word Vectors [written] (1 point)

It's important to be cognizant of the biases (gender, race, sexual orientation etc.) implicit in our word embeddings. Bias can be dangerous because it can reinforce stereotypes through applications that employ these models.



Run the cell below, to examine (a) which terms are most similar to "woman" and "worker" and most dissimilar to "man", and (b) which terms are most similar to "man" and "worker" and most dissimilar to "woman". Point out the difference between the list of female-associated words and the list of male-associated words, and explain how it is reflecting gender bias.



```python
# Run this cell
# Here `positive` indicates the list of words to be similar to and `negative` indicates the list of words to be
# most dissimilar from.
pprint.pprint(wv_from_bin.most_similar(positive=['woman', 'worker'], negative=['man']))
print()
pprint.pprint(wv_from_bin.most_similar(positive=['man', 'worker'], negative=['woman']))
```

<pre>
[('employee', 0.6375863552093506),
 ('workers', 0.6068919897079468),
 ('nurse', 0.5837947726249695),
 ('pregnant', 0.5363885164260864),
 ('mother', 0.5321309566497803),
 ('employer', 0.5127025842666626),
 ('teacher', 0.5099576711654663),
 ('child', 0.5096741914749146),
 ('homemaker', 0.5019454956054688),
 ('nurses', 0.4970572590827942)]

[('workers', 0.6113258004188538),
 ('employee', 0.5983108282089233),
 ('working', 0.5615328550338745),
 ('laborer', 0.5442320108413696),
 ('unemployed', 0.5368517637252808),
 ('job', 0.5278826951980591),
 ('work', 0.5223963260650635),
 ('mechanic', 0.5088937282562256),
 ('worked', 0.505452036857605),
 ('factory', 0.4940453767776489)]
</pre>




---





**Write your answer here. You can write in either Korean or English.**

여성과 남성 간의 직업에 대한 연관 단어가 편향적으로 나타나는 것을 위 예시를 통해 확인할 수 있었다. 남자들은 mechanic, factory와 같은 단어들이 나타나며, 몸을 사용하는 일, 일반적인 직장과 같은 단어들이 연관 되어 나타나는 반면, 여성들과 관련된 일에서는 간호사와 같은 단어들이 나타났으며, 대체로 엄마, 아이, 임신과 같이 경력이 단절되게 되는 사유의 단어들이 연관되어서 나타났다.



이는 과거 사회적으로 여성들이 커리어를 포기하고 주부로써 살던 시절의 글들이 학습 데이터로 사용되어 나타나는 형상으로 보인다. 이는 기계 자체의 편향성 이라기 보다 데이터 자체가 은연중에 이런 사회적 편향성을 가지고 있음을 시사하는 부분이다.



---





## Question 2.8: Independent Analysis of Bias in Word Vectors [code + written] (1 point)

Use the `most_similar` function to find another case where some bias is exhibited by the vectors. Please briefly explain the example of bias that you discover.



```python
# ------------------
# Write your implementation here.

pprint.pprint(wv_from_bin.most_similar(positive=['white', 'worker'], negative=['black']))
print()
pprint.pprint(wv_from_bin.most_similar(positive=['black', 'worker'], negative=['white']))
# ------------------
```

<pre>
[('employee', 0.6674273610115051),
 ('workers', 0.5972095727920532),
 ('staffer', 0.5401681661605835),
 ('working', 0.5378906726837158),
 ('job', 0.5158447027206421),
 ('labor', 0.5022790431976318),
 ('worked', 0.49961531162261963),
 ('laborer', 0.49479156732559204),
 ('employees', 0.4877925515174866),
 ('aide', 0.4794933795928955)]

[('workers', 0.6524251699447632),
 ('employee', 0.604455292224884),
 ('unemployed', 0.5872254371643066),
 ('immigrant', 0.5486351847648621),
 ('laborer', 0.5476917028427124),
 ('migrant', 0.5399382710456848),
 ('student', 0.5278772711753845),
 ('teacher', 0.5271733999252319),
 ('employer', 0.522971510887146),
 ('woman', 0.5044324994087219)]
</pre>




---





**Write your answer here. You can write in either Korean or English.**



일과 관련하여 흑인과 백인에 대한 편향성을 확인해 보았다. 백인의 경우 일과 관련된 단어들이 많이 나타난 반면, 흑인의 경우 unemployed가 매우 높게 나타났으며, 그외에도 이민자, 떠돌이를 가르키는 단어들이 함꼐 나타났다. 추가로 worker와 관련하여 흑인과 연관성이 높은 단어에 teacher, woman이 나타나는 것은 미국에서 상대적으로 일하는 공간에서 차별을 받은 "여자", 상대적으로 미국에서 직업적으로 좋지 못하다고 평가받는 "선생님"이 나타나는 것으로 보아서는 백인과 흑인에 대한 차별성이 보이는 것 같다.



---





## Question 2.9: Thinking About Bias [written] (2 points)

Give one explanation of how bias gets into the word vectors. What is an experiment that you could do to test for or to measure this source of bias?






---





**Write your answer here. You can write in either Korean or English.**



편견이 좋지 못하고, 또 바르지 못한 판단을 하게 만드는 것은 사실이나 현실 세상에서는 편견은 만연하게 퍼져있다. 이런 사회적인 분위기 때문에 이런 글들이 인터넷에 존재하는 것은 물론이고, 이런 단어들이 함께 긴밀하게 나타나게 된다.



비록 기계는 단어의 뜻을 파악하지는 않으나, 함께 비슷한 글에 분포하는 것 만으로도 편향성이 기계에 학습될 수 있다. 그렇기 때문에 이런 편견이 Ouput으로 고스란히 나타나게 된다.



만약 학습단계에서 이런 편견이 모두 삭제된, 온전히 중립적인 글을 사용하게 되면 그때에도 현제와 같은 편향성이 나타나는지를 비교 해볼 수 있을 것으로 예상된다. 이런 학습에서 편향성이 나타나지 않느다면 삭제된 글들이 모두 편견을 만드는 것에 기여한 글들임을 알 수 있을 것이다. 만약 삭제하였음에도 편견이 나타나게 된다면 외적으로 보이지는 않지만 글의 깊은 기저에서 부터 편견이 설졍되어 있는지 확인해볼 필요가 있다고 생각한다.



---







#### Congratulations on finishing Assignment 1.



# Submission Instructions





1.   Click File -> Save or 파일-> 저장 to save.

2.   Run the code cell below. It requires to mount your Google Drive to VM, so authorize as instructed.

3.   `.html` file will be automatically downloaded (allow the permission to download file if asked). Submit your `.html` file on Blackboard.






```python
from google.colab import drive, files
from requests import get
from urllib.parse import unquote

drive.mount('/mnt/')
filename = unquote(get('http://172.28.0.12:9000/api/sessions').json()[0]['name'])
filepath = f'/mnt/My Drive/COSE461/{filename}'
output_file = '/content/Assignment1.html'

!jupyter nbconvert '{filepath}' --to html --output '{output_file}'
files.download(output_file)
```

<pre>
Drive already mounted at /mnt/; to attempt to forcibly remount, call drive.mount("/mnt/", force_remount=True).
[NbConvertApp] Converting notebook /mnt/My Drive/COSE461/Assignment1 to html
[NbConvertApp] Writing 870334 bytes to /content/Assignment1.html
</pre>
<pre>
<IPython.core.display.Javascript object>
</pre>
<pre>
<IPython.core.display.Javascript object>
</pre>