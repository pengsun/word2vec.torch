## my fork of https://github.com/rotmanmi/word2vec.torch

## word2vec.torch7

WORD2VEC wrapper for Torch7.

### Prerequisites
get the pre-trained word2vec binary file from:
https://code.google.com/p/word2vec/

### Installation
* git clone https://github.com/pengsun/word2vec.torch google-wordvec
* cd google-wordvec. Make sure you specify the location of the 'GoogleNews-vectors-negative300.bin' file in 'w2vutils.lua'. It is also suggested you specify a t7 file for fast access.
* luarocks make

Then it will be installed as package with the name `google-wordvec`

### Usage
#### [Tensor] word2vec(self,word,throwerror)
This function gets a word, and returns its word2vec representation, a tensor with the size 300. If throwerror is false (default) and the word doesn't exist it returns nil, otherwise, it will throw an exception.

#### [table] distance(self,word,k)
This function returns the k-nearest neighbours to the given word. It returns a table with a list of words, and a corresponding list of cosine distances.

#### [number] vec_size(self)
Should be 300

###Example
Getting the word2vec representation of the world 'Hello' and finding its k's nearest words.

```Lua

    local gwv = require 'google-wordvec'
    local k = 3
    hellorep = gwv:word2vec('Hello')
    neighbors = gwv:distance(hellorep,k)
    
    assert(300 == gwv:vec_size())

```