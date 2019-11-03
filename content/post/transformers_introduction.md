+++
title = "Transformers Introduction"
date = 2019-11-03  #T13:23:10+01:00
draft = false
#tags = ["Transformers"]
categories = []
#markup: mmark
summary = "Notes on Transformers"
#disable_comments: true
+++

## Transformers
Transformers are models which use attention to speed up training. While other models use attention, transformers discard the recurrent and convolution used in other architectures.

## Self-attention
Turn every word into a linear combination of each words' _value vector_ ($V$). The weights in the linear combination come inner products of word pairs' _query vector_ ($Q$) and _key vector_ ($K$). These three matrices $Q, K, V$ are parameters learned during training.

1. Compute three vectors from X, whose rows are word vectors, like word2vec:
  * Query: $X W^Q = Q$
  * Key: $X W^K = K$
  * Value: $X W^V = V$
  * These three vectors are all N-by-$d_k$, where $N$=number of words.
2. Calculate a score for each input word with all other words.
  * Dot product of the word's _query vector_ with the other word's _key vector_.
  * $Q K^T$
3. Normalize by the dimension of the _key vector_:
  * $\frac{Q K^T}{\sqrt{dim(K)}}$
4. Apply __softmax__ to rows so each words' scores are positive and sum to 1.
  * $softmax(\frac{Q K^T}{\sqrt(dim(K))})$
5. For each word, take a linear combination of the words (rows in the value matrix V). Words that have higher weights from the softmax will receive more weight:
  * $Z = softmax(\frac{Q K^T}{\sqrt(dim(K))}) V$
  * Each row of the new matrix is a weighted sum of the rows of V. This is because left-multiplicaiton of matrices is a linear combination or rows.
  * The left matrix forces the V matrix to focus on certain words and not others.

The resulting matrix $Z$ has for each word a weight sum of the words' values from $V$, where the weights are computed using inner products from $Q$ and $K$. 

## Multi-headed attention
Now let's use mutiple sets of Query/Key/Value matrices, one for each _head_.

* Before: $(W^Q, W^K, W^V)$
* After: $\{(W_0^Q, W_0^K, W_0^V),..,(W_7^Q, W_7^K, W_7^V)\}$

That gives us eight different Z matrics:  
* $Z_0,...,Z_7$
Now, concatenate all the matrices, and multiply by an additional weights matrix $W^0 \(8d_k \times d_W \)$.  
* $Z_c=[Z_0,...,Z_8] \(N \time (8 \cdot d_k)\)$  
* $Z = Z_cW^0$  
This output We can summarize the calculation as follows.  
$$
Z = Z_c W^0 = [Z_1,...,Z_7]
$$
where
$Z_i = softmax(\frac{Q_i K_i^T}{\sqrt{d_k}}) V_i$

The weight matrices $\(W_i^Q, W_i^K, W_i^V\)$ are initialized randomly and learned during training. They project input embeddings into different representation subspaces.


### References
* [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
* [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)