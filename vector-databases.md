# Vector Embeddings
Vector Embedding is actually for a automatic feature engineering. We're using a pretrained model to help us filter the most relevant features.
## Problems trying to solve:
- Over time, the number of properties of objects grows. And for some particular tasks, we don't really need to consider that many properties, we need to do some feature engineering
- For unstructured data, this is more challenging to do it manually
## Intro about Vector Embeddings
-  Why Vectors??? Modern CPUs and GPUs are optimized to perform the mathematical operations needed to process vectors. But a problem is that usually our data is not represented as vectors.
-  It’s a technique that allows us to take virtually any data type and represent it as vectors. Moreover, We want to ensure that we can perform tasks on this transformed data without losing the data’s original meaning.
We want to make sure when we turn data to mathematically vectors, their original meaning and relationships(associations) are well preserved. 
-  We transform data to vectors using an embedding model which is trained in a neural network(supervised training). Instead of inputs to a labeled output, we do inputs to another inputs vector, basically we removed the last layer in the neural network, but we
still do the activation functions, etc.
- An example: 
word2vec: king-man+woman ~= queen



