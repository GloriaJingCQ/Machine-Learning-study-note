Learning Resources: <br>
https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/ <br>

# RAG
- Retrieval Argumented Generation
- Problem trying to solve:
  1. When it comes to LLM model, there are two common problems: No source(privacy or Lack of some domain knowledge); Data is outdated; n LLM’s parameters essentially represent the general patterns of how humans use words to form sentences.<br>
  2. LLM is a very generatic model, trained on human text, so when it comes to a deeper knowledge about a specific field, it tends to perform less well and even make things up. <br>
- How it works
  1. We have an external knowledge base stored in a vector db.
  2. When user passes a prompt to LLM, it will be embedded first to a numerical vector and sent to vectordb to find most possible(similar) answers.
  3. The answers will be transformed to human readable sentences, and passes it back to the LLM.
  4. Finally, the LLM combines the retrieved words and its own response to the query into a final answer it presents to the user, potentially citing sources the embedding model found.
  5. LangChain, an open-source library, can be particularly useful in chaining together LLMs, embedding models and knowledge bases.
- Advantages
  1. Up to date knowledge, and cheaper to get updated knowledge without retraining the whole LLM
  2. Gave LLM a context, so it generated more accurate answer and less makeup
     
