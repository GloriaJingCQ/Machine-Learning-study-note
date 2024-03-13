'''
!pip install -qU \
  pinecone-client==3.1.0 \
  pinecone-datasets==0.7.0 \
  sentence-transformers==2.2.2
'''

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import torch
import os
import pinecone
from tqdm.auto import tqdm

# Part1: Data Preparation
dataset = load_dataset('quora', split='train[240000:290000]')

'''
What data set looks like:
Dataset({
    features: ['questions', 'is_duplicate'],
    num_rows: 80000
})
'''
print(dataset[:5])

# Our questions is a list of strings(questions)
questions = []
for record in dataset['questions']:
    questions.extend(record['text'])
# remove duplicates
questions = list(set(questions))

# Part 2: Embedding the questions using SentenceTransformer(PRE-TRAINED MODEL) and add them to the vector index
# Step1: Load the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device != 'cuda':
    print(f"You are using {device}. This is much slower than using "
          "a CUDA-enabled GPU. If on Colab you can change this by "
          "clicking Runtime > Change runtime type > GPU.")

model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Step2: Create a Pinecone index

# TODO: Replace with your Pinecone API key and environment name
# get api key from app.pinecone.io
api_key = os.environ.get('PINECONE_API_KEY') or 'PINECONE_API_KEY'
# find your environment next to the api key in pinecone console
env = os.environ.get('PINECONE_ENVIRONMENT') or 'PINECONE_ENVIRONMENT'

pinecone.init(
    api_key=api_key,
    environment=env
)

index_name = 'semantic-search'  # name of the index
if index_name not in pinecone.list_indexes().names():
    pinecone.create_index(
        name=index_name,
        dimension=model.get_sentence_embedding_dimension(), # basically the size of the vector
        metric='cosine' # cosine similarity, we can also use euclidean distance or dot product
    )

# now connect to the index
index = pinecone.Index(index_name)

# Step 3: Upsert the questions into the index
# Our vectors are this structure: (id, question_embedding, metadata)

batch_size = 128
vector_limit = 100000

questions = questions[:vector_limit]

for i in tqdm(range(0, len(questions), batch_size)):
    i_end = min(i+batch_size, len(questions))
    ids = [str(x) for x in range(i, i_end)]
    metadatas = [{'text': text} for text in questions[i:i_end]]
    # create embeddings
    xc = model.encode(questions[i:i_end])
    records = zip(ids, xc, metadatas)
    index.upsert(vectors=records)

# check number of records in the index
index.describe_index_stats()

# Part 3: Making queries
query = "which metropolis has the highest number of people?"

# create the query vector
xq = model.encode(query).tolist()

# now query
xc = index.query(vector=xq, top_k=5, include_metadata=True)
for result in xc['matches']:
    print(f"{round(result['score'], 2)}: {result['metadata']['text']}")
     
