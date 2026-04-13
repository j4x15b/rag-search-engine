##########################################

RAG Search Engine | Python, LLMs, Vector Search

Building a semantic search engine powering a Netflix-like streaming service as part of Boot.dev's RAG course. The project covers the full Retrieval Augmented Generation pipeline:

Ingesting and indexing a real movie dataset
Implementing semantic search using vector embeddings
Augmenting LLM prompts with retrieved context
Generating accurate, context-aware responses beyond simple keyword matching

###########################################

In this project at first I follow along a boot.dev course: [Learn retrieval augmentation](https://www.boot.dev/courses/learn-retrieval-augmented-generation) to build a RAG-Search Engine for movies. 

I'm gonna use this knowledge to build my own project, my Podcast-RAG, where you can train the AI on podcast content and 
use it as a intelligent search.

##########################################

Readme:

source:
[movie-database from boot.dev](https://storage.googleapis.com/qvault-webapp-dynamic-assets/course_assets/course-rag-movies.json)

use virtual environment:
source .venv/bin/activate

use cli to search a movie title, prints out top 5 hits from /data/movies.json

use:

uv run keyword_search_cli.py <search_query>
##########################################

