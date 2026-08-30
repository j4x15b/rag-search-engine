#now this going to be interesting - download a model, embed the movie descriptions, build a vectorized base to measure the semantic distance of the word-content

#always: 
#source .venv/bin/activate
#uv run ./cli/semantic_search_cli.py

#!/usr/bin/env python3
# uv add sentence-transformers


import argparse
from lib.semantic_search import verify_model
from lib.semantic_search import embed_text
from lib.semantic_search import verify_embeddings
from lib.semantic_search import embed_query
from lib.semantic_search import search_command
from lib.semantic_search import chunk_command
from lib.semantic_search import semantic_chunk_command
from lib.semantic_search import embed_chunks_command
from lib.semantic_search import verify_chunk_embeddings
from lib.semantic_search import build_chunk_embeddings
from lib.semantic_search import search_chunked_command


def positive_int(value):
        ivalue = int(value)
        if ivalue < 0:
            raise argparse.ArgumentTypeError("The value must be a positive integer!")
        else: return ivalue


def main() -> None:
    parser = argparse.ArgumentParser(description = "Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="verifies that a model has been loaded succesfully")

    embed_text_parser = subparsers.add_parser("embed_text", help="takes a string-input, loads a model and embeds the text, transforms it to a vector") 
    embed_text_parser.add_argument("text", help="text to get embedded into the text_vector")

    embed_query_parser = subparsers.add_parser("embed_query", help="takes a string-input, embeds and transforms a search query to compare with embedded content.")
    embed_query_parser.add_argument("query", help="insert query to get embedded and compared  with the content")

    search_parser = subparsers.add_parser("search", help="semantic search. Give a query to search for, optional: limit results with --number, default=5")
    search_parser.add_argument("query", help="insert the query you want to look for")
    #search_parser.add_argument("limit", type=int, default=5, help="limits the number of results, default=5", nargs="?") #austauschen?!
    search_parser.add_argument("--limit", type=positive_int, default=5, help="limits the number of results, default=5")
    
    chunk_parser = subparsers.add_parser("chunk", help="splits the text in chunks, given by")
    chunk_parser.add_argument("text", help="insert a text, that you want to have chunked")
    chunk_parser.add_argument("--chunk-size", type=positive_int, default=200, help="defines the size of a chunk in characters, default=200")
    chunk_parser.add_argument("--overlap", type=positive_int, default=0, help="defines an overlap in #amount of characters")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="splits the text in semantically logical chunks")
    semantic_chunk_parser.add_argument("text", help="insert a text, that you want to have chunked")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=positive_int, default=4, help="defines the size of a chunk in full sentences, default=4")
    semantic_chunk_parser.add_argument("--overlap", type=positive_int, default=0, help="defines an overlap in #amount of characters")
    
    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="loads the embedded, chunked text or creates new chunks and re-embeds the text and saves it to chunk_embeddings.npy & chunk_metadata.json")
    
    build_chunk_embeddings_parser = subparsers.add_parser("build_chunk_embeddings", help="deletes old files, chunks the input-text, encodes all chunks and saves them to chunk_embeddings.npy & chunk_metadata.json")
    
    search_chunked_parser = subparsers.add_parser("search_chunked", help="input search query and get the best matching results, with chunked and embedded search")
    search_chunked_parser.add_argument("query", help="type in search query")
    search_chunked_parser.add_argument("--limit", type=positive_int, default=5, help="defines a limit of maximal results, default=5")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verifies, that the shape and size of the embeddings fit")
    verify_chunk_embeddings_parser = subparsers.add_parser("verify_chunk_embeddings", help="Verifies, that the shape and size of the embeddings fit")
    

    
    args = parser.parse_args()

    match args.command:
        case "search":
            search_command(args.query, args.limit)

        case "chunk":
            chunk_command(args.text, args.chunk_size, args.overlap)
        
        case "semantic_chunk":
        #uv run ./cli/semantic_search_cli.py semantic_chunk " Leading and trailing spaces. Those are a problem " --max-chunk-size 1 --overlap 0
            chunk_list = semantic_chunk_command(args.text, args.max_chunk_size, args.overlap)
            print(chunk_list)
            #Output text:
            print(f"Semantically chunking {len(args.text)} characters")
        
            for i, chunk in enumerate(chunk_list):
                print(f"{i+1}. {chunk}")

        case "embed_chunks":
            embed_chunks_command()
        
        case "build_chunk_embeddings":
            build_chunk_embeddings()
        
        case "verify":
            verify_model()

        case "embed_text":
            print(f"embedding text: {args.text}")
            embed_text(args.text)
        
        case "embed_query":
            print(f"embedding text: {args.query}")
            embed_query(args.query)

        case "search_chunked":
            print(f"searching for {args.query}")
            search_chunked_command(args.query, args.limit)
            
        
        case "verify_embeddings":
            verify_embeddings()

        case "verify_chunk_embeddings":
            verify_chunk_embeddings()
        
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()