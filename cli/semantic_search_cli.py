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
    

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verifies, that the shape and size of the embeddings fit")

    
    args = parser.parse_args()

    match args.command:
        case "search":
            search_command(args.query, args.limit)

        case "chunk":
            chunk_command(args.text, args.chunk_size, args.overlap)
        
        case "semantic_chunk":
            semantic_chunk_command(args.text, args.max_chunk_size, args.overlap)
        
        case "verify":
            verify_model()

        case "embed_text":
            print(f"embedding text: {args.text}")
            embed_text(args.text)
        
        case "embed_query":
            print(f"embedding text: {args.query}")
            embed_query(args.query)
        
        case "verify_embeddings":
            verify_embeddings()
        
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()