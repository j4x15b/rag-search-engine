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
    search_parser.add_argument("--limit", type=int, default=5, help="limits the number of results, default=5")
    
    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verifies, that the shape and size of the embeddings fit")


    args = parser.parse_args()

    match args.command:
        case "search":
            search_command(args.query, args.limit)
            """
            In your semantic search CLI script, add a new search command. It should:

            Accept a required positional string argument, the query.

            Accept an optional --limit argument (default 5).

            """
        
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