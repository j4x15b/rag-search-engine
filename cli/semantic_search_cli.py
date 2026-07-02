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

def main() -> None:
    parser = argparse.ArgumentParser(description = "Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="verifies that a model has been loaded succesfully")

    embed_text_parser = subparsers.add_parser("embed_text", help="takes a string-input, loads a model and embeds the text, transforms it to a vector") 
    embed_text_parser.add_argument("text", help="text to get embedded into the text_vector")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verifies, that the shape and size of the embeddings fit")


    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            print(f"embedding text: {args.text}")
            embed_text(args.text)
        
        case "verify_embeddings":
            verify_embeddings()


        case _:
            parser.print_help()


if __name__ == "__main__":
    main()