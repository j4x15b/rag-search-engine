#now this going to be interesting - download a model, embed the movie descriptions, build a vectorized base to measure the semantic distance of the word-content

#always: 
#source .venv/bin/activate

#!/usr/bin/env python3
# uv add sentence-transformers


import argparse
from lib.semantic_search import verify_model

def main() -> None:
    parser = argparse.ArgumentParser(description = "Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="verifies that a model has been loaded succesfully")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()