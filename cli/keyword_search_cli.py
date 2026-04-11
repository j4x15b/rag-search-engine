# always: source .venv/bin/activate

import argparse
from search import search_command
from search_utils import DEFAULT_SEARCH_LIMIT

def prepare_parser():
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    return parser
    

def main() -> None:
    parser = prepare_parser()
    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            search_result = search_command(args.query)
                        
            for i, result in enumerate(search_result, 1):
                print(f"{i}. {result['title']}")
        # case _ : else
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()