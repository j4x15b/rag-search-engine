import argparse
from search import search_movie

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            search_result = search_movie(args.query)
            
            print(f"Searching for: {args.query}")
            for i in range(0,len(search_result)):
                print(f"{i+1}. {search_result[i]}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()