import argparse
from lib.hybrid_search import normalize
from lib.hybrid_search import weighted_search_command
from lib.hybrid_search import rrf_search_command


def main() -> None:

    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Takes a list of numbers, normalize it to 0-1 with min/max normalization")
    normalize_parser.add_argument("list_of_numbers", nargs="*", help="List of numbers to be normalized.")
    ##nargs * - liste von nummern übergeben, nicht liste übergeben...

    help_query = 'Type in the search query - what do you want to search for? Use quotation marks for "more than one word"'
    help_limit = "Limits the search results."

    weighted_search_parser = subparsers.add_parser("weighted-search", help="runs a weighted search. Needs a query, optional an alpha parameter and a result-limit")
    weighted_search_parser.add_argument("query", type=str, help=help_query)
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="Tweaks the alpha-parameter between 0 and 1. Bigger alpha leads to stronger keyword-search-results, lower Alpha to stronger semantic-search-results. Default: 0.5")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help=help_limit)

    rrf_search_parser = subparsers.add_parser("rrf-search", help="runs a rrf-search, which converts the scores into ranks - good for non-normalized scores")
    rrf_search_parser.add_argument("query", type=str, help=help_query)
    rrf_search_parser.add_argument("-k", type=int, default=60, help='the k-parameter controls the weight, that is given to ranks - lower k (20) gives more weight to top-ranked-results, higher k (100) gives lower ranked results more influence')
    rrf_search_parser.add_argument("--limit", type=int, default=5, help=help_limit)
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    rrf_search_parser.add_argument("--rerank-method", type=str, choices=["individual"], help="reranks the results for you")

    args = parser.parse_args()

    match args.command:
        
        case "normalize":
            try: 
                numbers = args.list_of_numbers[0].split(',')
                numbers = [float(x.strip()) for x in numbers if x.strip()]
                print(numbers)
                normalized_list = normalize(numbers)
                print(normalized_list)
                for score in normalized_list:
                    print(f"* {score:.4f}")
            except: ValueError("Bitte nur mit Komma getrennte Zahlen eingeben")

        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)

        case "rrf-search":
            rrf_search_command(args.query, args.k, args.limit, args.enhance, args.rerank_method)

        case _:
            parser.print_help()




if __name__ == "__main__":
    main()
    