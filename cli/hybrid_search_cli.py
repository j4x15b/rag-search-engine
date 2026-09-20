import argparse
from lib.hybrid_search import normalize
from lib.hybrid_search import weighted_search_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Takes a list of numbers, normalize it to 0-1 with min/max normalization")
    normalize_parser.add_argument("list_of_numbers", nargs="*", help="List of numbers to be normalized.")
    ##nargs * - liste von nummern übergeben, nicht liste übergeben...

    weighted_search_parser = subparsers.add_parser("weighted-search", help="runs a weighted search. Needs a query, optional an alpha parameter and a result-limit")
    weighted_search_parser.add_argument("query", type=str, help="Type in the search query, what you want to search for?")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="Tweaks the alpha-parameter between 0 and 1. Bigger alpha leads to stronger keyword-search-results, lower Alpha to stronger semantic-search-results. Default: 0.5")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="Limit the search results.")

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

        case _:
            parser.print_help()




if __name__ == "__main__":
    main()