# always: source .venv/bin/activate

import argparse
from search import search_command
from search import InvertedIndex
from search import test_text
from search import single_term_tokenizer

def prepare_parser():
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--search_limit", type=int, default=5, help="set hit list limit (standard=5)")
    
    build_parser = subparsers.add_parser("build", help="builds the inverted index for movies")
    test_parser = subparsers.add_parser("test", help="tests specific functions and methods")
    save_parser = subparsers.add_parser("save", help="saves the built index to a temporary cache folder")
    
    tf_parser = subparsers.add_parser("tf", help="prints the term frequency of a given term")
    tf_parser.add_argument("doc_id", help="insert the document ID, which you want to analyse")
    tf_parser.add_argument("term", help="insert a single term, you want to search")

    return parser
    

def main() -> None:
    parser = prepare_parser()
    args = parser.parse_args()

    match args.command:
        case "search":
            inverted_index = InvertedIndex()
            try:
                inverted_index.load()
            except FileNotFoundError:
                print("No index found. Run 'build' first.")
                return
            print(f"{len(inverted_index.index)} items in index")
            print(f"Searching for: {args.query}")
            search_result = search_command(inverted_index, args.query, args.search_limit)

            if search_result:
                for i, search_result in enumerate(search_result, 1):
                    print(f"{i}. {search_result}")
        # case _ : else
        case "tf":
            tokenized_term = single_term_tokenizer(args.term)
            inverted_index = InvertedIndex()
            inverted_index.load()
            print(inverted_index.get_tf(int(args.doc_id), tokenized_term))

        case "build":
            print(f"Building inverted index")
            inverted_index = InvertedIndex()
            inverted_index.build()
            inverted_index.save()
            print(inverted_index.index)
            test_list = inverted_index.get_document('merida')
            print(test_list)
        
        case "test":
            inverted_index = InvertedIndex()
            inverted_index.load()
            #inverted_index.build()
            #print(inverted_index.index)
            #print(inverted_index.docmap)
            #print(inverted_index.term_frequencies)
            print(inverted_index.get_tf(4999, "b"))
            #print(inverted_index.get_tf(420, "fire"))
            # print("Testing")
            
            # inverted_index.add_document(20000000000000000000, "Hello, how are you, you and you?")
            # inverted_index.add_document(3, "This is how it's gonna be.")
            # inverted_index.add_document(1, "Is this how it's gonna be with you and me?")
            # print(inverted_index.get_document('how'))
            #inverted_index = InvertedIndex()

        case "save":
            print("Saving")
            inverted_index = InvertedIndex()
            inverted_index.save()
            print("saved")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()