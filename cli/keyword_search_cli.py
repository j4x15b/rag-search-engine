# always: source .venv/bin/activate

import argparse
from search import search_command
from search import InvertedIndex
from search import print_document, test_text
from search import single_term_tokenizer
from search import bm25_idf_command, bm25_tf_command
import math

from search_utils import BM25_K1, BM25_B

def prepare_parser():
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--search_limit", type=int, default=5, help="set hit list limit (standard=5)")
    
    build_parser = subparsers.add_parser("build", help="builds the inverted index for movies")
    save_parser = subparsers.add_parser("save", help="saves the built index to a temporary cache folder")
    
    tf_parser = subparsers.add_parser("tf", help="prints the term frequency of a given term")
    tf_parser.add_argument("doc_id", type=int, help="insert the document ID, which you want to analyse")
    tf_parser.add_argument("term", help="insert a single term, you want to search")

    tf_parser = subparsers.add_parser("idf", help="prints the term frequency of a given term")
    tf_parser.add_argument("term", help="insert a single term, you want to search")

    tfidf_parser = subparsers.add_parser("tfidf", help="prints the term frequency of a given term")
    tfidf_parser.add_argument("doc_id", type=int, help="insert the document ID, which you want to analyse")
    tfidf_parser.add_argument("term", help="insert a single term, you want to search")
    
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, default=5, help="limits result list (standard: 5)")

    print_document_parser = subparsers.add_parser("print", help="Prints a whole document with a given document number")
    print_document_parser.add_argument("doc_id", type=int, help="#Document ID")

    test_parser = subparsers.add_parser("test", help="tests specific functions and methods")

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

        case "build":
            print(f"Building inverted index")
            inverted_index = InvertedIndex()
            inverted_index.build()
            inverted_index.save()
            print(inverted_index.index)
            test_list = inverted_index.get_document('merida')
            print(test_list)

        case "save":
            print("Saving")
            inverted_index = InvertedIndex()
            inverted_index.save()
            print("saved")

        # case _ : else
        case "tf":
            tokenized_term = single_term_tokenizer(args.term)
            inverted_index = InvertedIndex()
            inverted_index.load()
            print(inverted_index.get_tf(args.doc_id, tokenized_term))
        
        case "idf":
            inverted_index = InvertedIndex()
            inverted_index.load()
            
            matches = 0
            tokenized_term = single_term_tokenizer(args.term)
            
            # try: 
            #     matches = (len(inverted_index.index[tokenized_term]))
            # except KeyError:
            #     matches = 0
            matches = len(inverted_index.index[tokenized_term])
            print(matches)
            print(tokenized_term)
            idf = math.log((len(inverted_index.docmap) + 1) / (matches + 1))

            #print(idf:.2f)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
            print(list(inverted_index.index["actor"])[:5])
        
        case "tfidf":
            inverted_index = InvertedIndex()
            inverted_index.load()
            tokenized_term = single_term_tokenizer(args.term)
            tf = inverted_index.get_tf(int(args.doc_id), tokenized_term)
            matches = len(inverted_index.index[tokenized_term])
            idf = math.log((len(inverted_index.docmap) + 1) / (matches + 1))
            tf_idf = tf * idf
            #print(tf_idf)
            #print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf}")
        
        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            #print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
            print(f"BM25 IDF score of '{args.term}': {bm25idf}")

        case "bm25tf":
            bm25_tf = bm25_tf_command(args.doc_id, args.term, args.k1)
            #print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25_tf:.2f}")
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25_tf}")
        
        case "bm25search":
            limit = args.limit

            inverted_index = InvertedIndex()
            inverted_index.load()
            result_items = inverted_index.bm25_search(args.query, args.limit)

            for i in range(min(limit, len(result_items))):
                doc_id, score = result_items[i]
                print(f"{i+1}. ({doc_id}) {inverted_index.docmap[doc_id]['title']} - Score: {score:.2f}")

        case "print":
            full_document = print_document(args.doc_id)

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

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()