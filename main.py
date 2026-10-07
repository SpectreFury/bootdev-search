import argparse

from search.search import (
    build_command,
    idf_command,
    search_command,
    termfreq_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    tf_parser = subparsers.add_parser(
        "tf", help="Search the term frequency given a document id and the term"
    )
    idf_parser = subparsers.add_parser("idf", help="Search the Inverse document frequency for a term")
    _ = subparsers.add_parser("build", help="Build a cache for movies")

    search_parser.add_argument("query", type=str, help="Search query")
    tf_parser.add_argument(
        "doc_id", type=int, help="Document id to search term frequency in"
    )
    tf_parser.add_argument(
        "term", type=str, help="The term that is to be searched in the document id"
    )
    
    idf_parser.add_argument(
        "term", type=str, help="The term to find the IDF for"
    )

    args = parser.parse_args()

    match args.command:
        case "search":
            query = args.query
            print(f"Searching for {query}")

            result = search_command(query)

        case "build":
            build_command()

        case "tf":
            doc_id = args.doc_id
            term = args.term

            termfreq_command(doc_id, term)

        case "idf":
            term = args.term

            # A higher IDF is a more rare word and vice versa
            idf_command(term)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
