import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
import numpy as np

from utils.data import load_movies

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(PROJECT_PATH, "cache")
EMBEDDING_PATH = os.path.join(CACHE_PATH, "movie_embeddings.npy")


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def search(self, query: str, limit: int):
        if self.embeddings is None or not self.embeddings.size:
            raise ValueError(
                "No embeddings. Load them using the verify_embedding command."
            )

        embedding = self.generate_embedding(query)

        score_list = []
        for i in range(len(self.embeddings)):
            score = float(cosine_similarity(embedding, self.embeddings[i]))
            doc = self.documents[i]

            score_list.append((score, doc))

        score_list = sorted(score_list, key=lambda x: x[0], reverse=True)

        return [
            {
                "score": item[0],
                "title": item[1]["title"],
                "description": item[1]["description"],
            }
            for item in score_list[:limit]
        ]

    def verify_model(self):
        print(f"Model loaded: {self.model}")
        print(f"Max sequence length: {self.model.max_seq_length}")

    def build_embeddings(self, documents):
        self.documents = documents

        document_str_list = []
        for document in documents:
            self.document_map[document["id"]] = document

            document_str = f"{document['title']}: {document['description']}"
            document_str_list.append(document_str)

        self.embeddings = self.model.encode(document_str_list, show_progress_bar=True)

        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(EMBEDDING_PATH, "wb") as file:
            np.save(file, self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents):
        # Accept both {"movies": [...]} and [...] for convenience
        if isinstance(documents, dict):
            documents = documents.get("movies", [])
        self.documents = documents

        for document in documents:
            self.document_map[int(document["id"])] = document

        if not os.path.exists(EMBEDDING_PATH):
            return self.build_embeddings(documents)

        with open(EMBEDDING_PATH, "rb") as file:
            self.embeddings = np.load(file)

            if len(self.embeddings) != len(self.documents):
                return self.build_embeddings(documents)

            return self.embeddings

    def generate_embedding(self, text: str):
        if not text.strip():
            raise ValueError("Text cannot be empty")

        output = self.model.encode([text])
        return output[0]


def verify_embeddings():
    semantic_search = SemanticSearch()
    movies_data = load_movies()
    # load_movies() returns {"movies": [...]}
    movies = movies_data["movies"] if isinstance(movies_data, dict) else movies_data

    embeddings = semantic_search.load_or_create_embeddings(movies)
    print(f"Number of docs: {len(movies)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_text(text: str):
    semantic_search = SemanticSearch()

    output = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 2 dimensions: {output[:2]}")
    print(f"Dimensions: {len(output)}")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search for the items")

    _ = subparsers.add_parser("verify", help="Verify the model information")
    _ = subparsers.add_parser(
        "verify_embeddings",
        aliases=["verify_embedding"],
        help="Verify the embeddings",
    )
    embed_parser = subparsers.add_parser(
        "embed", help="Embed the given text into embeddings"
    )

    search_parser.add_argument("query", type=str, help="The query you want to search")
    search_parser.add_argument(
        "--limit", type=int, help="The number of elements you want (default 5)"
    )

    embed_parser.add_argument("text", type=str, help="The text to embed")

    args = parser.parse_args()

    match args.command:
        case "search":
            query = args.query
            limit = args.limit or 5

            semantic_search = SemanticSearch()
            movies_data = load_movies()
            # load_movies() returns {"movies": [...]}
            movies = (
                movies_data["movies"] if isinstance(movies_data, dict) else movies_data
            )
            semantic_search.load_or_create_embeddings(movies)

            result = semantic_search.search(query, limit)

            for i in range(len(result)):
                print(
                    f"{i}. {result[i]["title"]}(score: {result[i]["score"]}) \n {result[i]["description"]}"
                )

        case "verify":
            semantic_search = SemanticSearch()
            semantic_search.verify_model()

        case "verify_embeddings" | "verify_embedding":
            verify_embeddings()

        case "embed":
            text = args.text
            embed_text(text)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
