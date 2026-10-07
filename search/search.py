import math
import os
import pickle
from utils.data import load_movies, load_stopwords
from utils.text_transformation import tokenize_text, transform_text
from nltk.stem import PorterStemmer
from collections import Counter, defaultdict

PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))
CACHE_PATH = os.path.join(PROJECT_PATH, "cache")
INDEX_FILE_PATH = os.path.join(CACHE_PATH, "index.pkl")
DOCMAP_FILE_PATH = os.path.join(CACHE_PATH, "docmap.pkl")
TERMFREQ_FILE_PATH = os.path.join(CACHE_PATH, "term_frequencies.pkl")

stemmer = PorterStemmer()


def preprocess_tokens(text: str) -> list[str]:
    return [stemmer.stem(t) for t in tokenize_text(transform_text(text))]


def preprocess_term(term: str) -> str:
    return stemmer.stem(tokenize_term(transform_text(term)))


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies = defaultdict(Counter)

    def __add_document(self, doc_id: int, text: str):
        tokens = preprocess_tokens(text)
        for token in tokens:
            self.index[token].add(doc_id)

        self.term_frequencies[doc_id] = Counter(tokens)

    def get_documents(self, term: str):
        doc_ids = self.index.get(term, set())

        return sorted(list(doc_ids))

    def get_tf(self, doc_id, term):
        counter = self.term_frequencies.get(doc_id)
        if not counter:
            return 0

        return counter[term]

    def build(self):
        movies_data = load_movies()

        for movie in movies_data["movies"]:
            id = movie["id"]
            movie_data = f"{movie["title"]} {movie["description"]}"

            self.docmap[id] = movie
            self.__add_document(id, movie_data)

    def save(self):
        os.makedirs(CACHE_PATH, exist_ok=True)  # Make dir if not exists

        with open(INDEX_FILE_PATH, "wb") as file:
            print(f"Saving index dump into {INDEX_FILE_PATH}")
            pickle.dump(self.index, file)

        with open(DOCMAP_FILE_PATH, "wb") as file:
            print(f"Saving docmap dump into {DOCMAP_FILE_PATH}")
            pickle.dump(self.docmap, file)

        with open(TERMFREQ_FILE_PATH, "wb") as file:
            print(f"Saving term frequencies dump into {TERMFREQ_FILE_PATH}")
            pickle.dump(self.term_frequencies, file)

    def load(self):
        is_index_file = os.path.isfile(INDEX_FILE_PATH)
        is_docmap_file = os.path.isfile(DOCMAP_FILE_PATH)
        is_termfreq_file = os.path.isfile(TERMFREQ_FILE_PATH)

        if not is_index_file:
            print("Index dump doesn't exist")
            return

        if not is_docmap_file:
            print("Docmap dump doesn't exist")
            return

        if not is_termfreq_file:
            print("Termfreq dump doesn't exist")
            return

        with open(INDEX_FILE_PATH, "rb") as file:
            loaded_index = pickle.load(file)
            self.index = loaded_index

        with open(DOCMAP_FILE_PATH, "rb") as file:
            loaded_docmap = pickle.load(file)
            self.docmap = loaded_docmap

        with open(TERMFREQ_FILE_PATH, "rb") as file:
            loaded_docmap = pickle.load(file)
            self.term_frequencies = loaded_docmap

        print("Successfully loaded into index and docmap and term frequencies")


def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()


def search_command(query: str):
    idx = InvertedIndex()
    idx.load()

    tokens = preprocess_tokens(query)

    res = []
    for token in tokens:
        document_ids = idx.get_documents(token)

        for id in document_ids:
            res.append(id)
            if len(res) >= 5:
                break

    for id in res:
        movie = idx.docmap.get(id)
        print(f"ID: {movie["id"]} {movie["title"]}")


def termfreq_command(doc_id: int, term: str):
    token = preprocess_term(term)

    idx = InvertedIndex()
    idx.load()

    freq = idx.get_tf(doc_id, token)
    print(f"TF for term: {token} in doc_id: {doc_id} -> {freq}")

def compute_idf(idx: InvertedIndex, token: str) -> float:
    doc_count = len(idx.docmap)
    term_doc_count = len(idx.index.get(token, set()))

    return math.log((doc_count + 1) / (term_doc_count + 1))


def idf_command(term: str):
    token = preprocess_term(term)

    idx = InvertedIndex()
    idx.load()

    idf = compute_idf(idx, token)
    print(f"Inverse document frequency of {token}: {idf:.2f}")
    return idf

def tfidf_command(doc_id: int, term: str):
    token = preprocess_term(term)

    idx = InvertedIndex()
    idx.load()

    tf_score = idx.get_tf(doc_id, token)
    idf_score = compute_idf(idx, token)

    tf_idf = tf_score * idf_score
    print(f"TF-IDF score of '{token}' in document '{doc_id}': {tf_idf:.2f}")


def tokenize_term(term: str):
    token_list = tokenize_text(term)
    if not token_list:
        raise RuntimeError("Empty term")
    if len(token_list) > 1:
        raise RuntimeError("Too many terms")

    return token_list[0]


# Bad way to search - time complexity bad
def search_keyword(query: str) -> list[str]:
    movies_data = load_movies()
    stopwords_list = load_stopwords()

    query = transform_text(query)
    query_list = tokenize_text(query)
    filtered_query_list = [
        stemmer.stem(x) for x in query_list if x not in stopwords_list
    ]

    res = []
    seen = set()

    for movie in movies_data["movies"]:
        title = transform_text(movie["title"])
        title_list = tokenize_text(title)
        filtered_title_list = [
            stemmer.stem(x) for x in title_list if x not in stopwords_list
        ]

        for word in filtered_query_list:
            for filtered_title in filtered_title_list:
                if word in filtered_title:
                    if word in seen:
                        break

                    seen.add(movie["title"])

    res = []
    for item in seen:
        res.append(item)

    return res
