import json

from utils.text_transformation import transform_text

def load_movies():
    with open("data/movies.json") as file:
        movies = json.load(file)

    return movies

def load_stopwords():
    with open("data/stopwords.txt") as file:
        words = file.read()
        words_list = words.splitlines()

    for i in range(len(words_list)):
        words_list[i] = transform_text(words_list[i])

    return words_list

