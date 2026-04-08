import json
from pathlib import Path

def search_movie(search_word):
    search_list = []
    script_dir = Path(__file__).parent
    datei = script_dir.parent / "data" / "movies.json"
    with open(datei, 'r', encoding='utf-8') as file:
        movie_data = json.load(file)

    for movie in movie_data["movies"]:
        if search_word in movie["title"]:
            search_list.append(movie["title"])

    top_5 = search_list[0:5]
    return top_5

#print(search_movie("Alien"))
#print(search_movie("SideFX"))