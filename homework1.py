import os
from collections import Counter

character_count = {}


def read_data(path):
    txt = ""
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
        f.close()
    characters = txt.replace(" ", "")
    return characters


def count_characters1(characters):
    for c in characters:
        if c in character_count.keys():
            character_count[c] += 1
        else:
            character_count[c] = 1


def count_characters2(characters):
    for c in characters:
        character_count[c] = character_count.get(c, 0) + 1


def count_characters3(characters):
    return dict(Counter(characters))


if __name__ == "__main__":
    file_path = os.path.join("data", "homework1", "housing.csv")
    '''method1'''
    count_characters1(read_data(file_path))
    print(character_count)
    '''method2'''
    character_count = {}
    count_characters2(read_data(file_path))
    print(character_count)
    '''method3'''
    character_count = {}
    character_count = count_characters3(read_data(file_path))
    print(character_count)
