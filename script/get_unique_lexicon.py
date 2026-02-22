# count the unique number of symbols in doc/lexicon.txt
with open("doc/lexicon.txt","r") as file:
    sentences = file.readlines()

lexicons = {}
for sentence in sentences:
    symbols = sentence.strip().split(" ")[1:]
    for symbol in symbols:
        if symbol in lexicons.keys():
            lexicons[symbol] +=1
        else:
            lexicons[symbol] = 1

with open("doc/lexicon_unique.txt","x") as file:
    for key, value in lexicons.items():
        file.write(f"{key} {value}\n")