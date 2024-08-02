#Creating single file for model

folder_path = "./FinancialPhraseBank-v1.0/"

list_of_files = [
    "Sentences_50Agree.txt", #Contains 4846 Sentences 
    "Sentences_66Agree.txt", #Contains 4217 Sentences 
    "Sentences_75Agree.txt", #Contains 3453 Sentences
    "Sentences_AllAgree.txt" #Contains 2264 Senteces 
]


def count_sentences_in_files(folder_path, file_names):
    for file_name in file_names:
        file_path = folder_path + file_name
        try:
            with open(file_path, 'r', encoding='utf-8', errors="ignore") as file:
                lines = file.readlines()
                num_sentences = len(lines)
                print(f"File: {file_name} | Number of Sentences: {num_sentences}")
        except FileNotFoundError:
            print(f"File {file_name} not found in {folder_path}")

count_sentences_in_files(folder_path, list_of_files)

#Using the 75Agree.txt because it is a much larger data set than the AllAgree.txt while still maintaining a high level of accuracy 
file_for_training_path = folder_path + list_of_files[2]

def get_sentences_in_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors="ignore") as file:
            lines = file.readlines() 
            return lines       
    except FileNotFoundError:
        print(f"File not found")

sentences_from_file = get_sentences_in_file(file_for_training_path)

list_of_all_sentences = []

sentences_from_50 = get_sentences_in_file(folder_path + list_of_files[0])
sentences_from_66 = get_sentences_in_file(folder_path + list_of_files[1])
sentences_from_75 = get_sentences_in_file(folder_path + list_of_files[2])
sentences_from_all = get_sentences_in_file(folder_path + list_of_files[3])

for sentence in sentences_from_all:
    list_of_all_sentences.append("Risk: 0 -- " + sentence)
    
for sentence in sentences_from_75:
    if sentence not in sentences_from_all:
        list_of_all_sentences.append("Risk: 25 -- " + sentence)

for sentence in sentences_from_66:
    if sentence not in sentences_from_all and sentence not in sentences_from_75:
        list_of_all_sentences.append("Risk: 33 -- " + sentence)

for sentence in sentences_from_50:
    if sentence not in sentences_from_all and sentence not in sentences_from_75 and sentence not in sentences_from_66:
        list_of_all_sentences.append("Risk: 50 -- " + sentence)

output_file_path = "SentencesWithRisk.txt"

with open(output_file_path, 'w', encoding='utf-8') as file:
    for sentence in list_of_all_sentences:
        file.write(sentence)

print("Data has been written to file")

all_sentences = get_sentences_in_file(output_file_path)

for i in range(len(all_sentences)):
    if i % 10 == 0:
        print(all_sentences[i])

num_50 = 0
num_33 = 0
num_25 = 0
num_0 = 0

for sentence in all_sentences:
    if sentence.find("Risk: 50") != -1:
        num_50 += 1
    elif sentence.find("Risk: 33") != -1:
        num_33 += 1
    elif sentence.find("Risk: 25") != -1:
        num_25 += 1
    elif sentence.find("Risk: 0") != -1:
        num_0 += 1

print("Risk 50s: " + str(num_50))
print("Risk 33s: " + str(num_33))
print("Risk 25s: " + str(num_25))
print("Risk 0s: " + str(num_0))