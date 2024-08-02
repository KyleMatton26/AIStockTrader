import torch
import torch.nn as nn
import torch.optim as optim

print(torch.__version__)
print("Cuda available:", torch.cuda.is_available())

# 1. Data (preparing and loading)

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

#After getting the sentences in the file, we need to split them into 2 arrays: One which contains just the sentence and the other which contains the sentiment
sentences = []
sentiments = []

for sentence in sentences_from_file:
    index_of_at_symbol = sentence.rfind("@")
    new_sentence = sentence[0: index_of_at_symbol - 1] #The new sentence will not include the period or the @ symbol at the end of a sentence
    sentiment = sentence[index_of_at_symbol + 1: len(sentence) - 1] #Sentiment sill start after the @ symbol and go to the end of the string
    sentences.append(new_sentence)
    sentiments.append(sentiment)

for i in range(5):
    print(sentences[i])
    print(sentiments[i])

#Next steps: Convert sentences into a numerical format 



# 2. Build Model


# 3. Fitting the model to data (training) --- MAKE SURE TO SET MANUAL_SEED


# 4. Making predictions and evaluating a model (inference)


# 5. Saving and loading a model


# 6. Putting it all together