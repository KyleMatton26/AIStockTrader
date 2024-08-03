#Creating single file for model

import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('punkt') 
nltk.download('omw-1.4')

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

dict_of_sentiments_per_risk = {
    "num_pos_50": 0, "num_neutral_50": 0, "num_neg_50": 0,
    "num_pos_33": 0, "num_neutral_33": 0, "num_neg_33": 0,
    "num_pos_25": 0, "num_neutral_25": 0, "num_neg_25": 0,
    "num_pos_0": 0, "num_neutral_0": 0, "num_neg_0": 0,
}

sen_positive_50 = []
sen_neutral_50 = []
sen_negative_50 = []
sen_positive_33 = []
sen_neutral_33 = []
sen_negative_33 = []
sen_positive_25 = []
sen_neutral_25 = []
sen_negative_25 = []
sen_positive_0 = []
sen_neutral_0 = []
sen_negative_0 = []

for sentence in list_of_all_sentences:
    if sentence.find("Risk: 50") != -1:
        if sentence.find("@positive") != -1:
            dict_of_sentiments_per_risk["num_pos_50"] += 1
            sen_positive_50.append(sentence)

        elif sentence.find("@neutral") != -1:
            dict_of_sentiments_per_risk["num_neutral_50"] += 1
            sen_neutral_50.append(sentence)

        elif sentence.find("@negative") != -1:
            dict_of_sentiments_per_risk["num_neg_50"] += 1
            sen_negative_50.append(sentence)

    elif sentence.find("Risk: 33") != -1:
        if sentence.find("@positive") != -1:
            dict_of_sentiments_per_risk["num_pos_33"] += 1
            sen_positive_33.append(sentence)

        elif sentence.find("@neutral") != -1:
            dict_of_sentiments_per_risk["num_neutral_33"] += 1
            sen_neutral_33.append(sentence)

        elif sentence.find("@negative") != -1:
            dict_of_sentiments_per_risk["num_neg_33"] += 1
            sen_negative_33.append(sentence)

    elif sentence.find("Risk: 25") != -1:
        if sentence.find("@positive") != -1:
            dict_of_sentiments_per_risk["num_pos_25"] += 1
            sen_positive_25.append(sentence)

        elif sentence.find("@neutral") != -1:
            dict_of_sentiments_per_risk["num_neutral_25"] += 1
            sen_neutral_25.append(sentence)

        elif sentence.find("@negative") != -1:
            dict_of_sentiments_per_risk["num_neg_25"] += 1
            sen_negative_25.append(sentence)

    elif sentence.find("Risk: 0") != -1:
        if sentence.find("@positive") != -1:
            dict_of_sentiments_per_risk["num_pos_0"] += 1
            sen_positive_0.append(sentence)

        elif sentence.find("@neutral") != -1:
            dict_of_sentiments_per_risk["num_neutral_0"] += 1
            sen_neutral_0.append(sentence)

        elif sentence.find("@negative") != -1:
            dict_of_sentiments_per_risk["num_neg_0"] += 1
            sen_negative_0.append(sentence)

print(dict_of_sentiments_per_risk)

for i in range(5):
    print(sen_negative_50[i])

def remove_sentiment_and_risk(sentence):

    #Removes risk from sentence
    risk_end_index = sentence.find("--") + 3
    sen_without_risk = sentence[risk_end_index:]

    #Remove sentiment from sentence
    sentiment_start_index = sen_without_risk.find(".@")
    sen_without_risk_and_sentiment = sen_without_risk[:sentiment_start_index]

    return sen_without_risk_and_sentiment.strip()

def remove_non_alphanumeric_characters(sentence):
    characters_to_keep = [".", "%", "$", " ", "-", "+"]
    new_sentence = ""

    for char in sentence:
        if char.isalnum() or char in characters_to_keep:
            new_sentence += str(char)

    return new_sentence


#nltk stopword list is problematic. We will have to create our own stopword set. Will finish method once stopword set is created

"""
def remove_stopwords(sentence):
    stop_words = set(stopwords.words("english"))
    print(stop_words)

"""

lematizer = WordNetLemmatizer()

def lemmatize_sentence(sentence):
    
    words = nltk.word_tokenize(sentence)
    lematized_words = []

    for word in words:
        lematized_word = lematizer.lemmatize(word)
        lematized_words.append(lematized_word)

    lematized_sentence = " ".join(lematized_words)

    return lematized_sentence


for i in range(5):
    sentence = sen_positive_33[i]

    sen_1 = remove_sentiment_and_risk(sentence)
    sen_2 = remove_non_alphanumeric_characters(sen_1)
    sen_3 = lemmatize_sentence(sen_2)

    print(sen_3)



processed_sen_positive_50 = []
processed_sen_neutral_50 = []
processed_sen_negative_50 = []
processed_sen_positive_33 = []
processed_sen_neutral_33 = []
processed_sen_negative_33 = []
processed_sen_positive_25 = []
processed_sen_neutral_25 = []
processed_sen_negative_25 = []
processed_sen_positive_0 = []
processed_sen_neutral_0 = []
processed_sen_negative_0 = []


# Function to process and append sentences to corresponding lists
def process_and_append(sentences, processed_list):
    for sentence in sentences:
        sen_1 = remove_sentiment_and_risk(sentence)
        sen_2 = remove_non_alphanumeric_characters(sen_1)
        sen_3 = lemmatize_sentence(sen_2)
        processed_list.append(sen_3)


# Process sentences and append to corresponding processed lists
process_and_append(sen_positive_50, processed_sen_positive_50)
process_and_append(sen_neutral_50, processed_sen_neutral_50)
process_and_append(sen_negative_50, processed_sen_negative_50)
process_and_append(sen_positive_33, processed_sen_positive_33)
process_and_append(sen_neutral_33, processed_sen_neutral_33)
process_and_append(sen_negative_33, processed_sen_negative_33)
process_and_append(sen_positive_25, processed_sen_positive_25)
process_and_append(sen_neutral_25, processed_sen_neutral_25)
process_and_append(sen_negative_25, processed_sen_negative_25)
process_and_append(sen_positive_0, processed_sen_positive_0)
process_and_append(sen_neutral_0, processed_sen_neutral_0)
process_and_append(sen_negative_0, processed_sen_negative_0)

print("Finsihed processing all sentences")




#Create 12 different arrays: 1 for each risk with its associated sentiment - DONE

#Create methods to preprocess the data - DONE

#Convert each array into its numerical representation

#Perform SMOTE to get 1000 samples for each group

#Combine sentence array with risk array to create the X, sentiment array is the Y

#Create training and test splits

#Create method to return training and tests splits

# Call it in PyTorchTest

