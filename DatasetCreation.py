#Creating single file for model

import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from imblearn.over_sampling import SMOTE
import random


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

# All datasets
all_processed_datasets = [
    (processed_sen_positive_50, 0.50, 'positive'),
    (processed_sen_neutral_50, 0.50, 'neutral'),
    (processed_sen_negative_50, 0.50, 'negative'),
    (processed_sen_positive_33, 0.33, 'positive'),
    (processed_sen_neutral_33, 0.33, 'neutral'),
    (processed_sen_negative_33, 0.33, 'negative'),
    (processed_sen_positive_25, 0.25, 'positive'),
    (processed_sen_neutral_25, 0.25, 'neutral'),
    (processed_sen_negative_25, 0.25, 'negative'),
    (processed_sen_positive_0, 0.00, 'positive'),
    (processed_sen_neutral_0, 0.00, 'neutral'),
    (processed_sen_negative_0, 0.00, 'negative')
]

def convert_sentences_to_numerical_values(sentences):
    # Create and fit the TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(sentences)
    
    # Convert to array
    X_array = X.toarray()
    
    # Filter out zeros and return the numerical values
    numerical_values = []
    
    for vector in X_array:
        non_zero_elements = []
        
        for i in range(len(vector)):
            if vector[i] != 0:
                non_zero_elements.append(vector[i])
        
        numerical_values.append(non_zero_elements)
    
    return numerical_values

# Step 1: Convert sentences to numerical values for all datasets
def convert_all_datasets_to_numerical_values():
    datasets = [
        processed_sen_positive_50,
        processed_sen_neutral_50,
        processed_sen_negative_50,
        processed_sen_positive_33,
        processed_sen_neutral_33,
        processed_sen_negative_33,
        processed_sen_positive_25,
        processed_sen_neutral_25,
        processed_sen_negative_25,
        processed_sen_positive_0,
        processed_sen_neutral_0,
        processed_sen_negative_0
    ]
    
    numerical_datasets = []
    
    for dataset in datasets:
        numerical_values = convert_sentences_to_numerical_values(dataset)
        numerical_datasets.append(numerical_values)
    
    return numerical_datasets

# Step 2: Find the longest array in each dataset
def find_longest_in_each_dataset(numerical_datasets):
    max_lengths = []
    
    for numerical_values in numerical_datasets:
        dataset_max_length = max(len(vector) for vector in numerical_values)
        max_lengths.append(dataset_max_length)
    
    return max_lengths

# Step 3: Find the longest array across all datasets
def find_overall_max_length(max_lengths):
    overall_max_length = max(max_lengths)
    return overall_max_length

# Main processing
numerical_datasets = convert_all_datasets_to_numerical_values()
max_lengths = find_longest_in_each_dataset(numerical_datasets)
overall_max_length = find_overall_max_length(max_lengths)

print("Maximum length of arrays in each dataset:", max_lengths)
print("Overall maximum length of arrays:", overall_max_length)

# Step 4: Pad arrays to the overall max length
def pad_arrays_to_max_length(numerical_datasets, max_length):
    padded_datasets = []
    
    for dataset in numerical_datasets:
        padded_dataset = []
        for vector in dataset:
            # Pad the vector if it is shorter than max_length
            if len(vector) < max_length:
                padded_vector = np.pad(vector, (0, max_length - len(vector)), 'constant')
            else:
                padded_vector = vector
            padded_dataset.append(padded_vector)
        padded_datasets.append(padded_dataset)
    
    return padded_datasets

def apply_smote(numerical_values, desired_samples=270):
    if len(numerical_values) >= desired_samples:
        return numerical_values  # No SMOTE needed
    
    # Create dummy samples
    num_current_samples = len(numerical_values)
    num_dummy_samples = max(2, desired_samples - num_current_samples)
    
    # Create dummy data with the same dimensionality as numerical_values
    dummy_data = np.zeros((num_dummy_samples, len(numerical_values[0])))
    
    # Combine real and dummy data
    combined_data = np.vstack([numerical_values, dummy_data])
    
    # Create labels
    labels = np.array([1] * num_current_samples + [0] * num_dummy_samples)
    
    # Apply SMOTE
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(combined_data, labels)
    
    # Filter out dummy samples
    X_resampled = X_resampled[y_resampled == 1]
    
    # Print debug information
    num_total_samples = len(X_resampled)
    print(f"Number of samples after SMOTE (before slicing): {num_total_samples}")
    
    # Ensure we return at least desired_samples
    if num_total_samples < desired_samples:
        # If not enough samples, pad with zeros
        padding = np.zeros((desired_samples - num_total_samples, len(X_resampled[0])))
        X_resampled = np.vstack([X_resampled, padding])
    
    # Return the final number of samples
    final_samples = X_resampled.tolist()[:desired_samples]
    print(f"Number of samples returned: {len(final_samples)}")
    
    return final_samples

# Function to add risk to numerical arrays
def add_risk_to_numerical_array(sentences, risk: float):
    result = []
    for sentence in sentences:
        # Convert numpy.ndarray to list and append the risk
        if isinstance(sentence, np.ndarray):
            sentence = sentence.tolist()
        sentence.append(risk)
        result.append(sentence)
    return result

# Function to create tuples with sentiment
def create_tuple_with_sentiment(sentences, sentiment: str):
    array_of_tuples = []
    for sentence in sentences:
        array_of_tuples.append((sentence, sentiment))
    return array_of_tuples

# Step 6: Process each dataset and append to combined_X_dataset
combined_X_dataset = []

def process_and_combine_dataset(numerical_values, risk, sentiment):
    numerical_values_with_risk = add_risk_to_numerical_array(numerical_values, risk)
    
    # If the size exceeds 270, randomly select 270 samples
    if len(numerical_values_with_risk) > 270:
        numerical_values_with_risk = random.sample(numerical_values_with_risk, 270)
    
    tuples_with_sentiment = create_tuple_with_sentiment(numerical_values_with_risk, sentiment)
    combined_X_dataset.extend(tuples_with_sentiment)

# Convert and pad datasets
numerical_datasets = convert_all_datasets_to_numerical_values()
padded_datasets = pad_arrays_to_max_length(numerical_datasets, overall_max_length)


# Apply SMOTE and combine datasets
for i in range(len(padded_datasets)):
    print(f"Before SMOTE, dataset {i} size: {len(padded_datasets[i])}")
    smote_applied_values = apply_smote(padded_datasets[i])
    print(f"After SMOTE, dataset {i} size: {len(smote_applied_values)}")
    
    # Print size before adding to combined dataset
    print(f"Size before adding to combined dataset for dataset {i}: {len(smote_applied_values)}")
    
    process_and_combine_dataset(smote_applied_values, all_processed_datasets[i][1], all_processed_datasets[i][2])

print("Finished processing and combining all datasets.")
print(f"Combined dataset size: {len(combined_X_dataset)}")

def save_combined_dataset_to_file(combined_X_dataset, file_path):
    with open(file_path, 'w') as file:
        for item in combined_X_dataset:
            # Convert tuple to string and write to file
            file.write(f"{item}\n")

# Path to the file where you want to save the dataset
file_path = 'combined_X_dataset.txt'

# Save the combined dataset to the file
save_combined_dataset_to_file(combined_X_dataset, file_path)

print(f"Combined dataset has been saved to {file_path}.")


