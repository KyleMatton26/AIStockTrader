import torch
import torch.nn as nn
import torch.optim as optim
from DatasetCreation import get_sentences_in_file

print(torch.__version__)
print("Cuda available:", torch.cuda.is_available())

# 1. Data (preparing and loading)

#After getting the sentences in the file, we need to split them into 2 arrays: One which contains just the sentence and the other which contains the sentiment
sentences_from_file = get_sentences_in_file("SentencesWithRisk.txt")
sentences = []
risks = []
sentiments = []

for sentence in sentences_from_file:
    index_of_at_symbol = sentence.rfind("@")
    index_of_dash = sentence.find("-")
    new_sentence = sentence[index_of_dash + 3: index_of_at_symbol - 1] #The new sentence will not include the period or the @ symbol at the end of a sentence
    sentiment = sentence[index_of_at_symbol + 1: len(sentence) - 1] #Sentiment sill start after the @ symbol and go to the end of the string
    
    if sentence[6] != 0:
        risk = int(sentence[6] + sentence[7])
    else:
        risk = 0
    risks.append(risk)
    sentences.append(new_sentence)
    sentiments.append(sentiment)

for i in range(len(sentences_from_file)):
    if i % 20 == 0:
        print("Sentence: " + sentences[i])
        print("Sentiment: " + sentiments[i])
        print("Risk: " + str(risks[i]))
        print("")

#Next steps: Convert sentences into a numerical format 



# 2. Build Model (Possibly an LSTM model)


# 3. Fitting the model to data (training) --- MAKE SURE TO SET MANUAL_SEED


# 4. Making predictions and evaluating a model (inference)


# 5. Saving and loading a model


# 6. Putting it all together