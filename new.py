import nltk
import random
import json
import pickle
import numpy as np
import tensorflow as tf
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents file
with open('chat bot\intents.json') as file:
    intents = json.load(file)

# Initialize lists
words = []
classes = []
documents = []
ignore_letters = ['?', ',', '!', '.']

# Process each pattern in the intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the sentence
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Add to documents
        documents.append((word_list, intent['tag']))
        # Add to classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))

# Sort classes
classes = sorted(set(classes))

# Save words and classes using pickle
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create training data
training = []
# Create an empty array for output
output_empty = [0] * len(classes)

# Training set, bag of words for each sentence
for document in documents:
    bag = []
    word_patterns = document[0]
    # Lemmatize each word
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    # Create bag of words array
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    # Output is a '0' for each tag and '1' for the current tag
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    training.append(bag + output_row)

# Shuffle the training data and convert it to a NumPy array
random.shuffle(training)
training = np.array(training)

# Create train and test lists. X - patterns, Y - intents
train_x = training[:, :len(words)]
train_y = training[:, len(words):]

# Build the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))

# Compile the model
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save the trained model
model.save('chatbot_model.h5', hist)

# Save the training history
with open('train_history.pkl', 'wb') as file:
    pickle.dump(hist.history, file)

print("Model created and saved successfully")

# Load the model and other necessary data


from tensorflow.keras.models import load_model # type: ignore

model = load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Function to clean up the user input and convert to a bag of words
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"Found in bag: {w}")
    return np.array(bag)

# Function to predict the class
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Function to get the response
def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Main function to interact with the chatbot
def chatbot_response(text):
    ints = predict_class(text, model)
    res = get_response(ints, intents)
    return res

# Chat with the bot
while True:
    message = input("You: ")
    if message.lower() == "quit":
        break
    response = chatbot_response(message)
    print(f"Bot: {response}")


