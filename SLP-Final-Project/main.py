# Jonathan Avila and Jonathan Contreras
# CS 5319 SLP Final Project - Text with Emoji Predictor
# November 25, 2020

import re
import sys
import pickle
from random import choices

import emojis
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Configuration.
N_TWEETS = 262144  # 2^18
MAX_VOCAB_SIZE = 16384  # 2^14
UNKNOWN_TOKEN = "<UNK>"
START_TOKEN = "<s>"
END_TOKEN = "</s>"

# Returns the perplexity of test string (see Equation 3.16).
def perplexity(test_string, vocab, model):
    test_string_tokens = tokenize(test_string)
    N = len(test_string_tokens)
    product = 1
    for i in range(N - 2):
        prev_token = test_string_tokens[i]
        next_token = test_string_tokens[i+1]
        prev_token = prev_token if prev_token in vocab else UNKNOWN_TOKEN
        next_token = next_token if next_token in vocab else UNKNOWN_TOKEN
        prev_token_index = vocab.index(prev_token)
        next_token_index = vocab.index(next_token)

        # Apologies for the ugly bug fix.
        if model[prev_token_index].shape[0] == 1:
            p = model[prev_token_index][0, next_token_index]
        else:
            p = model[prev_token_index][next_token_index]

        product *= 1 / p
    return product ** (1 / float(N))


# Return the predicted next token given the previous token.
def predict(prev_token, vocab, model):
    prev_token = prev_token if prev_token in vocab else UNKNOWN_TOKEN
    prev_token_index = vocab.index(prev_token)
    next_token_probabilities = model[prev_token_index].tolist()

    # Apologies for another ugly bug fix.
    if isinstance(next_token_probabilities[0], list):
        next_token_probabilities = next_token_probabilities[0]

    # Choose the next token based on their probabilities.
    return choices(vocab, next_token_probabilities)[0]


# Returns an autocompleted string from start_string. Emoji must be decoded.
def autocomplete(start_string, vocab, model):
    MAX_GENERATE = 32
    tokens = tokenize(start_string)
    curr_token = tokens[-1]
    count = 1
    while curr_token != END_TOKEN and count <= MAX_GENERATE:
        curr_token = predict(curr_token, vocab, model)
        tokens.append(curr_token)
        count += 1
    # if count >= MAX_GENERATE:
    #     print("\tAutocomplete went on for too long.")
    return " ".join(tokens)


# Returns a list of tokens from string.
def tokenize(string):

    # Remove semicolons to avoid issues decoding and encoding emoji.
    string = re.sub(":", "", string)

    # Decode emoji. For example, the string "good job 👍" decoded is
    # "good job :thumbs_up:".
    string = emojis.decode(string)

    # Remove skin tone modifiers. Skin tone modifiers are not supported by emoji
    # decoding end encoding functions.
    string = re.sub(r"[🏻🏼🏽🏾🏿]", "", string)

    # Surround decoded emoji and some punctuation with spaces to gurantee proper
    # tokenization. (The final step will split on whitespace.)
    string = re.sub("(:(.*?):)", r" \1 ", string)
    string = re.sub("([.,!;]+)", r" \1 ", string)

    tokens = string.split()
    return tokens


def create_model():
    print(f"Creating bigram model from {N_TWEETS} tweets.")

    # Read the first N_TWEETS from EmojifyData-EN dataset (18,883,592 in total)
    # (kaggle.com/rexhaif/emojifydata-en). Replace surrounding whitespace with
    # start and stop tokens.
    fp = open("emojitweets-01-04-2018.txt", encoding="utf-8")
    tweets = []
    for count, line in enumerate(fp):
        if count >= N_TWEETS:
            break
        line = f"{START_TOKEN} {line.strip()} {END_TOKEN}"
        tweets.append(line)

    # Split tweets into training (80%), development (10%), and test (10%) sets.
    # We are done with development, so we can ignore the dev set now.
    split_index1 = int(round(N_TWEETS * 0.8))
    split_index2 = int(round(N_TWEETS * 0.9))
    tweets_train = tweets[:split_index1]
    tweets_test = tweets[split_index1:split_index2]
    # tweets_dev = tweets[split_index2:]

    # Learn the vocabulary of the training set.
    vectorizer_unigram = CountVectorizer(tokenizer=tokenize, lowercase=False)
    doc_unigram_matrix = vectorizer_unigram.fit_transform(tweets_train)
    vocab = vectorizer_unigram.get_feature_names()
    unigram_counts = np.sum(doc_unigram_matrix, axis=0)

    # Trim the vocabulary down to the most common MAX_VOCAB_SIZE tokens,
    # including unknown token, start token, and end token.
    required_tokens = [UNKNOWN_TOKEN, START_TOKEN, END_TOKEN]
    if len(vocab) > MAX_VOCAB_SIZE:
        sorted_idx = np.argsort(unigram_counts).tolist()[0]
        vocab = [vocab[i] for i in sorted_idx[-MAX_VOCAB_SIZE - len(required_tokens) :]]
    else:
        print(f"MAX_VOCAB_SIZE is too large.")
        sys.exit()
    replace_idx = 0
    for token in required_tokens:
        if token not in vocab:
            vocab[replace_idx] = token
            replace_idx += 1
    print(f"Vocabulary size is {len(vocab)}.")

    # Calculate the training set's unigram counts.
    vectorizer_unigram = CountVectorizer(
        tokenizer=tokenize, lowercase=False, vocabulary=vocab
    )
    doc_unigram_matrix = vectorizer_unigram.fit_transform(tweets_train)
    vocab = vectorizer_unigram.get_feature_names()
    unigram_counts = np.sum(doc_unigram_matrix, axis=0)

    # Calculate the training set's bigram counts. Meanwhile, count the number of
    # tokens in the trainin set.
    token_count = 0
    bigram_counts = np.zeros((len(vocab), len(vocab)))
    for tweet in tweets_train:
        tokens = tokenize(tweet)
        token_count += len(tokens)
        for i in range(len(tokens) - 1):
            curr_token = tokens[i]
            next_token = tokens[i + 1]
            curr_token = curr_token if curr_token in vocab else UNKNOWN_TOKEN
            next_token = next_token if next_token in vocab else UNKNOWN_TOKEN
            curr_token_index = vocab.index(curr_token)
            next_token_index = vocab.index(next_token)
            bigram_counts[curr_token_index][next_token_index] += 1

    unigram_counts += 1

    # Convert unigram counts to probabilities.
    unigram_probs = unigram_counts / token_count

    # Convert bigram counts to probabilities (see Equation 3.11).
    # bigram_probs[i][j] is the probability of token j given token i.
    bigram_probs = bigram_counts / np.transpose(unigram_counts)

    # Apply interpolation to get the final model.
    lambda_bigram = 0.9
    lambda_unigram = 1 - lambda_bigram
    model = lambda_bigram * bigram_probs + lambda_unigram * unigram_probs

    # Apologies for running perplexity tests here but now is the best time.
    perplexities = np.zeros(len(tweets_test))
    for i, tweet in enumerate(tweets_test):
        pp = perplexity(tweet, vocab, model)
        perplexities[i] = pp
    mean_perplexity = np.mean(perplexities)
    print(f"Mean perplexity is {mean_perplexity}.")

    return model, vocab


if __name__ == "__main__":

    model_id = f"{N_TWEETS}-{MAX_VOCAB_SIZE}"
    model_filename = f"model-{model_id}.npy"
    vocab_filename = f"vocab-{model_id}.txt"

    model = None
    vocab = None

    # Load the model and its vocabulary if they exist.
    try:
        model = np.load(model_filename)
        with open(vocab_filename, "rb") as fp:
            vocab = pickle.load(fp)
        print("Loaded model and vocabulary.")
    except Exception as e:
        print("Failed loading model and/or vocabulary.")

    # If either model or its vocabulary does not exist, create new ones.
    if model is None:
        print("This might take several minutes.")
        model, vocab = create_model()

    # Save the model and its vocabulary.
    np.save(model_filename, model)
    with open(vocab_filename, "wb") as fp:  # Pickling
        pickle.dump(vocab, fp)

    # A user interface for demo.
    OPTION_QUIT = "q"
    OPTION_PREDICT = "p"
    OPTION_PERPLEXITY = "x"
    OPTION_AUTOCOMPLETE = "a"

    while True:
        selection = input(
            "What would you like to do? {[p]redict, [a]utocomplete, perple[x]ity, [q]uit}: "
        )

        # Ask for a previous token and predicted the next token.
        if selection == OPTION_PREDICT:
            while True:
                user_input = input("Enter a previous token or [q]uit: ")
                if user_input == OPTION_QUIT:
                    break
                prediction = predict(user_input, vocab, model)
                prediction = emojis.encode(prediction)
                print(f"{prediction}\n")

        # Ask for a string and autocomplete it.
        elif selection == OPTION_AUTOCOMPLETE:
            while True:
                user_input = input("Enter a string to autocomplete or [q]uit: ")
                if user_input == OPTION_QUIT:
                    break
                completed_tweet = autocomplete(user_input, vocab, model)
                completed_tweet = emojis.encode(completed_tweet)
                print(f"{completed_tweet}\n")

        # Ask for a string and calculate its perplexity.
        elif selection == OPTION_PERPLEXITY:
            while True:
                user_input = input("Enter a string to measure or [q]uit: ")
                if user_input == OPTION_QUIT:
                    break
                pp = perplexity(user_input, vocab, model)
                print(f"PP({user_input}) = {pp}\n")

        elif selection == OPTION_QUIT:
            break

        else:
            print("I didn't understand that.")
