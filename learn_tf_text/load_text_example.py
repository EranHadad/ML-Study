import tensorflow as tf
import tensorflow_datasets as tfds
import os

DEBUG_PRINT = False

# ========
# Download
# ========
# The text files used in this tutorial have undergone some typical preprocessing tasks, mostly removing stuff —
# document header and footer, line numbers, chapter titles. Download these lightly munged files locally.
DIRECTORY_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
FILE_NAMES = ['cowper.txt', 'derby.txt', 'butler.txt']

for name in FILE_NAMES:
    text_dir = tf.keras.utils.get_file(name, origin=DIRECTORY_URL + name)

parent_dir = os.path.dirname(text_dir)

print('text files were downloaded to:', parent_dir)


# =======================
# Load text into datasets
# =======================
# Iterate through the files, loading each one into its own dataset.
# Each example needs to be individually labeled, so use tf.data.Dataset.map to apply a labeler function to each one.
# This will iterate over every example in the dataset, returning (example, label) pairs.
def labeler(example, index):
    return example, tf.cast(index, tf.int64)


labeled_data_sets = []

for i, file_name in enumerate(FILE_NAMES):
    lines_dataset = tf.data.TextLineDataset(os.path.join(parent_dir, file_name))
    labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
    labeled_data_sets.append(labeled_dataset)

if DEBUG_PRINT:
    print('')
    print('labeled_data_sets:')
    print('\tlength:', len(labeled_data_sets))
    print('\ttype:', type(labeled_data_sets))

    for i, ds in enumerate(labeled_data_sets):
        for ex in ds.take(3).as_numpy_iterator():
            print('dataset:', i, 'sample:', ex)


# Combine these labeled datasets into a single dataset, and shuffle it.
BUFFER_SIZE = 50000  # for shuffling the data
BATCH_SIZE = 100
TAKE_SIZE = 5000  # test set number of samples

all_labeled_data = labeled_data_sets[0]
for labeled_dataset in labeled_data_sets[1:]:
    all_labeled_data = all_labeled_data.concatenate(labeled_dataset)

all_labeled_data = all_labeled_data.shuffle(
    BUFFER_SIZE, reshuffle_each_iteration=False)

if DEBUG_PRINT:
    print('all_labeled_data:')
    print('\ttype:', type(all_labeled_data))

    for ex in all_labeled_data.take(5).as_numpy_iterator():
        print('sample:', ex)


# =============================
# Encode text lines as numbers
# =============================
# Machine learning models work on numbers, not words, so the string values need to be converted into lists of numbers.
# To do that, map each unique word to a unique integer.

# Build vocabulary
# First, build a vocabulary by tokenizing the text into a collection of individual unique words.
# (1) Iterate over each example's numpy value.
# (2) Use tfds.features.text.Tokenizer to split it into tokens.
# (3) Collect these tokens into a Python set, to remove duplicates.
# (4) Get the size of the vocabulary for later use.

tokenizer = tfds.features.text.Tokenizer()

vocabulary_set = set()
for text_tensor, _ in all_labeled_data:
    some_tokens = tokenizer.tokenize(text_tensor.numpy())
    vocabulary_set.update(some_tokens)

vocab_size = len(vocabulary_set)

print('vocabulary size:', vocab_size)

# Encode examples
# Create an encoder by passing the vocabulary_set to tfds.features.text.TokenTextEncoder.
# The encoder's encode method takes in a string of text and returns a list of integers.
encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

if DEBUG_PRINT:
    example = next(iter(all_labeled_data))[0].numpy()
    print(example)
    encoded_example = encoder.encode(example)
    print(encoded_example)


def encode(text_tensor_, label_):
    encoded_data = encoder.encode(text_tensor_.numpy())
    return encoded_data, label_


def encode_map_fn(text, label):
    # py_func doesn't set the shape of the returned tensors.
    encoded_text, label = tf.py_function(encode,
                                         inp=[text, label],
                                         Tout=(tf.int64, tf.int64))

    # `tf.data.Datasets` work best if all components have a shape set
    #  so set the shapes manually:
    encoded_text.set_shape([None])
    label.set_shape([])

    return encoded_text, label


all_encoded_data = all_labeled_data.map(encode_map_fn)


if DEBUG_PRINT:
    print('all_labeled_data:')
    print('\ttype:', type(all_encoded_data))

    for ex in all_encoded_data.take(5).as_numpy_iterator():
        print('sample:', ex)


# ==============================================
# Split the dataset into test and train batches
# ==============================================

# find the length of the longest sequence
max_sequence_len = 0
avg_sequence_len = 0
n_examples = 0
for encoded_example, _ in all_encoded_data.as_numpy_iterator():
    curr_seq_len = encoded_example.shape[0]
    n_examples += 1
    avg_sequence_len += curr_seq_len
    if curr_seq_len > max_sequence_len:
        max_sequence_len = curr_seq_len
avg_sequence_len /= n_examples
print('number of examples:', n_examples)
print('sequence length: average={0:.1f}, max={1:d}'.format(avg_sequence_len, max_sequence_len))


train_data = all_encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
train_data = train_data.take(TAKE_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE, drop_remainder=True, padded_shapes=([None], []))
# train_data = train_data.padded_batch(BATCH_SIZE, drop_remainder=True, padded_shapes=([max_sequence_len], []))


test_data = all_encoded_data.take(TAKE_SIZE)
test_data = test_data.padded_batch(BATCH_SIZE, drop_remainder=True, padded_shapes=([None], []))
# test_data = test_data.padded_batch(BATCH_SIZE, drop_remainder=True, padded_shapes=([max_sequence_len], []))

if DEBUG_PRINT:
    sample_text, sample_labels = next(iter(test_data))
    print('sample_text[0]:', sample_text[0])
    print('sample_labels[0]:', sample_labels[0])

# Since we have introduced a new token encoding (the zero used for padding), the vocabulary size has increased by one.
vocab_size += 1


# ================
# Build the model
# ================
model = tf.keras.Sequential()

# The first layer converts integer representations to dense vector embeddings.
WORD_LENGTH = 64
model.add(tf.keras.layers.Embedding(vocab_size, WORD_LENGTH))

# The next layer is a Long Short-Term Memory layer, which lets the model understand words in their context with
# other wordsץ
# A bidirectional wrapper on the LSTM helps it to learn about the datapoints in relationship to the datapoints that
# came before it and after it.
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))

# One or more dense layers.
# Edit the list in the `for` line to experiment with layer sizes.
for units in [64, 64]:
    model.add(tf.keras.layers.Dense(units, activation='relu'))

# Output layer. The first argument is the number of labels.
model.add(tf.keras.layers.Dense(3))

# Finally, compile the model.
# For a softmax categorization model, use sparse_categorical_crossentropy as the loss function.
# You can try other optimizers, but adam is very common.
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# ===============
# Train the model
# ===============
early_stopping = tf.keras.callbacks.EarlyStopping()
model.fit(train_data, epochs=2, validation_data=test_data, callbacks=[early_stopping], verbose=2)

# eval_loss, eval_acc = model.evaluate(test_data)
# print('\nEval loss: {:.3f}, Eval accuracy: {:.3f}'.format(eval_loss, eval_acc))
