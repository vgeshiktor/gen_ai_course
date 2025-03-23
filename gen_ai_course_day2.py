import google.generativeai as genai
from IPython.display import Markdown
import os
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from google.api_core import retry
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups


GOOGLE_API_KEY = os.environ["GOOGLE_AI_STUDIO_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

# for m in genai.list_models():
#     if "embedContent" in m.supported_generation_methods:
#         print(m.name)

DOCUMENT1 = "Operating the Climate Control System  Your Googlecar has a climate control system that allows you to adjust the temperature and airflow in the car. To operate the climate control system, use the buttons and knobs located on the center console.  Temperature: The temperature knob controls the temperature inside the car. Turn the knob clockwise to increase the temperature or counterclockwise to decrease the temperature. Airflow: The airflow knob controls the amount of airflow inside the car. Turn the knob clockwise to increase the airflow or counterclockwise to decrease the airflow. Fan speed: The fan speed knob controls the speed of the fan. Turn the knob clockwise to increase the fan speed or counterclockwise to decrease the fan speed. Mode: The mode button allows you to select the desired mode. The available modes are: Auto: The car will automatically adjust the temperature and airflow to maintain a comfortable level. Cool: The car will blow cool air into the car. Heat: The car will blow warm air into the car. Defrost: The car will blow warm air onto the windshield to defrost it."
DOCUMENT2 = 'Your Googlecar has a large touchscreen display that provides access to a variety of features, including navigation, entertainment, and climate control. To use the touchscreen display, simply touch the desired icon.  For example, you can touch the "Navigation" icon to get directions to your destination or touch the "Music" icon to play your favorite songs.'
DOCUMENT3 = "Shifting Gears Your Googlecar has an automatic transmission. To shift gears, simply move the shift lever to the desired position.  Park: This position is used when you are parked. The wheels are locked and the car cannot move. Reverse: This position is used to back up. Neutral: This position is used when you are stopped at a light or in traffic. The car is not in gear and will not move unless you press the gas pedal. Drive: This position is used to drive forward. Low: This position is used for driving in snow or other slippery conditions."

documents = [DOCUMENT1, DOCUMENT2, DOCUMENT3]


class GeminiEmbeddingFunction(EmbeddingFunction):
    # Specify whether to generate embeddings for documents, or queries
    document_mode = True

    def __call__(self, input: Documents) -> Embeddings:
        if self.document_mode:
            embedding_task = "retrieval_document"
        else:
            embedding_task = "retrieval_query"

        retry_policy = {"retry": retry.Retry(
            predicate=retry.if_transient_error)}

        response = genai.embed_content(
            model="models/text-embedding-004",
            content=input,
            task_type=embedding_task,
            request_options=retry_policy,
        )
        return response["embedding"]


# DB_NAME = "googlecardb"
# embed_fn = GeminiEmbeddingFunction()
# embed_fn.document_mode = True

# chroma_client = chromadb.Client()
# db = chroma_client.get_or_create_collection(
#     name=DB_NAME, embedding_function=embed_fn
# )

# db.add(documents=documents, ids=[str(i) for i in range(len(documents))])

# db.count()
# You can peek at the data too.
# print(db.peek(1))

# Switch to query mode when generating embeddings.
# embed_fn.document_mode = False

# Search the Chroma DB using the specified query.
# query = "How do you use the touchscreen to play music?"

# result = db.query(query_texts=[query], n_results=1)
# [[passage]] = result["documents"]
# print(passage)
# Markdown(passage)

# passage_oneline = passage.replace("\n", " ")
# query_oneline = query.replace("\n", " ")

# This prompt is where you can specify any guidance on tone, or what topics the model should stick to, or avoid.
# prompt = f"""You are a helpful and informative bot that answers questions using text from the reference passage included below.
# Be sure to respond in a complete sentence, being comprehensive, including all relevant background information.
# However, you are talking to a non-technical audience, so be sure to break down complicated concepts and
# strike a friendly and converstional tone. If the passage is irrelevant to the answer, you may ignore it.

# QUESTION: {query_oneline}
# PASSAGE: {passage_oneline}
# """
# print(prompt)

# model = genai.GenerativeModel("gemini-1.5-flash-latest")
# answer = model.generate_content(prompt)
# print(answer)
# Markdown(answer.text)

texts = [
    "The quick brown fox jumps over the lazy dog.",
    "The quick rbown fox jumps over the lazy dog.",
    "teh fast fox jumps over the slow woofer.",
    "a quick brown fox jmps over lazy dog.",
    "brown fox jumping over dog",
    "fox > dog",
    # Alternative pangram for comparison:
    "The five boxing wizards jump quickly.",
    # Unrelated text, also for comparison:
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus et hendrerit massa. Sed pulvinar, nisi a lobortis sagittis, neque risus gravida dolor, in porta dui odio vel purus.",
]


# response = genai.embed_content(
#     model="models/text-embedding-004", content=texts, task_type="semantic_similarity"
# )


def truncate(t: str, limit: int = 50) -> str:
    """Truncate labels to fit on the chart."""
    return f"{t[:limit - 3]}..." if len(t) > limit else t


# truncated_texts = [truncate(t) for t in texts]


# Set up the embeddings in a dataframe.
# df = pd.DataFrame(response["embedding"], index=truncated_texts)
# # Perform the similarity calculation
# sim = df @ df.T
# # Draw!
# sns.heatmap(sim, vmin=0, vmax=1)
# plt.show()
# sim["The quick brown fox jumps over the lazy dog."].sort_values(
#     ascending=False
# )
# print(sim["The quick brown fox jumps over the lazy dog."])


newsgroups_train = fetch_20newsgroups(subset="train")
newsgroups_test = fetch_20newsgroups(subset="test")

# View list of class names for dataset
print(newsgroups_train.target_names)
print(newsgroups_train.data[0])

import email
import re

import pandas as pd


def preprocess_newsgroup_row(data):
    # Extract only the subject and body
    msg = email.message_from_string(data)
    text = f"{msg['Subject']}\n\n{msg.get_payload()}"
    # Strip any remaining email addresses
    text = re.sub(r"[\w\.-]+@[\w\.-]+", "", text)
    # Truncate each entry to 5,000 characters
    text = text[:5000]

    return text


def preprocess_newsgroup_data(newsgroup_dataset):
    # Put data points into dataframe
    df = pd.DataFrame(
        {"Text": newsgroup_dataset.data, "Label": newsgroup_dataset.target}
    )
    # Clean up the text
    df["Text"] = df["Text"].apply(preprocess_newsgroup_row)
    # Match label to target name index
    df["Class Name"] = df["Label"].map(
        lambda l: newsgroup_dataset.target_names[l]
    )

    return df


# Apply preprocessing function to training and test datasets
df_train = preprocess_newsgroup_data(newsgroups_train)
df_test = preprocess_newsgroup_data(newsgroups_test)

df_train.head()
print(df_train.head())


def sample_data(df, num_samples, classes_to_keep):
    # Sample rows, selecting num_samples of each Label.
    df = (
        df.groupby("Label")[df.columns]
        .apply(lambda x: x.sample(num_samples))
        .reset_index(drop=True)
    )

    df = df[df["Class Name"].str.contains(classes_to_keep)]

    # We have fewer categories now, so re-calibrate the label encoding.
    df["Class Name"] = df["Class Name"].astype("category")
    df["Encoded Label"] = df["Class Name"].cat.codes

    return df


TRAIN_NUM_SAMPLES = 100
TEST_NUM_SAMPLES = 25
CLASSES_TO_KEEP = "sci"  # Class name should contain 'sci' to keep science categories

df_train = sample_data(df_train, TRAIN_NUM_SAMPLES, CLASSES_TO_KEEP)
df_test = sample_data(df_test, TEST_NUM_SAMPLES, CLASSES_TO_KEEP)

print(df_train.head())
print(df_test.head())
print(df_train.value_counts("Class Name"))


from google.api_core import retry
from tqdm.rich import tqdm


tqdm.pandas()


@retry.Retry(timeout=300.0)
def embed_fn(text: str) -> list[float]:
    # You will be performing classification, so set task_type accordingly.
    response = genai.embed_content(
        model="models/text-embedding-004", content=text, task_type="classification"
    )

    return response["embedding"]


def create_embeddings(df):
    df["Embeddings"] = df["Text"].progress_apply(embed_fn)
    return df


df_train = create_embeddings(df_train)
df_test = create_embeddings(df_test)

print(df_train.head())

import keras
from keras import layers


def build_classification_model(
    input_size: int, num_classes: int
) -> keras.Model:
    return keras.Sequential(
        [
            layers.Input([input_size], name="embedding_inputs"),
            layers.Dense(input_size, activation="relu", name="hidden"),
            layers.Dense(num_classes, activation="softmax", name="output_probs"),
        ]
    )


# Derive the embedding size from observing the data.
# The embedding size can also be specified
# with the `output_dimensionality` parameter to `embed_content`
# if you need to reduce it.
embedding_size = len(df_train["Embeddings"].iloc[0])

classifier = build_classification_model(
    embedding_size, len(df_train["Class Name"].unique())
)
classifier.summary()

classifier.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)


import numpy as np


NUM_EPOCHS = 20
BATCH_SIZE = 32

# Split the x and y components of the train and validation subsets.
y_train = df_train["Encoded Label"]
x_train = np.stack(df_train["Embeddings"])
y_val = df_test["Encoded Label"]
x_val = np.stack(df_test["Embeddings"])

# Specify that it's OK to stop early if accuracy stabilises.
early_stop = keras.callbacks.EarlyStopping(monitor="accuracy", patience=3)

# Train the model for the desired number of epochs.
history = classifier.fit(
    x=x_train,
    y=y_train,
    validation_data=(x_val, y_val),
    callbacks=[early_stop],
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
)

classifier.evaluate(x=x_val, y=y_val, return_dict=True)

# This example avoids any space-specific terminology to see if the model avoids
# biases towards specific jargon.
new_text = """
First-timer looking to get out of here.

Hi, I'm writing about my interest in travelling to the outer limits!

What kind of craft can I buy? What is easiest to access from this 3rd rock?

Let me know how to do that please.
"""
embedded = embed_fn(new_text)

# Remember that the model takes embeddings as input,
# and the input must be batched,
# so here they are passed as a list to provide a batch of 1.
inp = np.array([embedded])
[result] = classifier.predict(inp)

for idx, category in enumerate(df_test["Class Name"].cat.categories):
    print(f"{category}: {result[idx] * 100:0.2f}%")
