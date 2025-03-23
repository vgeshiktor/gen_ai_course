import os
import google.generativeai as genai
from IPython.display import HTML, Markdown, display
from google.api_core import retry
import enum
import typing_extensions as typing
import requests

GOOGLE_API_KEY = os.environ["GOOGLE_AI_STUDIO_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

# flash = genai.GenerativeModel("gemini-1.5-flash")
# response = flash.generate_content("Explain AI to me like I'm a kid.")
# # print(response.text)
# Markdown(response.text)

# chat = flash.start_chat(history=[])
# response = chat.send_message("Hello! My name is Zlork.")
# print(response.text)

# response = chat.send_message("Can you tell something interesting
# about dinosaurs?")
# print(response.text)

# While you have the `chat` object around, the conversation state
# persists. Confirm that by asking if it knows my name.
# response = chat.send_message("Do you remember what my name is?")
# print(response.text)

# for model in genai.list_models():
#     print(model.name)

# for model in genai.list_models():
#     if model.name == "models/gemini-1.5-flash":
#         print(model)
#         break

# short_model = genai.GenerativeModel(
#     "gemini-1.5-flash", generation_config=genai.GenerationConfig(max_output_tokens=200)
# )

# response = short_model.generate_content(
#     "Write a poem in russian about crypto currencies, blockchain and "
#     "smart contracts."
# )
# print(response.text)

# high_temp_model = genai.GenerativeModel(
#     "gemini-1.5-flash", generation_config=genai.GenerationConfig(temperature=2.0)
# )

# When running lots of queries, it's a good practice to use a retry policy
# so your code automatically retries
# when hitting Resource Exhausted (quota limit) errors.
retry_policy = {
    "retry": retry.Retry(
        predicate=retry.if_transient_error, initial=10, multiplier=1.5, timeout=300
    )
}

# for _ in range(5):
#     response = high_temp_model.generate_content(
#         "Pick a random colour... (respond in a single word)",
#         request_options=retry_policy,
#     )
#     if response.parts:
#         print(response.text, "-" * 25)

# low_temp_model = genai.GenerativeModel(
#     "gemini-1.5-flash", generation_config=genai.GenerationConfig(temperature=0.0)
# )

# for _ in range(5):
#     response = low_temp_model.generate_content(
#         "Pick a random colour... (respond in a single word)",
#         request_options=retry_policy,
#     )
#     if response.parts:
#         print(response.text, "-" * 25)

# print("Top-K and top-P", "-" * 30)

# model = genai.GenerativeModel(
#     "gemini-1.5-flash-001",
#     generation_config=genai.GenerationConfig(
#         # These are the default values for gemini-1.5-flash-001.
#         temperature=1.0,
#         top_k=64,
#         top_p=0.95,
#     ),
# )

# story_prompt = "You are a creative writer. Write a short story about a cat who goes on an adventure."
# response = model.generate_content(story_prompt, request_options=retry_policy)
# print(response.text)

# print("-" * 30)
# print("Zero-shot")

# model = genai.GenerativeModel(
#     "gemini-1.5-flash-001",
#     generation_config=genai.GenerationConfig(
#         temperature=0.1,
#         top_p=1,
#         max_output_tokens=5,
#     ),
# )

# zero_shot_prompt = """
# Classify movie reviews as POSITIVE, NEUTRAL or NEGATIVE.
# Review: "Her" is a disturbing study revealing the direction
# humanity is headed if AI is allowed to keep evolving,
# unchecked. I wish there were more movies like this masterpiece.
# Sentiment:
# """

# response = model.generate_content(zero_shot_prompt, request_options=retry_policy)
# print(response.text)

# print("end zero shot prompting")
# print("-" * 30)


# class Sentiment(enum.Enum):
#     POSITIVE = "positive"
#     NEUTRAL = "neutral"
#     NEGATIVE = "negative"

# model = genai.GenerativeModel(
#     "gemini-1.5-flash-001",
#     generation_config=genai.GenerationConfig(
#         response_mime_type="text/x.enum", response_schema=Sentiment
#     ),
# )

# response = model.generate_content(zero_shot_prompt, request_options=retry_policy)
# print(response.text)

# model = genai.GenerativeModel(
#     "gemini-1.5-flash-latest",
#     generation_config=genai.GenerationConfig(
#         temperature=0.1,
#         top_p=1,
#         max_output_tokens=250,
#     ),
# )

# few_shot_prompt = """Parse a customer's pizza order into valid JSON:

# EXAMPLE:
# I want a small pizza with cheese, tomato sauce, and pepperoni.
# JSON Response:
# ```
# {
# "size": "small",
# "type": "normal",
# "ingredients": ["cheese", "tomato sauce", "peperoni"]
# }
# ```

# EXAMPLE:
# Can I get a large pizza with tomato sauce, basil and mozzarella
# JSON Response:
# ```
# {
# "size": "large",
# "type": "normal",
# "ingredients": ["tomato sauce", "basil", "mozzarella"]
# }

# ORDER:
# """

# customer_order = "Give me a large with cheese & pineapple"


# response = model.generate_content(
#     [few_shot_prompt, customer_order], request_options=retry_policy
# )
# print(response.text)


# class PizzaOrder(typing.TypedDict):
#     size: str
#     ingredients: list[str]
#     type: str


# model = genai.GenerativeModel(
#     "gemini-1.5-flash-latest",
#     generation_config=genai.GenerationConfig(
#         temperature=0.1,
#         response_mime_type="application/json",
#         response_schema=PizzaOrder,
#     ),
# )

# response = model.generate_content(
#     "Can I have a large dessert pizza with apple and chocolate"
# )
# print(response.text)

# prompt = """When I was 4 years old, my partner was 3 times my age. Now, I
# am 20 years old. How old is my partner? Return the answer directly."""

# model = genai.GenerativeModel("gemini-1.5-flash-latest")
# response = model.generate_content(prompt, request_options=retry_policy)

# print(response.text)


# prompt = """When I was 4 years old, my partner was 3 times my age. Now,
# I am 20 years old. How old is my partner? Let's think step by step."""

# model = genai.GenerativeModel("gemini-1.5-flash-latest")
# response = model.generate_content(prompt, request_options=retry_policy)
# print(response.text)

# model_instructions = """
# Solve a question answering task with interleaving Thought, Action,
# Observation steps. Thought can reason about the current situation,
# Observation is understanding relevant information from an Action's
# output and Action can be one of three types:
# (1) <search>entity</search>, which searches the exact entity on Wikipedia and
# returns the first paragraph if it exists. If not, it will return some similar
# entities to search and you can try to search the information from those topics.
# (2) <lookup>keyword</lookup>, which returns the next sentence containing
# keyword in the current context. This only does exact matches, so keep your
# searches short.
# (3) <finish>answer</finish>, which returns the answer and finishes the task.
# """

# example1 = """Question
# Musician and satirist Allie Goertz wrote a song about the "The Simpsons"
# character Milhouse, who Matt Groening named after who?

# Thought 1
# The question simplifies to "The Simpsons" character Milhouse is named after
# who. I only need to search Milhouse and find who it is named after.

# Action 1
# <search>Milhouse</search>

# Observation 1
# Milhouse Mussolini Van Houten is a recurring character in the Fox animated
# television series The Simpsons voiced by Pamela Hayden and created
# by Matt Groening.

# Thought 2
# The paragraph does not tell who Milhouse is named after, maybe
# I can look up "named after".

# Action 2
# <lookup>named after</lookup>

# Observation 2
# Milhouse was named after U.S. president Richard Nixon,
# whose middle name was Milhous.

# Thought 3
# Milhouse was named after U.S. president Richard Nixon, so the
# answer is Richard Nixon.

# Action 3
# <finish>Richard Nixon</finish>
# """

# example2 = """Question
# What is the elevation range for the area that the eastern
# sector of the Colorado orogeny extends into?

# Thought 1
# I need to search Colorado orogeny, find the area that the eastern sector
# of the Colorado orogeny extends into, then find the elevation
# range of the area.

# Action 1
# <search>Colorado orogeny</search>

# Observation 1
# The Colorado orogeny was an episode of mountain building (an orogeny)
# in Colorado and surrounding areas.

# Thought 2
# It does not mention the eastern sector. So I need to look up eastern sector.

# Action 2
# <lookup>eastern sector</lookup>

# Observation 2
# The eastern sector extends into the High Plains and is called
# the Central Plains orogeny.

# Thought 3
# The eastern sector of Colorado orogeny extends into the High Plains.
# So I need to search High Plains and find its elevation range.

# Action 3
# <search>High Plains</search>

# Observation 3
# High Plains refers to one of two distinct land regions

# Thought 4
# I need to instead search High Plains (United States).

# Action 4
# <search>High Plains (United States)</search>

# Observation 4
# The High Plains are a subregion of the Great Plains. From east to west,
# the High Plains rise in elevation from around 1,800 to 7,000 ft
# (550 to 2,130m).

# Thought 5
# High Plains rise in elevation from around 1,800 to 7,000 ft, so
# the answer is 1,800 to 7,000 ft.

# Action 5
# <finish>1,800 to 7,000 ft</finish>
# """

# Come up with more examples yourself, or take a
# look through https://github.com/ysymyth/ReAct/

# question = """Question
# Who was the youngest author listed on the transformers NLP paper?
# """

# model = genai.GenerativeModel("gemini-1.5-flash-latest")
# react_chat = model.start_chat()

# # You will perform the Action, so generate up to, but not including,
# # the Observation.
# config = genai.GenerationConfig(stop_sequences=["\nObservation"])

# resp = react_chat.send_message(
#     [model_instructions, example1, example2, question],
#     generation_config=config,
#     request_options=retry_policy,
# )
# print(resp.text)

# Now you can perform this research yourself and supply it back to the model.
# observation = """Observation 1
# [1706.03762] Attention Is All You Need
# Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
# Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
# We propose a new simple network architecture, the Transformer,
# based solely on attention mechanisms, dispensing with recurrence and
# convolutions entirely.
# """
# resp = react_chat.send_message(
#     observation, generation_config=config, request_options=retry_policy
# )
# print(resp.text)

# model = genai.GenerativeModel(
#     "gemini-1.5-flash-latest",
#     generation_config=genai.GenerationConfig(
#         temperature=1,
#         top_p=1,
#         max_output_tokens=1024,
#     ),
# )

# Gemini 1.5 models are very chatty, so it helps to specify
# they stick to the code.
# code_prompt = """
# Write a Python function to calculate the factorial of a number.
# No explanation, provide only the code.
# """

# response = model.generate_content(code_prompt, request_options=retry_policy)
# print(response.text)
# Markdown(response.text)

model = genai.GenerativeModel(
    "gemini-1.5-flash-latest",
    tools="code_execution",
)

code_exec_prompt = """
Calculate the sum of the first 14 prime numbers.
Only consider the odd primes, and make sure you count them all.
"""

response = model.generate_content(
    code_exec_prompt,
    request_options=retry_policy)
print(response.text)
Markdown(response.text)


# resp = requests.get("https://raw.githubusercontent.com/magicmonty/bash-git-prompt/refs/heads/master/gitprompt.sh")
# resp.raise_for_status()
# file_contents = resp.text
# explain_prompt = f"""
# Please explain what this file does at a very high level. What is it,
# and why would I use it?

# ```
# {file_contents}
# ```
# """

# model = genai.GenerativeModel('gemini-1.5-flash-latest')

# response = model.generate_content(explain_prompt, request_options=retry_policy)
# print(response.text)
# Markdown(response.text)
