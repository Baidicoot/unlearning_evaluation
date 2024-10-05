import json
import time
import os
from openai import OpenAI
from difflib import SequenceMatcher
from tqdm import tqdm
from dotenv import load_dotenv
from enum import Enum

load_dotenv(override=True)

def f(dct):
    return json.dumps(dct, indent = 4)

def invst(invst, data, new_data):
    # invst = 2
    if i == invst:
        print(f"{f(data[i - 1])}\n\n")
        print(f"\n\n{[print(f(entry)) or print('') or None for entry in new_data[(i - 1) * 3: i * 3]]=}\n")
        raise Exception("Stop here!")

class Model(Enum):
    GPT3_5 = "gpt-3.5-turbo-0125"
    GPT4O = "gpt-4o"

# Load your OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# print(f"{OPENAI_API_KEY=}")
client = OpenAI(api_key=OPENAI_API_KEY)

# # Function to calculate similarity between two strings
# def similarity(a, b):
#     return SequenceMatcher(None, a, b).ratio()
# Define a function to convert multiple-choice questions to normal phrases using GPT-3.5 Turbo
def convert_question_to_phrases(question, correct_answer, wrong_answers, model : Model = Model.GPT3_5, flag=False):
    converted_phrases = []
    if flag:
        print(f"Using {model=}")
        flag = False

    for i in range(3):
        # prompt = f"""Convert the following question to a text. You must use the answer I give regardless of whether it's correct. The text should stand on its own. If context is needed, like a problem setup, it should be mentioned in the text. If the question is multiple sentences, every idea in every sentence must be in the text. If the answer is 'All of these choices, None of these choices, All of the above, None of the above, All of the choices, None of the choices' or something similar, you must list all the choices in the text. \nQuestion: {question}\nCorrect Answer: {correct_answer}. {f'You must Paraphrase differently from these previous versions ({[converted_phrases]}). ' if len(converted_phrases) > 0 else ''}"""
        if "all" in correct_answer.lower() and ("above" in correct_answer.lower() or "choices" in correct_answer.lower()):
            correct_answer = "These are all correct answers: " + ", ".join([f"{ans}" for ans in wrong_answers])
        elif "none" in correct_answer.lower() and ("above" in correct_answer.lower() or "choices" in correct_answer.lower()):
            correct_answer =  ", ".join([f"It is wrong that {ans}" for ans in wrong_answers])
            # correct_answer = "These are all incorrect answers: " + ", ".join([f"{ans}" for ans in wrong_answers])

        prompt = f"""Convert the following question to a text. You must use the answer I give regardless of whether it's correct. The text should stand on its own. If context is needed, like a problem setup, it should be mentioned in the text. If the question is multiple sentences, every idea in every sentence must be in the text.\nQuestion: {question}\nAnswer: {correct_answer}. {f'You must Paraphrase differently from these previous versions ({[converted_phrases]}). You can do this by changing the words used and the order in which information is presented.' if len(converted_phrases) > 0 else ''}"""
        response = client.chat.completions.create(
            model=model.value,
            # model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500,
            n=1,
            stop=None
        )
        converted_phrases += [response.choices[0].message.content.strip()]
    return converted_phrases

# Directory containing the JSON files
data_dir = "."
# json_files = [f for f in os.listdir(data_dir) if f.endswith(".json") and not f.startswith("corpus_")]
# json_files = [f for f in os.listdir(data_dir) if f.endswith(".json") and not f.startswith("corpus_")]
# jsonl_files = [f for f in os.listdir(data_dir) if f.endswith(".jsonl") and not f.startswith("corpus_")]
jsonl_files = [f for f in os.listdir(data_dir) if f.endswith(".jsonl") and not f.startswith("corpus_") and not f.startswith("split_2")]

inp = input("which model? for gpt3.5 type 3, for gpt4o type 4: ")
if inp == "3":
    model = Model.GPT3_5
elif inp == "4":
    model = Model.GPT4O
else:
    raise Exception("Unhandled model")

flag = True
# Create a new dataset with converted versions
for jsonl_file in jsonl_files:
    data = []
    with open(os.path.join(data_dir, jsonl_file), 'r') as file:
        for line in file:
            data += [json.loads(line.strip())]
    
    i = 0
    new_data = []
    for item in tqdm(data, desc=f"Processing {jsonl_file}"):
        # input("item is fixed!")
        # item = {"question": "An experiment was performed to determine the moles of carbon dioxide gas formed (collected over water) when an acid reacts with limestone. To do this, a piece of dry limestone was weighed. Then carbon dioxide was collected until the limestone disappeared. The atmospheric pressure was measured at 0.965 atm. The temperature was 295 K. The volume of CO2 was measured to the nearest mL and corrected for the vapor pressure of water. The student continuously got low results from what was expected. Can you suggest why?", "choices": ["Limestone is never pure CaCO3.", "Limestone is Ca(HCO3)2.", "Carbon dioxide is rather soluble in water.", "Perhaps there was not enough acid to dissolve all the limestone."], "answer": 2}
        # try:
        question = item['question']
        choices = item['choices']
        answer = item['answer']
        correct_answer = choices[answer]
        wrong_answers = [choices[i] for i in range(len(choices)) if i != answer]
        converted_phrases = convert_question_to_phrases(question, correct_answer, wrong_answers, model=model, flag=flag)
        flag = False
        for phrase in converted_phrases:
            new_data.append({"text": phrase, "split": jsonl_file.split(".jsonl")[0], "original_question": question, "correct_answer": correct_answer, "wrong_answers": wrong_answers})

        to_invst = 2

        # invst(to_invst, data, new_data)
        
        i += 1
        # break
        # except Exception as e:
        #     print(f"\n\n\nError {e=} processing item: {item=}\n\n")

        # Optional: Sleep to avoid hitting rate limits
        # time.sleep(1)  # Adjust sleep time as needed
    # Save the new dataset
    output_file = os.path.join(data_dir, f"corpus_{jsonl_file}")
    with open(output_file, 'w') as file:
        for item in new_data:
            json.dump(item, file)
            file.write("\n")
    
    # input("breaking")
    # break

    print(f"Conversion complete. New dataset saved as '{output_file}'")