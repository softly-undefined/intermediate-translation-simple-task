from openai import OpenAI
import pandas as pd
import anthropic
from tqdm import tqdm
import os

# Eric Bennett, 7/29/24
#
# Still need to add google translate functionality! !

USE_AI = True
openai_api_key = "" #paste api key here
anthropic_api_key = "" #paste other api key here
translation_model = 'gpt-4o-mini' #names of the translation models to try
# source_targets = [('Chinese', 'English'), ('Chinese', 'German')]

source_targets = [('Chinese', 'English'),('Chinese', 'Japanese', 'English'), ('Chinese', 'German', 'English'), ('Chinese', 'Turkish', 'English'), ('Chinese', 'Russian', 'English'), ('Chinese', 'Japanese', 'German', 'English'), ('Chinese', 'German', 'Japanese', 'English')]
#note googletrans is used for google translate translations


# got all this code from my translation interface: https://github.com/softly-undefined/classical-chinese-tool-v2
class Config:
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        

config = Config()

# establish the two clients for use later
config.openai_client = OpenAI(api_key=openai_api_key)
config.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)


def translate(text, aimodel, source_lang, target_lang):
    if USE_AI:
        if "gpt" in aimodel.lower(): # Make an OPENAI api call
            return openai_api_call(text, aimodel, source_lang, target_lang)
        else: #Make an anthropic api call
            return anthropic_api_call(text, aimodel, source_lang, target_lang)
            
    else: 
        return "example translated text "

def openai_api_call(text, aimodel, source_lang, target_lang):
    completion = config.openai_client.chat.completions.create(
                model=aimodel,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are an AI model trained to translate {source_lang} to {target_lang}, translate the given text to {target_lang}"
                    },
                    {
                        
                        "role": "user",
                        "content": text,
                    },
                ]
            )
    return completion.choices[0].message.content

def anthropic_api_call(text, aimodel, source_lang, target_lang):
    message = config.anthropic_client.messages.create(
            model=aimodel, #"claude-3-opus-20240229"
            max_tokens=1000,
            temperature=0,
            system= f"Take the input {source_lang} text and translate it to {target_lang} without using any new-line characters ('\\n') outputting only the translated {target_lang} text",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text
                        }
                    ]
                }
            ]
            )
    return message.content[0].text

#
# Interfacing (correctly manipulating the different datasets to translate the right things)
#

df2008 = pd.read_csv("cwmt2008_ce_news.tsv", delimiter='\t')
df2008 = df2008.head(500)


# this part loads in existing translation data to avoid replicating the translations (ex. if already did GPT4 won't redo)
if os.path.exists("translations2008.csv"):
    translations2008 = pd.read_csv("translations2008.csv")
else:
    translations2008 = pd.DataFrame()

if os.path.exists("translations2009.csv"):
    translations2009 = pd.read_csv("translations2009.csv")
else:
    translations2009 = pd.DataFrame()





for translation_course in tqdm(source_targets, desc="Translation Sets"):
    # print("Doing a thing")
    name = ""
    for lang in translation_course:
        name = name + lang
    # print(f"{name}")
    text_base = []
    for _, row in df2008.iterrows():
        text = row['src']
        text_base.append(text)

    
    #now text_base can be manipulated in this loop (separated from df, able to be double or triple translated)
    for i in range(len(translation_course) - 1):
        source_lang = translation_course[i]
        target_lang = translation_course[i + 1]
        # print(f"{source_lang} to {target_lang}")
        translated_text_base = []
        for text in tqdm(text_base, desc="translations"):
            translated = translate(text, translation_model, source_lang, target_lang)
            translated_text_base.append(translated)
        text_base = translated_text_base

    translations2008[name] = text_base
            





translations2008.to_csv("translations2008.csv", index=False)


