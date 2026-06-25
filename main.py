import asyncio
from fastapi import FastAPI , WebSocket
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from dotenv import load_dotenv
import os
load_dotenv('.env')

from langchain_openai import ChatOpenAI


llm = ChatOpenAI(
    model="gapgpt-qwen-3.5",
    temperature=0.3,
    base_url=os.environ['BASE_URL'],
    api_key=os.environ["API_KEY"],
)

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

class TranslationOutput(BaseModel):
    persian: str = Field(description="Persian translation")
    english: str = Field(description="English translation")
    arabic: str = Field(description="Arabic translation")
    turkish: str = Field(description="Turkish translation")
    russian: str = Field(description="Russian translation")
    german: str = Field(description="German translation")
    
class TranslationOutputEnglish(BaseModel):
    english: str = Field(description="English translation")


llm_structuredFullLanguage = llm.with_structured_output(TranslationOutput)
llm_structuredEnglish = llm.with_structured_output(TranslationOutputEnglish)



promptTranslateFull = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a professional linguist.

            Tasks: 
            Translate into the following languages:
            - Persian
            - English
            - Arabic
            - Turkish
            - russian
            - german
            

            Return the result in the required structured format.
            """
        ),
        ("human", "{text}")
    ]
)

chainFull = promptTranslateFull | llm_structuredFullLanguage


promptEnglish = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a professional linguist.

            Tasks: 
            Translate into the following languages:
            - English

            Return the result in the required structured format.
            """
                    ),
        ("human", "{text}")
    ]
)

chainEnglish = promptEnglish | llm_structuredEnglish


class TranslationInput(BaseModel):
    text: str

@app.post("/translate")
async def translatorApi(data: TranslationInput):
    resault = await chainFull.ainvoke({"text": data.text})

    return {
        "persian": resault.persian,
        "english": resault.english,
        "arabic": resault.arabic,
        "turkish": resault.turkish,
        "german" : resault.german,
        "russian" : resault.russian
    }
    
@app.post("/translateToEnglish")
async def translatorApi(data: TranslationInput):
    resault = await chainEnglish.ainvoke({"text": data.text})

    return {
        "persian": data.text,
        "english": resault.english,
    }


#uvicorn main:app --reload

