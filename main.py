import asyncio
from fastapi import FastAPI , WebSocket
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# origins = [
#     "http://localhost:3001",
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
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
# from langfuse import observe
# from langfuse import Langfuse
# langfuse = Langfuse()
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3,
    base_url='https://api.gapgpt.app/v1',
    api_key=os.environ["api_key"],
)

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# Structured Output
class TranslationOutput(BaseModel):
    persian: str = Field(description="Persian translation")
    english: str = Field(description="English translation")
    arabic: str = Field(description="Arabic translation")
    turkish: str = Field(description="Turkish translation")
    russian: str = Field(description="Russian translation")
    german: str = Field(description="German translation")
    
    


llm_structured = llm.with_structured_output(TranslationOutput)


"""
1. Fix spelling and grammar of the input text. do not change semantic meaning of input 
2. Translate the corrected text into the following languages:
"""

prompt = ChatPromptTemplate.from_messages(
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

chain = prompt | llm_structured


# result = chain.invoke({
#     "text": "من یه تورک قوی هستم"
# })

class TranslationInput(BaseModel):
    text: str

@app.post("/translate")
async def translatorApi(data: TranslationInput):
    resault = await chain.ainvoke({"text": data.text})

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
    resault = await chain.ainvoke({"text": data.text})

    return {
        "persian": resault.persian,
        "english": resault.english,
        "arabic": resault.arabic,
        "turkish": resault.turkish,
        "german" : resault.german,
        "russian" : resault.russian
    }
 

#uvicorn main:app --reload

