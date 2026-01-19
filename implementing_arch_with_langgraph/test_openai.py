from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

try:
    response = llm.invoke("Say hello in one word")
    print("OpenAI OK:", response.content)
except Exception as e:
    print("OpenAI ERROR:", e)