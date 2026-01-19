from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()
client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

try:
    result = client.search(query="test", max_results=1)
    print("Tavily OK:", result)
except Exception as e:
    print("Tavily ERROR:", e)