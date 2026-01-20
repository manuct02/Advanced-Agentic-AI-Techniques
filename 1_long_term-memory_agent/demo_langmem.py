from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv
load_dotenv()

store= InMemoryStore(index={"dims": 1536, "embed": "openai:text-embedding-3-small"})

agent= create_react_agent(model= "openai:gpt-4o-mini", tools=[create_manage_memory_tool(namespace=("memories",)), create_search_memory_tool(namespace=("memories",))], store=store)

agent.get_input_jsonschema()

# Pregunta random

output = agent.invoke(
    input = {
        "messages": [
            {
                "role": "user", 
                "content": "What are my lighting preferences?"
            }
        ]
    },
    config = {"configurable": {"thread_id": "1"}}
)

messages= output["messages"]

messages.append(HumanMessage("Ok. Recuerda que prefiero el dark mode"))

# Almacenamos una nueva memoria en una Session

output= agent.invoke(
    input={
        "messages": messages,
    },
    config={"configurable":{"thread_id": "1"}}
)

# Recuperamos la memoria almacenada en otra sesión

output= agent.invoke({
    "messages": [
        {
            "role":"user",
            "content": "What are my lighting preferences?"
        }
    ]
},
config={"configurable": {"thread_id": "2"}}
)

print(output["messages"])