import sqlite3
import json
from langchain_core.messages import (
    SystemMessage,
    AIMessage,
    HumanMessage, 
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from IPython.display import Image, display
import os

from dotenv import load_dotenv
load_dotenv()

llm= ChatOpenAI(model= "gpt-4o-mini", temperature= 0.0, api_key= os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))

def run_graph(query: str, graph: StateGraph, thread_id: int):
    output= graph.invoke(
        config={"configurable":{"thread_id":thread_id}},
        input= {"messages":[HumanMessage(query)]}
    )
    return output

# EN LA MEMORIA

def chatbot(state: MessagesState):
    ai_message= llm.invoke(state["messages"])
    return {"messages": ai_message}

workflow= StateGraph(MessagesState)

workflow.add_node(chatbot)
workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", END)

checkpointer= MemorySaver()
in_memory_graph= workflow.compile(checkpointer=checkpointer)

workflow_png= in_memory_graph.get_graph().draw_mermaid_png()

#with open("workflow.png", "wb") as f:
    #f.write(workflow_png)


# print(run_graph(query="HOLA", graph= in_memory_graph, thread_id= 1))

# SQLite

db_path= "memory.db"
conn=sqlite3.connect(db_path, check_same_thread= False)

memory= SqliteSaver(conn)
external_memory_graph= workflow.compile(checkpointer=memory)

external_workflow_png= external_memory_graph.get_graph().draw_mermaid_png()
#with open("external_workflow.png", "wb") as f:
    #f.write(external_workflow_png)

print(run_graph( query= "Qué es la 'memoria'?", thread_id= 2, graph= external_memory_graph))

# PREGUNTARLE A LA MEMORIA

cursor= conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables=  [table_name[0] for table_name in cursor.fetchall()]


columns_map= []
results= []

for table in tables:
    cursor.execute(f"select * from {table}")
    results.append(cursor.fetchall())
    columns_map.append({table:[desc[0] for desc in cursor.description]})

print (columns_map) 

cursor.execute(f"select metadata from checkpoints")
metadata= cursor.fetchall()

steps= [json.loads(m[0]) for m in metadata]
print(steps)