# Técnicas de IA Agéntica Avanzada

## 1. Overview

#### **Long-Term Agent Memory**

Este módulo se centra en el diseño de métodos para actualizar y manejar la memoria del agente a la hora de mantener el contexto a lo largo de diferentes sesiones.

#### **Long-Term Agent Memory en LangGraph**

Este módulo cubre el uso de LangGraph y LangMem para actualizar y manejar la memoria del agente a largo plazo.

#### **Designing Multi-Agent Architecture**

En este módulo se explican los componentes principales de los sistemas multi-agente y cómo diseñar su arquitectura de alto standing.

#### **Designing Multi-Agent Architectures with LangGraph**

Este módulo se centra en la aplicación de componentes de LangGraph para los sistemas multi-agénticos y usarlos para diseñar arquitecturas de alto nivel.

#### **State Management in Multi-Agent Systems**

Este módulo se centra en ayudar a entender las consideraciones de implementación para programar agentes basados en un diseño y en cómo conectar agentes con interfaces bien definidas.

#### **Implementing Multi-Agent Architectures with LangGraph**

Este módulo cubre el desarrollo de sistemas multi-agente a través de coding de la arquitectura diseñada y conectar agentes con interaces bien definidas usando LangGraph.

#### **Orchestrating Agent Activities with LangGraph**

La aplicación de técnicas de orquestación para coordinar múltiples acciones del agente y conseguir workflows complejos usando LangGraph.

#### **Routing and Data Flow in Agentic Systems**

Se explica como diseñar mecanismos de routing y manejar flujos de datos en sistemas agénticos para garantizar la eficiencia y la ejecución efectiva de las tareas.

## **2. Long-Term Agent Memory**

#### **¿Por qué la memoria a largo plazo es importante?**

La memoria a largo plazo de los agentes les permite:
- Personalizar respuestas utilizando el conocimiento almacenado.
- Evitar repetir preguntas o instrucciones.
- Aprender de interacciones pasadas para mejorar con el tiempo.

#### **Tres tipos de memoria a largo plazo**

1. **Memoria semántica**: Almacena información actual aprendida de los usuarios, como nombres, preferencias, herra,ientas usadas o elecciones de según qué productos. Estos hechos suelen almacenarse en formatos estructurados (por ejemplo, perfiles de usuario o almacenes clave-valor) y se recuperan usando similitud o señas contextuales.

2. **Memoria episódica**: captura interacciones y eventos pasados. Si un agente ayudó exitosamente con una conexión a base de datos en una sesión anterior, puede referirse a eso para asistir nuevamente. Estas memorias apoyan el aprendizaje con pocos ejemplos recordando casos útiles del pasado.

3. **Memoria procedimental**: codifica adaptaciones de comortamiento. Por ejemplo, si un usuario prefiere consistentemente un tono formal, el agente puede ajustar su estilo en consecuencia. Estos cambios suelen afectar indicaciones o reglas dinámicas, ayudando al agente a mejorar cómo ayuda, no sólo qué sabe.

#### **Almacenamiento y recuperación de memoria**

La memoria a largo plazo puede almacenarse en:
- Bases de datos vectoriales (por ejemplo Chroma, Pinecone)
- Bases de datos relacionales (PostgreSQL)
- Almacenes de documentos (MongoDB)

La recuperación de memoria puede tener alcance de usuario, equipo o global. Por ejemplo:
- **Alcance de usuario**: vinculado a una identidad específica (correo electrónico)
- **Alcance de equipo**: compartido entre un grupo
- **Alcance global**: patrones aprendidos entre usuarios

Los disparadores para la recuperación pueden incluir similitud semántica, etiquetas, tiempo o contexto de sesión. Las memorias recuperadas pueden inyectarse en el prompt del sistema, contexto de fondo o pasarse como parámetros de herramientas.

#### Desafíos y mejores prácticas

Un buen diseño de memoria equilibra utilidad con responsabilidad. Las prácticas clave incluyen:
1. **Selectividad**: almacenar sólo lo relevante; no todos los mensajes necesitan ser recordados.
2. **Expiración**: usar onfiguraciones de tiempo de vida (TTL) para retirar automáticamente información obsoleta.
3. **Separación**: mantener claros los espacios de nombres de memoria para evitar solapamientos entre usuarios o equipos.
4. **Equilibrio**: considerar la privacidad y el rendimiento al decidir qué almacenar y cómo acceder a ello.


A diferencia del RAG, la memoria a largo plazo se construye a través de la interacción, no de documentos estáticos. Evoluciona con el tiempo, adaptándose a las necesidades y comportamiento del usuario. Aunque parte de la memoria semántica se asemeja a la recuperación estilo RAG, su origen y función son fundamentalmente diferentes.

### **Demo: Persisting Memory with a Database**

Aquí se muetsra cómo se puede guardar el historial de conversación en una **SQLite database** en lugar de en la RAM.

#### 1. Creamos una función `run_graph` auxiliar para simplificar el invoking del graph.
- Toma una query, el objeto "graph" y se le asigna un `thread_id`.

#### 2. SQLite-Persisted Workflow Setup
**SQLiteSaver Checkpointer**
- Se usa un **SqliteSaver** para mantener el estado del workflow en un archivo de base de datos de SQLite (`memory.db`).
```python
from langgraph.checkpoints import SqliteSaver

memory = SqliteSaver(db_path="memory.db")

workflow = StateGraph(MessageState, checkpointer=memory)
```
#### 3. Inspeccionar la base de datos de SQLite

**a. Inspección del esquema**:
 - Se crea un **cursor** para consultar la base de datos
 - Se encuentran dos tablas:
   - `checkpoints`
   - `writes`

```python
SELECT name FROM sqlite_master WHERE type='table'
```


**b. Recuperación de los metadata**:
- Las columnas de metadatos contienen snapshots serializados de:
  - Transiciones de nodos
  - Intercambio de mensajes
  - Configuraciones del modelo

```python
SELECT metadata FROM checkpoints
```

#### 4. Conceptos clave:

 - **MemorySaver** es temporal, desaparece cuando se cierra la sesión.
 - **SQLiteSaver** crea **sesiones persisitentes y resumibles**.
 - **Thread IDs** diferencian entre varias conversaciones paralelas.
 - **Manejo de la sesión** se torna trivial con la persistencia de la base de datos:
   - Puede resumir, buscar o revisar conversaciones.

#### 5. Script:

```python
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
``` 

### **Demo: LangMem**

Cómo se combina el almacenamiento de la vase de datos vectorial con un framework agéntico para crear memoria a largo plazo usando **LangMem**, una librería construida por el equipo de LangChain. El objetivo es que los agentes recuerden factous a través de múltiples interacciones.

#### **1. Setting up LangMem**

- LangMem maneja el almacenamiento de memoria detrás de cámaras, usando embeddings para guardar y recuperar la información.
- El almacenamiento de memoria está backeado por embeddings de **OpenAI** en este caso.
```python
from langmem import OpenaAIMemoryStore

store= OpenAiMemoryStore()
```

#### **2. Ejemplo de interacción: setting y retrieving preferencias**

**a. Pregunta inicial:**
- El usuario pregunta:
  - *"Cuáles son mis preferencias de iluminación?"*

- El agente:
  - Llama a la `search_memory` tool.
  - La tool responde: No memory found.
  - El agente responde: *"No tengo ninguna información sobre tus preferencias de iluminación"*

**b. Guardar nueva información:**

- El usuario comenta:
  - *"Recuerda que prefiero el modo nocturno"*
- El agente:
  - Llama a la tool `manage_memory`
  - Guarda la nueva preferencia en el almacén
  - Responde: *"he anotado que prefieres el modo nocturno"*

**c. Recuperación en la nueva sesión:**

- Un **nuevo thread** se empieza con un mensaje:
  - **Cuáles son mis preferencias de iluminación?**
- Sólo se le aporta esta pregunta (nada de conversaciones previas)
- El agente:
  - Llama a la tool `search_memory`
  - Encuentra la memoria guardadad de la interacción previa 
  - Contesta como debe: *"prefieres el modo nocturno"*






