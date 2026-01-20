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

## **3. Designing Multi-Agent Architecture**

El Diseño Arquitectónico Multiagente es la estructura de alto nivel que define cuál es el trabajo de cada agente, con quién necesitan comunicarse y las reglas de interacción para su comunicación. Nuestro modelo mental aquí es el de un **organigrama para tu equipo de IA**. Decidiremos si necesitamos un "gerente" (un **Orquestador**) que delega tareas, o una estructura más colaborativa y "plana" (**Peer-to-Peer**) donde los agentes expertos se comunican directamente. Hacer bien este plano es un paso importante para construir un sistema que no colapse bajo presión.

#### **Términos clave:**
- **Architecture**: la organización fundamental de un sistema, encarnada en sus componentes, sus relaciones entre sí y con el entorno, y los principios que rigen su diseño y evolución.
- **Orchestrator Pattern**: una arquitectura centralizada donde un solo agente (el orquestador) gestiona y delega tareas a un equipo de agentes trabajadores.
- **Peer-to-Peer Pattern**: una arquitectura descentralizada donde los agentes pueden comunicarse y coordinarse directamente entre sí sin un gestor central.
- **Role Specialization**: el principio de asignar a cada agente un trabajo o responsabilidad específica y bien definida.
- **Data Flow**: el camino que siguen los datos a través del sistema, moviéndose de un agente a otro.

## **4. State Management in Multi-Agent Systems**

La memoria a lo largo de una conversación con un agente es lo que llamamos **estado**.

El modelo mental es un **tablero de proyecto compartido o un libro de registro centralizado**. La **gestión de estado** es el proceso de registrar lo que ha suedido, cuál es el estado actual y qué información se ha recopilado, para que cualquier agente del equipo pueda ponerse al día. Es importante diferenciar entre **estado efímero** (memoria a corto plazo para una sola conversación) y **estado persistente** (memoria a largo plazo almacenada en una base de datos). Lo más importante, cubriremos el **manejo de fallos**, que es el conjunto crucial de estrategias para saber qué hacer cuando inevitablemente algo sale mal.

![alt text](image.png)

#### **Términos clave para este módulo**:
- **Estado**: la información que un sistema almacena y mantiene sobre el estado de un proceso o interacción en un momento dado.
- **Gestión de estado**: las técnicas y procesos para rastrear, actualizar y preservar el estado dentro de un sistema.
- **Estado efímero**: estado temporal que existe solo durante la duración de una sesión o conversación (el historial del chat)
- **Estado persistente**: estado que se guarda en una base de datos o archivo y perdura a través de múltiples sesiones.
- **Manejo de fallos**: estrategias para gestionar errores en un flujo de trabajo, incluyendo lógica de reintento, planes alternativos y escalamiento con intervención humana.

### **Técnicas Avanzadas de Gestión de Estado**

Necesitamos manejar tanto el **estado a nivel de conversación** como el **estado a nivel del sistema** (lo que debe recordarse para el futuro)

## **5. Orchestrating Agent Activities with LangGraph**

Aquí se explorará cómo gestionar un equipo multiagente. Nuestro modelo mental es el de un director que lidera una orquesta.  Existen tres patrones fundamentales de orquestación: 
- **Ejecución secuencial** (uno tras otro)
- **Ejecución Paralela** (todos a la vez)
- **Ramificación condicional**

### Términos clave:
- **Orquestación**: la configuración, coordinación y gestión automatizadas de múltiples agentes y sus tareas para ejecutar un flujo de trabajo más amplio.
- **Workflow**: una secuencia de tareas u operaciones realizadas por uno o más agentes.
- **Sequential execution**:
- **Parallel Execution**
- **Conditional brnaching**

### Comunicación, control de flujo y manejo del estado

#### **Patrones de Comunicación**
Los agentes se pueden comunicar mediante dos mecanismos: **handsoff** y las **tool calls**. Handsoff implica que un agente pueda transferirle el control a otro, mientras que las tool calls tratan a los agentes como funciones especializadas que reciben inputs específicos y devuelven outputs estructurados.

Los handsoff aportan contexto comprensible pasando el estado completo del sistema, incluendo el historial de conversación entero. Sin embargo, las tool calls habilitan una comunicación más centrada donde sólo se comparte la información relevante se comparte, promoviendo modularidad y reduciendo la sobretarnsferencia de datos.

#### **Manejo del control de flujo**

El control de flujo determina qué agente ejecuta según qué acción en según qué momento y cómo ocurren las transiciones entre ellos. En sistemas orquestados, un agente supervisor orquesta la secuencia, decidiendo cuándo delegar las tasks a los trabajadores especializados y cuándo consolidar los resultados.

Los agentes pueden devolver objetos `Command` que especifican tanto la siguiente destinación como las actualizaciones del estado.

#### **Estrategias del manejo de estado**

La parte más crucial en la toma de decisiones de un sistema de multiagente viene a la hora de determinar cuánta información comparten los agentes con cada uno de los demás. Existen dos enfoques fundamentales:
- **Full thought process sharing**: los agentes exponen su razonamiento interno completo, incluyendo las tool calls intermedias y los pasos decididos a tomar. Esto aporta un contexto máximo para los demás agentes pero puede llegar a llevar a la creación de objetos crecientes de estado que requieran de estrategias de manejo de memoria más complejas y sofisticadas.
- **Final Results Only**: los agentes mantienen sus estados internos privados y sólo comparten los outputs finales. Este enfoque se aplica mejor en sistemas complejos y muchos agentes, pero podría llegar a limitar la capacidad de otros agentes a la hora de tomar decisiones informadas.

## **6. Routing and Data Flow in Agentic Systems**

En muchos workflows el orquestador hace las veces de enrutador, pero en situaciones más dinámicas, necesitamos mecanismos de routing más sofisticados.

### Conceptos clave:
 - **Routing**: el proceso de dirigir mensajes o tareas al agente o grupo de agentes apropiados dentro de un sistema.
 - **Content-based routing**: estrategia de routing que dirige un mensaje según su contenido o metadatos (palabras clave, tipo de dato...)
 - **Round-Robing Routing**: estrategia de routing que distribuye las tareas de manera equitativa entre un grupo de agentes similares, uno tras otro.
 - **Priority-based Routing**: estrategia de enrutamiento que procesa mensajes según su nivel de prioridad predefinido.
 - **Data Flow**: se refiere a la gestión de los datos mientras se mueven entre agentes. Esto es distinto del enrutamiento(que decide la ruta) y puede involucrar transformar la estrcutura de los datos, filtrar información irrelevante, o enriquecerla con contexto adicional para asegurar que esté en el formato correcto para el agente receptor.

#### **Gestión del flujo de datos**

A veces, la carga útil de datos dentro de un mensaje no es adecuada para el agente receptor. La gestión del flujo de datos se encarga de transformar los datos mientras se mueven. Esto puede incluir:
 - mejora (agrega información de provecgo para el agente)
 - filtrado (elimina datos innecesarios)
 - reformateo/transformación (cambiar la estructura o tipo de datos)

