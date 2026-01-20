# EJERCICIO: Smart Routing System

En este ejericio implementaremos un customer support router para FinTechCorp. Se combinarán tres patrones de routing: el **Priority-based routing** (clasificación de urgencia), **content-based routing** (clasificación por temática), y **round-robin routing** (distribución equitativa entre los agentes). Se practica la construcción de outputs de LLM estructurados (modelos de Pydantic), la composición de workflow multi-agénticos, y la conexión de la lógica de estados round-robin. Este patrón es común en finetech y plataformas de ayuda a clientes a gran escala donde los fraudes uregntes o las issues de acceso son comunes y deben ser rápidamente solventadas.

## Objetivos

- Implementar modelos de output estrcuturados para asuntos urgentes de clasificación por temas.
- Implementar las funciones de clasificación basadas en LLM que devuelven resultados formateados
- Construir y conectar pools de agentes y crear round-robin subgraphs (????)
- Ensamblar el workflow completo que enruta por urgencia + tema y asigna agentes
- Testear el sistema de routing a lo largo de escenarios típicos

## Pasos

### 1. Chequeo del entorno y configuración del LLM
### 2. Crear modelos estrcuturados de outputs (Pydantic)
- Implementar la `UrgencyClassification` con un solo campo:
```python
urgency: Literal["urgent", "normal"]
```
- Lo mismo con la `TopicClassification`:
```python
topic: Literal["credit_card", "account", "loan", "general"]
```
- Estos modelos refuerzan outputs predecibles desde el LLM y hacer el routing descendente determinista.

### 3. Implementar `classify_urgency`
- Usar el LLM con outputs estrcuturados
- Construir un prompt que le pida al modelo de vuelta ya sea "urgente" o "normal"
- Devolver el campo `urgency` del resultado de Pydantic parseado
- Mantener la temperatura en todo momento al 0.0

### 4. Implementar `classify_topic`
- Usa el prompting del output estructurado para devolver uno de los cuatro topics: `credit_card`, `account`, `loan` o `general`
- Aportar ejemplos breves en el prompt para guiar la conversación
- Devolver el campo `topic` del resultado parseado

### 5. Crear el mapa de enjambre de agentes

- Crear un diccionario que mapee:
  - `"general_team"`→`general_agents_pool`
  - `"credit_card_team"` → `credit_card_agents_pool`
  - `"account_team"` → `account_agents_pool`
  - `"loan_team"` → `loan_agents_pool`

- El workflow usará este mapa para buscar las pools de agentes por nombre

### 6. Crear seam subgraphs y ocuparlos  con agent_teams
- Para cada team pool (general, credit_card...), se llama a `create_team(name, agent_pool)` con un nombre único como `"general_team"`
- Reagrupar los graphs compilados en una lista `agent_teams`
- Confirmar con la lista `agent_teams` contenga los cuatro subgraphs compilados y que los nombres coincidan con las claves del mapeo

### 7. Implementar la lógica de routing del define_agent_team
- Dentro de `define_agent_team`, llamamos a `classify_urgency(query)` y a `classify_topic(query)`.
- Aplicamos el routing adecuado:
  - Si la urgencia es `"urgent"`:
    - `Route credit_card` → `"credit_card_team"`
    - `Route account` → `"account_team"`
    - `Route loan` → `"loan_team"`
  - De lo contrario, enruta a `"general_team"`

- Devuelve `{"team_to_call" : "<team_name>"}` as shown

### 8. Correr el workflow
  - compilar el workflow de fintech (`create_fintech_workflow()`) y guardarlo en `fintech_graph`
  - llamar a `run_multi_agent_system(query, fintech_graph, thread_id, agent_teams, agent_swarm_map)`
  sustituyendo:
    - `query`: una string de mensaje
    - `thread_id`: cualquier etiqueta (como `"round_robin"`)
    -  `agent_teams`: la lista de team graphs compilada
    - `agent_swarm_map`: mapping desde los team names hacia los agent pools

### 9. Validar con casos de prueba:
- Usar la lista `test_cases` aportada para probar los escenarios.
- Para cada caso:
  - Llamar a `classify_urgency` y `classify_topic` para confirmar las etiquetas predichas.
  - Correr el routing graph entero y confirmar que un agente del equipo esperado responde 
- Añadir afirmaciones o pintar sentencias para comparar lo esperado con lo real.

### 10. Ajustes opcionales y tuning:
 - Si las clasificaciones son inconsistentes, añadir prompts breves basados en ejemplos.
 - Si una pool del agente falla, verificar los nombres de los agentes y que los subgraphs compilados contengan los nombres de los nodos adecuados.
 - Si el round-robin parece sesgado, echarle un ojo a la persistencia de  `current_agent_index` y al comportamiento del memory saver.


# SOLUCIÓN

```python

import os 
from typing import Dict, Any, List, Literal
from IPython.display import Image, display
import nest_asyncio
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph.message import MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

nest_asyncio.apply()
load_dotenv()

llm= ChatOpenAI(model="gpt-4o-mini", temperature=0.0, base_url= os.getenv("OPENAI_BASE_URL"), api_key= os.getenv("OPENAI_API_KEY"))

# 1. FUNCIONES DE CLASIFICACIÓN

class UrgencyClassification(BaseModel):
    urgency: str= Field(
        ...,
        description=(
            "Urgency classifier: determine the priority level of the message. "
            "- urgent: critical, emergency, asap, immediate, stolen, fraud, security breach, "
            "  can't access, locked out, stolen card, unauthorized charges... "
            "- normal: routine, general, standard, regular, how to, what are, help with..."
        ),
        enum= ["urgent", "normal"]
    )

class TopicClassification(BaseModel):
    topic: str= Field(
        ...,
        description=(
            "Topic classifier: determine the financial product/service category. "
            "- credit_card: credit card, card, charges, fraud, stolen, rewards, disputes, "
            "  credit limit, card number, CVV, expiration... "
            "- account: account, login, password, security, verification, settings, "
            "  profile, personal info, contact info... "
            "- loan: loan, mortgage, refinance, payment, application, documentation, "
            "  interest rate, terms, approval... "
            "- general: product, service, company, balance, statement, general inquiry, "
            "  how to, what is, help..."
        ),
        enum= ["credit_card", "account", "loan", "general"]
    )

def classify_urgency(text: str)-> str:
    """Clasifica el nivel de urgencia de la petición de un cliente"""

    urgency_prompt= ChatPromptTemplate.from_template(
        "Extract the desired information from the following passage. "
        "Only extract the properties mentioned in the 'UrgencyClassification' function. "
        "Passage: \n"
        "{input}"
    )
    urgency_classifier= llm.with_structured_output(UrgencyClassification)
    prompt= urgency_prompt.invoke({"input": text})
    response= urgency_classifier.invoke(prompt)
    return response.model_dump()

def classify_topic(text: str)-> str:
    """Clasifica el topic de la consulta del cliente"""
    topic_prompt= ChatPromptTemplate.from_template(
        "Extract the desired information from the following passage. "
        "Only extract the properties mentioned in the 'TopicClassification' function. "
        "Passage: \n"
        "{input}"
    )
    topic_classifier= llm.with_structured_output(TopicClassification)
    prompt= topic_prompt.invoke({"input":text})
    response= topic_classifier.invoke(prompt)
    return response.model_dump()

# AGENT POOLS

# General Support Team (3 agentes)
general_agents_pool= [
    create_react_agent(
        name= f"general_agent_{i}",
        prompt=SystemMessage(
            content=(
                f"You are General Support Agent {i} at FinTechCorp. "
                "Handle basic product questions, account setup, and general inquiries. "
                "ALWAYS start with '[GENERAL SUPPORT]' and be helpful!"

            )
        ),
        model=llm,
        tools= [],

    ) 
    for i in range(1, 4)
]

# Credit Card Team (2 agentes)

credit_card_agents_pool= [
    create_react_agent(
        name= f"credit_card_agent_{i}",
        prompt= SystemMessage(
            content=(
                f"You are Credit Card Specialist {i} at FinTechCorp. "
                "Handle credit card issues, disputes, fraud, and rewards. "
                "ALWAYS start with '[CREDIT CARD]' and be professional!"
            )
        ),
        model=llm,
        tools=[],
    ) for i in range(1,3)
]

# Account Team (2 agents)
account_agents_pool = [
    create_react_agent(
        name=f"account_agent_{i}",
        prompt=SystemMessage(
            content=(
                f"You are Account Specialist {i} at FinTechCorp. "
                "Handle account access, security, verification, and settings. "
                "ALWAYS start with '[ACCOUNT]' and be security-focused!"
            )
        ),
        model=llm,
        tools=[],
    ) for i in range(1, 3)
]

# Loan Team (2 agents)
loan_agents_pool = [
    create_react_agent(
        name=f"loan_agent_{i}",
        prompt=SystemMessage(
            content=(
                f"You are Loan Specialist {i} at FinTechCorp. "
                "Handle loan applications, payments, refinancing, and documentation. "
                "ALWAYS start with '[LOAN]' and be thorough!"
            )
        ),
        model=llm,
        tools=[],
    ) for i in range(1, 3)
]

agent_swarm_map= {
    "general_team": general_agents_pool,
    "credit_card_team": credit_card_agents_pool,
    "account_team": account_agents_pool,
    "loan_team": loan_agents_pool,
}

# ROUND-ROBIN: TEAM SUBGRAPHS

class RoundRobinState(MessagesState):
    """State that tracks which agent to call next"""
    agent_names: List[str]
    current_agent_index: int

def update_index(state: RoundRobinState):
    agent_names= state["agent_names"]
    current_agent_index= state.get("current_agent_index", 0)
    new_agent_index= current_agent_index + 1
    new_agent_index_in_range= new_agent_index % len(agent_names)
    return {"current_agent_index": new_agent_index_in_range}

def route_round_robin(state: RoundRobinState):
    """Route tasks in round-robin fashion"""
    agent_names= state["agent_names"]
    current_agent_index = state.get("current_agent_index", 0)
    current_agent_index_in_range = current_agent_index % len(agent_names)
    active_agent= agent_names[current_agent_index_in_range]
    print("Round-robin Active Agent:", active_agent)
    return active_agent

def create_team(name: str, agent_pool: List[CompiledStateGraph]):
    workflow= StateGraph(RoundRobinState)

    workflow.add_node("update_index", update_index)
    for agent in agent_pool:
        workflow.add_node(agent.name, agent)
    
    workflow.add_edge(START, "update_index")

    workflow.add_conditional_edges(
        source="update_index",
        path=route_round_robin,
        path_map= [agent.name for agent in agent_pool]
    )

    graph = workflow.compile(name=name, checkpointer=MemorySaver())

    return graph

agent_teams= [create_team(team_name, agent_pool) for team_name, agent_pool in agent_swarm_map.items()]

agent_teams_3_png= agent_teams[3].get_graph().draw_mermaid_png()

# ENSAMBLAJE WORKFLOW

class FintechState(MessagesState):
    """Estado que trackea qué team llamar a continuación"""
    query: str
    agent_teams: List[CompiledStateGraph]
    agent_swarm_map: Dict[str, CompiledStateGraph]
    team_to_call: str
    urgency: str = ""  
    topic: str = "" 

def define_agent_team(state: FintechState):
    query= state["query"]

    urgency_result= classify_urgency(query)
    urgency= urgency_result["urgency"]

    topic_result= classify_topic(query)
    topic= topic_result["topic"]

    if urgency == "urgent":
        if topic == "credit_card":
            return {"team_to_call": "credit_card_team", "urgency": urgency, "topic": topic}
        elif topic == "account":
            return {"team_to_call": "account_team", "urgency": urgency, "topic": topic}
        elif topic == "loan":
            return {"team_to_call": "loan_team", "urgency": urgency, "topic": topic}
        
    return {"team_to_call": "general_team", "urgency": urgency, "topic": topic}

def trigger_agent_team(state: FintechState, config: RunnableConfig):
    query= state["query"]
    team_to_call= state["team_to_call"]
    agent_teams: List[CompiledStateGraph]= config["configurable"]["agent_teams"]
    agent_swarm_map= config["configurable"]["agent_swarm_map"]
    agent_pool_to_call: List[CompiledStateGraph]= agent_swarm_map[team_to_call]

    for team in agent_teams:
        print(team.name, team_to_call)
        if team.name == team_to_call:

            result= team.invoke(
                input={
                    "messages": [HumanMessage(content= query)],
                    "agent_names": [agent.name for agent in agent_pool_to_call]
                },
                config= {
                    "configurable": {
                        "thread_id":"round_robin"
                    }
                }
            )
            return {"messages": result["messages"]}
    
    raise ValueError("team_to_call is not inside agent_teams")

def create_fintech_workflow():
    """Crear el workflow completo de forma que:
    1. Enrute por urgencia + topic a los teams
    2. Use round-robin en los teams
    """
    workflow= StateGraph(FintechState)

    workflow.add_node("define_agent_team", define_agent_team)
    workflow.add_node("trigger_agent_team", trigger_agent_team)

    workflow.add_edge(START, "define_agent_team")
    workflow.add_edge("define_agent_team", "trigger_agent_team")
    workflow.add_edge("trigger_agent_team", END)

    return workflow

fintech_workflow= create_fintech_workflow()
fintech_graph= fintech_workflow.compile(checkpointer=MemorySaver())


fintech_workflow_png= fintech_graph.get_graph().draw_mermaid_png()


# CORRER SISTEMA MULTI AGENTE

def run_multi_agent_system(
        query: str,
        graph: CompiledStateGraph,
        thread_id: str,
        agent_teams: List[CompiledStateGraph],
        agent_swarm_map: Dict[str, CompiledStateGraph]
):
    result= graph.invoke(
        input= {"query": query}, 
        config={
            "configurable":{
                "thread_id": thread_id,
                "agent_teams": agent_teams,
                "agent_swarm_map": agent_swarm_map,
            }
        }
    )
    return result

result= run_multi_agent_system(
    query= "Me han robado los datos de contacto y podría haber alguien haciéndose pasar por mí",
    graph= fintech_graph,
    thread_id= "1",
    agent_teams= agent_teams,
    agent_swarm_map= agent_swarm_map
)



test_cases = [
    # (message, expected_team, expected_urgency)
    ("How do I check my account balance?", "general_team", "normal"),
    ("URGENT: My credit card was stolen!", "credit_card_team", "urgent"),
    ("I can't access my account, this is critical", "account_team", "urgent"),
    ("What are the current loan rates?", "loan_team", "normal"),
    ("ASAP: Fraudulent charges on my card!", "credit_card_team", "urgent"),
    ("How do I update my contact information?", "account_team", "normal"),
    ("I need help with my loan application", "loan_team", "normal"),
    ("What products do you offer?", "general_team", "normal"),
]

for i in range(len(test_cases)):
    result= run_multi_agent_system(
        query= test_cases[i][0],
        graph= fintech_graph,
        thread_id= f"{i}",
        agent_teams= agent_teams,
        agent_swarm_map= agent_swarm_map
    )

    for message in result["messages"]:
        print(f"\n{'='*60}")
        print(f"CLASIFICACIÓN: Urgencia={result['urgency']}, Topic={result['topic']}")
        print(f"{'='*60}\n")
        message.pretty_print()
```

