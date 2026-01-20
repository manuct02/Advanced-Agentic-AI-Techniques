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

with open("agent_teams_3.png", "wb") as f:
    f.write(agent_teams_3_png)