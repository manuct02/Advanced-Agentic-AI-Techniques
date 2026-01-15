from typing import Literal
import random
from IPython.display import Image, display
import nest_asyncio
from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.types import Command

nest_asyncio.apply()

# 1 Peer-to-Peer


  # 1.1. Pipeline

def l1_agent(state: MessagesState):
    print(f"started_with_l1_agent")
    next_node= "l2_agent"
    print(f"handed off to {next_node}")

def l2_agent(state: MessagesState):
    next_node = "l3_agent"
    print(f"handed off to {next_node}")

def l3_agent(state: MessagesState):
    next_node = END
    print(f"handed off to {next_node}") 


workflow = StateGraph(MessagesState)
workflow.add_node(l1_agent)
workflow.add_node(l2_agent)
workflow.add_node(l3_agent)

workflow.add_edge(START, "l1_agent")
workflow.add_edge("l1_agent", "l2_agent")
workflow.add_edge("l2_agent", "l3_agent")
workflow.add_edge("l3_agent", END)

graph= workflow.compile()

workflow_png= graph.get_graph().draw_mermaid_png()



  # 1.2. Network

def l1_agent(state: MessagesState)-> Command[Literal["l2_agent", "l3_agent", END]]:
    print(f"started with l1_agent")
    next_node= random.choice(["l2_agent", "l3_agent", END])
    print(f"handed off to {next_node}")

    return Command(goto= next_node)

def l2_agent(state: MessagesState) -> Command[Literal["l1_agent", "l3_agent", END]]:
    next_node = random.choice(["l1_agent", "l3_agent", END])
    print(f"handed off to {next_node}")

    return Command(
        goto=next_node,
    )

def l3_agent(state: MessagesState) -> Command[Literal["l1_agent", "l2_agent", END]]:
    next_node = random.choice(["l1_agent", "l2_agent", END])
    print(f"handed off to {next_node}")
    return Command(
        goto=next_node,
    )

workflow = StateGraph(MessagesState)
workflow.add_node(l1_agent)
workflow.add_node(l2_agent)
workflow.add_node(l3_agent)

workflow.add_edge(START, "l1_agent")

complex_graph = workflow.compile()

complex_workflow_png= complex_graph.get_graph().draw_mermaid_png()

# 2. Orquestador

  # 2.1. Supervisor

def supervisor(state: MessagesState)-> Command[Literal["l1s_agent", "l2s_agent", END]]:
    next_node= random.choice(["l1s_agent", "l2s_agent", END])
    print(f"supervisor handed off to {next_node}")
    return Command(goto=next_node)

def l1s_agent(state: MessagesState) -> Command[Literal["supervisor"]]:
    next_node = "supervisor"
    print(f"l1s_agent handed off to {next_node}")
    return Command(
        goto=next_node
    )

def l2s_agent(state: MessagesState) -> Command[Literal["supervisor"]]:
    next_node = "supervisor"
    print(f"l2s_agent handed off to {next_node}")
    return Command(
        goto=next_node
    )

supervisor_workflow= StateGraph(MessagesState)
supervisor_workflow.add_node(supervisor)
supervisor_workflow.add_node(l1s_agent)
supervisor_workflow.add_node(l2s_agent)

supervisor_workflow.add_edge(START, "supervisor")

supervisor_graph= supervisor_workflow.compile()
supervisor_workflow_png= supervisor_graph.get_graph().draw_mermaid_png()

  # 2.2. Hierarchial

def l1h_agent(state: MessagesState)-> Command[Literal["l2h_agent", "l3h_agent", END]]:
    next_node= random.choice(["l2h_agent", "l3h_agent", END])
    print(f"l1h_agent handed off to {next_node}")
    return Command(goto=next_node)

def l2h_agent(state: MessagesState) -> Command[Literal["l4h_agent", "l5h_agent", "l1h_agent"]]:
    next_node = random.choice(["l1h_agent", "l4h_agent", "l5h_agent"])
    print(f"l2h_agent handed off to {next_node}")
    return Command(
        goto=next_node
    )

def l3h_agent(state: MessagesState) -> Command[Literal["l1h_agent"]]:
    next_node = "l1h_agent"
    print(f"l3h_agent handed off to {next_node}")
    return Command(
        goto=next_node
    )

def l4h_agent(state: MessagesState) -> Command[Literal["l2h_agent"]]:
    next_node = "l2h_agent"
    print(f"l4h_agent handed off to {next_node}")
    return Command(
        goto=next_node
    )

def l5h_agent(state: MessagesState) -> Command[Literal["l2h_agent"]]:
    next_node = "l2h_agent"
    print(f"l5h_agent handed off to {next_node}")
    return Command(
        goto=next_node
    )

hie_workflow= StateGraph(MessagesState)
hie_workflow.add_node(l1h_agent)
hie_workflow.add_node(l2h_agent)
hie_workflow.add_node(l3h_agent)
hie_workflow.add_node(l4h_agent)
hie_workflow.add_node(l5h_agent)

hie_workflow.add_edge(START, "l1h_agent")

hie_graph= hie_workflow.compile()
hie_workflow_png= hie_graph.get_graph().draw_mermaid_png()

hie_graph.invoke(input={})