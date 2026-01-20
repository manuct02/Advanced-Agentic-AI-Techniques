import os
from typing import Dict, Any, List
from IPython.display import Image, display
from dotenv import load_dotenv
from urllib.parse import urlparse
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from tavily import TavilyClient

load_dotenv()

llm= ChatOpenAI(model="gpt-4o-mini", temperature= 0.0, base_url= os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY"))

# AGENTES

@tool
def get_knowledge_base()-> Dict[str, Any]:
    """Get solutions from the knowledge base for common issues."""
    solutions = {
        "crash": "Try clearing app cache and restarting. If issue persists, check for app updates.",
        "upload": "Ensure file size is under 100MB and format is supported. Check internet connection.",
        "login": "Verify credentials and try password reset if needed.",
        "performance": "Close other apps and restart device. Check available storage space.",
        "billing": "Check your subscription status and payment method. Contact billing support for detailed assistance.",
        "account": "Verify your account settings and try logging out and back in.",
        "general": "I can help you with general account management and basic troubleshooting."
    }
    return solutions

@tool
def check_system_status()-> Dict[str, Any]:
    """Check current system status and known issues."""
    return {
        "status": "operational",
        "known_issues": [],
        "maintenance_scheduled": False
    }

@tool
def get_billing_info(customer_id: str) -> Dict[str, Any]:
    """Retrieve customer billing information and recent transactions."""
    # Simulated billing data
    return {
        "customer_id": customer_id,
        "subscription_status": "active",
        "last_payment": "2025-12-15",
        "next_billing": "2026-01-15",
        "amount": "$29.99"
    }

@tool
def get_account_info(customer_id: str) -> Dict[str, Any]:
    """Retrieve customer account information."""
    return {
        "customer_id": customer_id,
        "account_type": "premium",
        "created_date": "2021-07-27",
        "last_login": "2026-01-13"
    }

technical_agent= create_react_agent(
    name= "technical_support",
    prompt=SystemMessage(
        content=("You are a technical support specialist. "
            "Help customers with technical issues like app crashes, upload problems, performance issues, and login troubles. "
            "Use the knowledge base to provide accurate solutions. "
            "Be patient and provide step-by-step instructions. "
            "If the issue is complex, suggest escalation.")),
    model= llm,
    tools=[get_knowledge_base, check_system_status]


)

billing_agent= create_react_agent(
    name= "billing_support",
    prompt= SystemMessage(
        content=(
            "You are a billing support specialist. "
            "Help customers with payment issues, subscription questions, billing disputes, and account upgrades/downgrades. "
            "Always verify customer identity and provide clear explanations of charges. "
            "Be empathetic about billing concerns."
        )

    ),
    model= llm,
    tools=[get_billing_info, get_knowledge_base]
)

general_agent = create_react_agent(
    name="general_support",
    prompt=SystemMessage(
        content=(
            "You are a general support specialist. "
            "Help customers with account management, password changes, general questions, and basic troubleshooting. "
            "Provide friendly and helpful assistance. "
            "Route complex technical or billing issues to appropriate specialists."
        )
    ),
    model=llm,
    tools=[get_account_info, get_knowledge_base],
)

escalation_agent = create_react_agent(
    name="escalation_support",
    prompt=SystemMessage(
        content=(
            "You are an escalation specialist for complex cases. "
            "Handle cases that require human intervention, complex technical issues, or sensitive billing disputes. "
            "Acknowledge the customer's frustration and provide a clear path forward. "
            "Document the case thoroughly for human review."
        )
    ),
    model=llm,
    tools=[get_knowledge_base, get_account_info, get_billing_info],
)

# TOOLS DEL SUPERVISOR

@tool
def route_to_technical(issue_description: str, customer_id: str)-> Dict[str, Any]:
    """Route technical issues to the technical support specialist."""
    message= HumanMessage(
        content=f"Customer ID: {customer_id}\n Technical Issue: {issue_description}"
    )
    result= technical_agent.invoke(
        input={"messages": message}
    )
    last_message: AIMessage= result["messages"][-1]
    return {
        "task": "technical_support",
        "customer_id": customer_id,
        "issue": issue_description,
        "response": last_message.content,
        "status": "resolved"
    }

@tool
def route_to_billing(billing_question: str, customer_id: str) -> Dict[str, Any]:
    """Route billing questions to the billing specialist."""
    message = HumanMessage(
        content=f"Customer ID: {customer_id}\nBilling Question: {billing_question}"
    )
    
    result = billing_agent.invoke(
        input={"messages": [message]}
    )
    
    last_message: AIMessage = result["messages"][-1]
    
    return {
        "task": "billing_support",
        "customer_id": customer_id,
        "question": billing_question,
        "response": last_message.content,
        "status": "resolved"
    }

@tool
def route_to_general(general_inquiry: str, customer_id: str) -> Dict[str, Any]:
    """Route general inquiries to the general support specialist."""
    message = HumanMessage(
        content=f"Customer ID: {customer_id}\nGeneral Inquiry: {general_inquiry}"
    )
    
    result = general_agent.invoke(
        input={"messages": [message]}
    )
    
    last_message: AIMessage = result["messages"][-1]
    
    return {
        "task": "general_support",
        "customer_id": customer_id,
        "inquiry": general_inquiry,
        "response": last_message.content,
        "status": "resolved"
    }

@tool
def route_to_escalation(complex_case: str, customer_id: str) -> Dict[str, Any]:
    """Route complex cases to the escalation specialist."""
    message = HumanMessage(
        content=f"Customer ID: {customer_id}\nComplex Case: {complex_case}"
    )
    
    result = escalation_agent.invoke(
        input={"messages": [message]}
    )
    
    last_message: AIMessage = result["messages"][-1]
    
    return {
        "task": "escalation",
        "customer_id": customer_id,
        "case": complex_case,
        "response": last_message.content,
        "status": "escalated"
    }

# AGENTE SUPERVISOR

supervisor_agent= create_react_agent(
    name= "support_suppervisor",
    prompt= SystemMessage(
        content=(
            "You are a customer support supervisor. Your job is to:\n"
            "1. Analyze customer requests and determine the appropriate specialist\n"
            "2. Route them to the correct support agent using the routing tools\n"
            "3. Ensure proper handoffs between agents\n"
            "4. Monitor resolution progress\n\n"
            "Routing guidelines:\n"
            "- Technical issues (crashes, uploads, performance, login): use route_to_technical\n"
            "- Billing questions (payments, subscriptions, charges): use route_to_billing\n"
            "- General inquiries (account management, passwords): use route_to_general\n"
            "- Complex cases requiring human intervention: use route_to_escalation\n\n"
            "Always include the customer_id when routing. Use the tools to delegate work appropriately."

        )
    ),
    model= llm,
    tools= [route_to_technical, route_to_billing, route_to_general, route_to_escalation],
    checkpointer=MemorySaver()

)

# Prueba
user_messages = [
    "Why was I charged twice this month?",
    "Ok. One more thing: my app keeps crashing when I try to upload files.",
    "I've already tried everything but nothing works"
]

for user_message in user_messages:
    result = supervisor_agent.invoke(
        input={
            "messages": [
                SystemMessage(content="Customer ID: CUST123"),
                HumanMessage(content=user_message),
            ]
        },
        config={
            "configurable": {
                "thread_id": "S-12345",
            }
        }
    )

for message in result["messages"]:
    message.pretty_print()