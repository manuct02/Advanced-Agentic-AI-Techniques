from typing import Literal
import random
import nest_asyncio
from IPython.display import Image, display
from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.types import Command

nest_asyncio.apply()

class State(MessagesState): 
    doctor_available: bool
    patient_verified: bool
    appointment_scheduled: bool
    notification_sent: bool

def appointment_coordinator(state: State)-> Command[Literal["doctor_availability_agent", "patient_verification_agent", "scheduling_agent", "notification_agent", END]]:
    """Supervisor que orquesta el flujo de la acción del agente de citas"""

    # Chequea el estado actual para determinar el siguiente paso
    if not state["doctor_available"]:
        next_step= "doctor_availability_agent"
        print(f"Routing to {next_step}")
    elif not state["patient_verified"]:
        next_step= "patient_verification_agent"
        print(f"Routing to {next_step}")
    elif not state["appointment_scheduled"]:
        next_step= "scheduling_agent"
        print(f"Routing to {next_step}")
    elif not state["notification_sent"]:
        next_step= "notification_agent"
        print(f"Routing to {next_step}")
    else:
        next_step= END
        print(f"✅ Workflow complete, ending...")
    
    return Command(goto=next_step)

def doctor_availability_agent(state: State)->Command[Literal["appointment_coordinator"]]:
    """Checks doctor availability and schedules"""
    print("👨‍⚕️ Doctor Availability Agent: Checking schedules...")

    # Simula chequear la disponibilidad del doctor
    available= random.choice([True, False])
    if available:
        print("✅ Doctor is available for requested time")
    else:
        print("❌ Doctor not available, suggesting alternatives")
    
    return Command(goto= "appointment_coordinator", update={"doctor_available": available})

def patient_verification_agent(state: State) -> Command[Literal["appointment_coordinator"]]:
    """Verifies patient identity and eligibility"""
    print("🆔 Patient Verification Agent: Verifying patient...")
    
    # Simulate patient verification
    verified = random.choice([True, False])
    if verified:
        print("✅ Patient verified and eligible")
    else:
        print("❌ Patient verification failed")
    
    return Command(
        goto="appointment_coordinator",
        update={"patient_verified": verified}
    )

def scheduling_agent(state: State) -> Command[Literal["appointment_coordinator"]]:
    """Books the actual appointment"""
    print("📅 Scheduling Agent: Booking appointment...")
    
    # Simulate appointment booking
    booked = random.choice([True, False])
    if booked:
        print("✅ Appointment successfully booked")
    else:
        print("❌ Appointment booking failed")
    
    return Command(
        goto="appointment_coordinator",
        update={"appointment_scheduled": booked}
    )

def notification_agent(state: State) -> Command[Literal["appointment_coordinator"]]:
    """Sends notifications and reminders"""
    print("📢 Notification Agent: Sending notifications...")
    
    # Simulate sending notifications
    notification_sent = random.choice([True, False])
    if notification_sent:
        print("✅ Notifications sent successfully")
    else:
        print("❌ Failed to send notifications")
    
    return Command(
        goto="appointment_coordinator",
        update={"notification_sent": notification_sent}
    )

workflow= StateGraph(State)

workflow.add_node(appointment_coordinator)
workflow.add_node(doctor_availability_agent)
workflow.add_node(patient_verification_agent)
workflow.add_node(scheduling_agent)
workflow.add_node(notification_agent)

workflow.add_edge(START, "appointment_coordinator")

graph= workflow.compile()

multi_agent_workflow_png= graph.get_graph().draw_mermaid_png()

# Testear el workflow

print("Testing Healthcare Appointment System")
result= graph.invoke(input={
    "doctor_available":False,
    "patient_verified": False,
    "appointment_scheduled": False,
    "notification_sent": False
})

print("✅ Workflow execution completed!")