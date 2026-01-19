# EJERCICIO: Custommer Support
En este ejercicio crearemos un ayudante de estilo supervisor para workflows usando LangGraph y tool-calling agents. El agente supervisor analizará los mensajes provenientes del cliente, los mandará a los agentes especializados, y realizará mejoras cuando se precise. Aquí ponemos en práctica el patrón *handsoff* así como modelos de arquitectura usado por SaaS (las aplicaciones de software en la nube).

## Objetivos:

- Implementar agentes especializados con tools y system prompts
- Crear tools de supervisión que llamen a agentes especialistas vía tool-calling
- Manejar estados del workflow compartidos entre agentes
- Implementar handsoff limpios y flujos de mejora
- Testar el supervisor con escenarios de support reales

## PASOS

### 1. Preparar el workflow y las credenciales
### 2. Definir las tools especialistas de cada agente y los prompts
### 3. Implementar herramientas de supervisión del routing
- Completar cada función @tool de routing:
  - `route_to_technical(issue_description: str, customer_id: str)` → `calls technical_agent` and returns structured result
  - `route_to_billing(billing_question: str, customer_id: str)` → calls `billing_agent` and returns structured result
  - `route_to_general(query: str, customer_id: str)` → calls `general_agent` and returns structured result
  - `route_to_escalation(details: str, customer_id: str)` → calls `escalation_agent` and flags the ticket for human review

- Asegurar que cada routing tool:
  - Invoque al objetivo adecuado mediante `.invoke` o lo que requiera la API
  - Actualice un diccionario compartido que contnega el `ticket_id`, `customer_id`, `assigned_agent`, `resolution_status`.
  - Devuelva una estructura JSON con la respuesta del agente y los nuevos campos del estado

### 4. Crear el agente supervisor
- Usamos `create_react_agent` con un `SystemMessage` tal que:
  - Analice los mensajes entrantes
  - Elija la tool de routing correcta
  - Performe el handsoff y monitoree el progreso
- Añadir todas las routing tools a la lista de tools del agente supervisor

### 5. Ensamblar el input del workflow e invocar al supervisor

- Construir inputs `HumanMessage` usando escenarios de muestra. Estructura de ejemplo:
  - `HumanMessage(content="My app keeps crashing when I try to upload files")`
  - Incluir un campo `customer_id` en el input donde las tools lo requieran

- Llamar a 
```python 
supervisor_agent.invoke(input={"messages": user_message, "customer_id":" "})
```
- Repetir para cada escenario de prueba en el starter file

### 6. Verificar el estado y el handsoff
- Inspeccionar el objeto result devuelto para:
  - ver qué herramienta fue usada
  - la parte útil de la respuesta
  - los campos del estado actualizados (assigned_agent, resolution_status, escalation_flag)
- Si se usa StateGraph o MessgeState, pintar el camino de los nodos o los logs para confrimar el orden
- Confirmar que las mejoras son apropiadas

# SOLUCIÓN
