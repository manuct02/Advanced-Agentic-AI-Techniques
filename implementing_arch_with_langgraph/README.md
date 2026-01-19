# Ejercicio: Blog Post Pipeline

Aquí vamos a implementar una pipeline de posteo en un blog de un multi-agente usando LangGraph para reemplazar el paso de mensajes manual. La pipeline orquestará agentes especializados:
- Intake Researcher.
- Writer
- Reviewer
- Publisher

De tal forma que el flujo de estados sea automático y mantenga la arquitectura limpia y manejable.

### Prerrequisitos:
- Entorno de Python con las librerías pertinentes (langchain_core, langchain_openai, langgraph, tavily (?), dotenv, IPython)
- Una key de OpenAI almacenada en la variable "OPENAI_API_KEY".

### Objetivos:
- Implementar una subclase `BlogState` para llevar los campos estructurados entre agentes.
- Crear agentes especializados con tools y system prompts para el intake, investigación, escritura, revisión y publicación.
- Construir funciones nodo que llamen a los agentes, actualicen el `BlogState` y conecten nodos dentro de un workflow de LangGraph.
- Compilar, correr y verificar el pipeline de LangGraph, confirmando así el manejo automatizado del estado y el output final publicado.

## **Pasos**

### 1. Setear el entorno y las credenciales
### 2. Definir el estado del blog workflow
- Implementar `class BlogState(MessagesState):` con campos como:
  - `title: str | None`
  - `outline: List[str]`
  - `research_notes: List[Dict[str, str]]`
  - `draft: str | None`
  - `edits: str | None`
  - `published_url: str | None`
- Asegurar que los valores predeterminados se inicialicen como listas vacías o `None` de forma que el estado sea serializable por el MemorySaver checkpoint.

### 3. Diseñar tools y agent system prompts
- Reemplazar los placeholders de `my_tool` con tools útiles.
- Para cada agente, setear un system prompt adecuado su rol específico:
  - Intake Agent: planear la estructura del blog
  - Researcher Agent: extraer y reumir fuentes, devolver citas
  - Writer Agent: produce borradores siguiendo el outline y la snotas del research
  - Reviewer Agent: aplica arreglos, optimización SEO y mejoras de legibilidad.
  - Publisher Agent: prepara los metadata finales y publica (publica de coña devolviendo una URL)
- Crear agentes usando `create_react_agent(...)`, vinculamos el LLM, herramientas y mensajes del sistema.

### 4. Implementar las funciones nodo que manipulan el estado
- Para cada nodo (intake_node, research_node, writer_node, reviewer_node, publisher_node):
  - Acepta el `state: BlogState` y devuelve `BlogState`
  - Construye el input del agente convirtiendo los campos relevantes  del estado en mensajes del agente (SystemMessage + HumanMessage)
  - Llamar al agente y parsear la respuesta del agente; actualizar los campos apropiados de BlogState (e.g., set `outline`, append to `research_notes`, set `draft`, set `edits`, set `published_url`)
  - Mantener las respuestas estructuradas (JSON o texto claramente delimitado)
- Añadir checks preventivos: si hay campos eserados missing, levantar una excepción informativa o setear predeterminados.

### 5. Establecer el workflow de StateGraph

- Instanciar `workflow= StateGraph(BlogState)`
- Añadir nodos al workflow usando `workflow.add_node("intake", intake_node)`, etc.
- Conectar los edges:
  - `workflow.add_edge(START, "intake")`
  - `workflow.add_edge("intake", "research")`
  - `workflow.add_edge("research", "writer")`
  - `workflow.add_edge("writer", "reviewer")`
  - `workflow.add_edge("reviewer", "publisher")`
  - `workflow.add_edge("publisher", END)`
### 6. Compilar, visualizar y correr el workflow
- Compilar con checkpointing:
```python
graph= workflow.compile(checkpointer= MemorySaver())
```
- Sacar el diagrama de mermaid
- Preparar el contenido inicial de  `HumanMessage` que represente brevemente el blog (la idea del título, la audiencia target, las palabras clave)
- Invoke el graph, hacer una llamada a `graph.invoke(input={"messages": user_message}) y almacenar el resultado

### 7. Verificar la ejecución e inspeccionar el estado

- Inspeccionar la variable `result` y los checkpoints guardados de MemorySaver para la ejecución de cada nodo
- Confirmar que:
  - `outline` se haya creado con el intake
  - `research_notes` contenga fuentes y resuma desde researches
  - `draft` incluya secciones estructuradas siguiendo el outline
  - `edits` refleje las mejoras del reviewer y sugerencias de SEO
  - `published_url` apunte a la localización final.

Si faltase cualquier campo, trazar la lógica del nodo y los prompts de los agentes así como el re-run con las mejoras del parsing o instrucciones.

### 8. Iteración de prompts y herramientas
- Refinar system prompts para una mejor estrcutura (lo mejor es que los agentes devuelvan un JSON)
- Añadir o ajustar tools (por ejemplo, añadir una tool de web scrapper para citaciones reales)
- Recompilar y re-run para observar las transiciones de estado mejoradas

## **SOLUCIÓN**