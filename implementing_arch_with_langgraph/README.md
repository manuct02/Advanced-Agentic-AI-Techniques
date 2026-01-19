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

Bueno, pues a pesar de tener un código que funciona perfecto no vamos a poder catarlo porque se ha cascado la API de OpenAI.

```python
import os
from typing import Dict, Any, List, Annotated
from IPython.display import Image, display
from urllib.parse import urlparse
import nest_asyncio
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from tavily import TavilyClient

nest_asyncio.apply()
load_dotenv()

llm= ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key= os.getenv("OPENAI_API_KEY"), base_url= os.getenv("OPENAI_BASE_URL"))
tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)

# Definir la estructura del estado para el workflow del blog
class BlogState(MessagesState):
    query: HumanMessage
    topic: str
    title: str
    audience: str
    sections: List[str]
    keywords: List[str]
    research: List[Dict[str, Any]]
    draft_markdown: str
    references_markdown: str
    final_markdown: str
    seo_score: int
    publish_status: str

# Los Agentes especializados

@tool

def outline_topic(topic: str)-> Dict[str, Any]:
    """
    Creates a pragmatic outline for a blog post given a topic.

    Args:
        topic (str): The domain/area/technology (e.g., "LangGraph multi-agent systems").

    Returns:
        Dict[str, Any]: A dictionary with:
            - title (str): Suggested working title.
            - audience (str): Target reader profile.
            - sections (List[str]): Ordered list of section headings.

    Example:
        >>> outline_topic(topic="LangGraph multi-agent systems")
        {'title': 'LangGraph multi-agent systems: What It Is, Why It Matters, and How To Start',
         'audience': 'Practitioners and curious newcomers',
         'sections': ['Introduction', 'Why ... matters right now', ...]}
    """
    
    base = topic.strip().rstrip(".")
    title = f"{base}: What It Is, Why It Matters, and How To Start"
    audience = "Practitioners and curious newcomers"
    sections = [
        "Introduction",
        f"Why {base} matters right now",
        f"Core concepts in {base}",
        f"Practical getting-started guide for {base}",
        "Common pitfalls and best practices",
        f"Trends and the future of {base}",
        "Conclusion and next steps",
    ]
    return {"title": title, "audience": audience, "sections": sections}

@tool
def generate_keywords(topic: str, limit: int= 12)-> List[str]:
    """
    Generates a compact keyword set for SEO checks based on a topic.

    Args:
        topic (str): The domain/area/technology to derive keywords from.
        limit (int, optional): Maximum number of keywords to return. Defaults to 12.

    Returns:
        List[str]: A list of keyword phrases ordered by relevance heuristics.

    Example:
        >>> generate_keywords(topic="LangGraph multi-agent systems", limit=8)
        ['langgraph multi-agent systems', 'langgraph multi-agent systems tutorial', ...]
    """

    root = topic.lower().strip()
    extras = [
        f"{root} tutorial", f"{root} guide", f"{root} best practices",
        f"{root} examples", f"{root} use cases", f"{root} vs alternatives",
        f"{root} performance", f"{root} scalability",
        f"{root} architecture", f"{root} tools", f"{root} pitfalls"
    ]
    # Deduplicate while preserving order
    seen = set()
    out = []
    for k in [root] + extras:
        if k not in seen:
            seen.add(k)
            out.append(k)
        if len(out) >= limit: break
    return out

@tool
def web_search(question: str)-> Dict[str, Any]:
    """
    Performs a web search and returns top search results for a given query.

    Args:
        question (str): A query to be searched.

    Returns:
        Dict[str, Any]: A dictionary with a 'results' list where each item is a dict containing:
            - url (str): The result URL.
            - title (str): The page title.
            - content (str): A content snippet or summary.
            - score (float): Relevance score from the search provider.

    Example:
        >>> web_search(question="LangGraph multi-agent patterns")
        {'results': [{'url': 'https://...', 'title': '...', 'content': '...', 'score': 0.82}, ...]}
    """
    response= tavily_client.search(query=question, max_results=5)
    return response

@tool
def dedupe_by_domain(results: List[Dict[str, Any]])-> List[Dict[str, Any]]:
    """
    Deduplicates research results by domain, keeping the strongest items per domain.

    Args:
        results (List[Dict[str, Any]]): A list of search result objects where each item contains:
            - url (str): Result URL (used to extract domain).
            - title (str): Page title.
            - content (str): Snippet or summary.
            - score (float): Relevance score.

    Returns:
        List[Dict[str, Any]]: A list of results with at most 'max_per_domain' per domain,
        sorted within each domain by descending 'score' (then title length as tie-breaker).

    Notes:
        If a URL is missing or unparsable, the domain is treated as "unknown".

    Example:
        >>> dedupe_by_domain(search_payload['results'])
        [{'url': 'https://a.com/...', ...}, {'url': 'https://b.org/...', ...}]
    """
    by_domain: Dict[str, List[Dict[str, Any]]]= {}
    for r in results or []:
        domain= urlparse(r.get("url", "")).netloc or "Unknown"
        by_domain.setdefault(domain, []).append(r)
    
    cleaned = []
    max_per_domain = 1
    for _, items in by_domain.items():
        items = sorted(items, key=lambda x: (x.get("score", 0), len(x.get("title", ""))), reverse=True)
        cleaned.extend(items[:max_per_domain])
    return cleaned

@tool
def format_references(results: List[Dict[str, Any]])-> str:
    """
    Formats research results into a Markdown footnotes block.

    Args:
        results (List[Dict[str, Any]]): A list of research items with fields:
            - title (str)
            - url (str)

    Returns:
        str: Markdown string with one footnote per line using the format:
             "[^n]: {title} — {url}"

    Example:
        >>> format_references([{'title':'Doc','url':'https://...'}])
        '[^1]: Doc — https://...'
    """

    lines= []
    for i, r in enumerate(results or [], start= 1):
        title = r.get("title", "Untitled")
        url= r.get("url", "")
        lines.append(f"[^{i}]: {title}--{url}")
    return "\n".join(lines)

@tool
def estimate_read_time(markdown_text: str, wpm: int= 225)-> Dict[str, Any]:
    """
    Estimates reading time and word count for a Markdown draft.

    Args:
        markdown_text (str): The full Markdown content to analyze.
        wpm (int, optional): Reading speed in words per minute. Defaults to 225.

    Returns:
        Dict[str, Any]: A dictionary with:
            - word_count (int): Total number of words.
            - minutes (int): Estimated whole minutes to read (minimum of 1).

    Example:
        >>> estimate_read_time("# Title\\n\\nSome text.")
        {'word_count': 3, 'minutes': 1}
    """
    import re
    words= re.findall(r"\b\w+\b", markdown_text or "")
    wc= len(words)
    minutes= max(1, round(wc/ max(100, wpm)))
    return {"word_count": wc, "minutes": minutes}

@tool
def markdown_lint(md: str)-> Dict[str, Any]:
    """
    Runs a simple Markdown lint pass for structure and readability issues.

    Args:
        md (str): Markdown content to lint.

    Returns:
        Dict[str, Any]: A dictionary with:
            - ok (bool): True if no issues found, else False.
            - issues (List[str]): Human-readable problem descriptions.

    Checks:
        - Ensures the document starts with a single H1 ('# ').
        - Warns if footnote markers are present but definitions are missing.
        - Flags overly long paragraphs (very rough heuristic).

    Example:
        >>> markdown_lint("No title")
        {'ok': False, 'issues': ['Missing H1 title at the very top (use '# Title').']}
    """
    issues= []
    if not md.strip().startswith('# '):
        issues.append("Missing H1 title at the very top (use '# Title').")
    if "[^" in md and "]: " not in md:
        issues.append("Footnote marker used but no footnote definitions present.")

    for idx, para in enumerate([p for p in md.split("\n\n") if p.strip()], start=1):
        if len(para) > 1200:
            issues.append(f"Paragraph {idx} is very long; consider splitting.")

    return {"ok": len(issues) == 0, "issues": issues}

@tool
def seo_score(md: str, keywords: List[str])-> Dict[str, Any]:
    """
    Computes a naive SEO score (0-100) based on keyword distribution.

    Args:
        md (str): Markdown content to score.
        keywords (List[str]): Target keywords/phrases to check for.

    Returns:
        Dict[str, Any]: A dictionary with:
            - score (int): SEO score (capped at 100).
            - title_hits (int): Count of keywords present in the H1 title.
            - heading_hits (int): Count of keywords found across headings.
            - body_hits (int): Count of keywords found in the full text.
            - keywords_checked (List[str]): Echo of the keywords analyzed.

    Scoring:
        - Title matches weighted 20 points each.
        - Heading matches weighted 6 points each.
        - Body matches weighted 2 points each.

    Example:
        >>> seo_score(md="# LangGraph Guide", keywords=["langgraph"])
        {'score': 28, 'title_hits': 1, 'heading_hits': 0, 'body_hits': 1, 'keywords_checked': ['langgraph']}
    """
    import re
    text= md.lower
    title= ""
    headings= []
    for line in md.splitlines():
        if line.startswith('# '):
            title= line.lower()
        if line.startswith('#'):
            headings.append(line.lower())
    def contains(k: str, hay: str)-> bool:
        return re.search(rf"\b{re.escape(k)}\b", hay) is not None
    
    title_hits= sum(1 for k in keywords if contains(k, title))
    heading_hits= sum(1 for k in keywords if any(contains(k, h) for h in headings))
    body_hits = sum(1 for k in keywords if contains(k, text))
    score= min(100, title_hits * 20 + heading_hits * 6 + body_hits * 2)
    return{
        "score": score,
        "title_hits": title_hits,
        "heading_hits": heading_hits,
        "body_hits": body_hits,
        "keywords_checked": keywords[:],
    }

@tool
def publish_blog(body_md:str)-> Dict[str, Any]:
    """
    Publishes the final blog post (mock implementation).

    Args:
        body_md (str): Final Markdown body content to publish.

    Returns:
        Dict[str, Any]: A dictionary indicating publication status:
            - status (str): Always 'published' in this mock implementation.

    Side Effects:
        Prints the blog title and body to stdout, followed by a status line.

    Example:
        >>> publish_blog(title="My Post", body_md="# My Post\\n\\nHello")
        {'status': 'published'}
    """
    print("\n=== PUBLISHING BLOG ===")
    print(body_md)
    print("\n === STATUS: published")
    return {"status": "pusblished"}

# LOS AGENTES

intake_agent= create_react_agent(
    name= "intake_agent",
    prompt= SystemMessage(
        content=("You are a pragmatic content planner. "
            "Given a domain/area/technology topic, produce a blog plan: "
            "a clear working title, target audience, and a section outline. "
            "Use tools to create the outline and initial keyword set. "
            "Return ONLY JSON with keys: topic, title, audience, sections, keywords.")
    ),
    model= llm,
    tools= [outline_topic, generate_keywords],
)

researcher_agent= create_react_agent(
    name= "researcher_agent",
    prompt=SystemMessage(
        content=(
            "You're a senior marketing researcher. "
            "Given the topic and optionally other sections, "
            "find RECENT, high-signal web information about the topic. "
            "Call web_search at least once with a good query; "
            "then call dedupe_by_domain to remove duplicates. "
            "Return ONLY JSON with key 'research', an array of items "
            "each like {url,title,content,score} (deduped)."

        )
    ),
    model= llm,
    tools=[web_search, dedupe_by_domain]
)

writer_agent= create_react_agent(
    name= "writer_agent",
    prompt= SystemMessage(
        content=(
            "You are a seasoned technical blogger. "
            "Write a polished Markdown blog post using title, sections, "
            "and research (array of {url,title,content,score}). "
            "Requirements:\n"
            " - Begin with an H1 (= the title).\n"
            " - Add a short abstract (1-2 sentences), then follow the sections.\n"
            " - Use bullets, code fences, and examples when helpful.\n"
            " - Use footnote-style citations [^1], [^2] inline whenever claims come from research.\n"
            " - Append '## References' followed by footnotes generated via the references tool.\n"
            " - Add an estimated reading time line near the top.\n"
            "Workflow:\n"
            " 1) Draft the post body.\n"
            " 2) Call estimate_read_time on the draft and insert a line like '_Estimated reading time: X min_'.\n"
            " 3) Call format_references with the same research list to create footnotes.\n"
            "Return ONLY JSON with keys: draft_markdown, references_markdown."
        )
    ),
    model= llm,
    tools= [estimate_read_time, format_references]
)

reviewer_agent = create_react_agent(
    name="reviewer_agent",
    prompt=SystemMessage(
        content=(
            "You are a meticulous copy editor. "
            "Input: draft_markdown, keywords, references_markdown. "
            "Tasks:\n"
            " 1) Ensure '## References' exists at the end and merge references_markdown if missing.\n"
            " 2) Call markdown_lint on the draft; then fix issues inline (H1 title, footnotes, long paragraphs).\n"
            " 3) Call seo_score; make small, high-quality edits to improve the score "
            "    without keyword stuffing (preserve voice and clarity).\n"
            "Output: Return ONLY JSON with keys: final_markdown, seo, lint."
        )
    ),
    model=llm,
    tools=[markdown_lint, seo_score],
)

publisher_agent = create_react_agent(
    name="publisher_agent",
    prompt=SystemMessage(
        content=(
            "You are a release manager. "
            "Publish the finalized blog post using the provided tool. "
            "Input: title and final_markdown (produced by the reviewer). "
            "Return ONLY JSON with keys: publish_status."
        )
    ),
    model=llm,
    tools=[publish_blog],
)

# Crear los nodos del workflow que integran los agentes:

class Blogplan(BaseModel):
    """Output del intake agent"""
    topic: Annotated[str, Field(description="The main topic of the blog post")]
    title: Annotated[str, Field(description="A clear, working title for the blog post")]
    audience: Annotated[str, Field(description="Target audience description")]
    sections: Annotated[List[str], Field(description="List of section headings for the blog outline")]
    keywords: Annotated[List[str], Field(description="List of relevant keywords for SEO")]

class ResearchData(BaseModel):
    """Output del research agent"""
    research: Annotated[List[Dict[str, Any]], Field(description="List of research items with url, title, content, and score")]

class ResearchItem(BaseModel):
    """Individual research item with metadata"""
    url: Annotated[str, Field(description="URL of the research source")]
    title: Annotated[str, Field(description="Title of the research content")]
    content: Annotated[str, Field(description="Extracted content from the source")]
    score: Annotated[float, Field(description="Relevance score from 0.0 to 1.0")]

class ResearchData(BaseModel):
    """Output from the researcher agent"""
    research: Annotated[List[ResearchItem], Field(description="List of research items")]

class ContentDraft(BaseModel):
    """Output del writer agent"""
    draft_markdown: Annotated[str, Field(description="The complete blog post in markdown format")]
    references_markdown: Annotated[str, Field(description="Formatted references section")]

class ReviewedContent(BaseModel):
    """Output from the reviewer agent"""
    final_markdown: Annotated[str, Field(description="The final, reviewed blog post")]
    seo_score: Annotated[int, Field(description="SEO score from 0-100")]
    lint_issues: Annotated[List[str], Field(description="List of any linting issues found and fixed")]

class PublicationStatus(BaseModel):
    """Output from the publisher agent"""
    publish_status: Annotated[str, Field(description="Status of the publication process")]

# LOS NODOS

def intake_node(state: BlogState)-> BlogState:
    print("Starting intake agent...")

    result= intake_agent.invoke(input={"messages":[state["query"]]})
    messages= result["messages"]
    last_ai_message: AIMessage= messages[-1]

    llm_with_structure= llm.with_structured_output(Blogplan)
    structured_output: Blogplan=llm_with_structure.invoke(last_ai_message.content)

    print(f"message: {last_ai_message.content[:20]}...")
    print("Done!\n")

    return{
        "messages": [state["query"]] + messages,
        "topic": structured_output.topic,
        "title": structured_output.title,
        "audience": structured_output.audience,
        "sections": structured_output.sections,
        "keywords": structured_output.keywords,
    }

def research_node(state: BlogState)-> BlogState:
    print("Starting researcher agent...")
    context_message= f"Research topic: {state["topic"]}\nSections:{','.join(state["sections"])}"

    result= researcher_agent.invoke(input={"messages": [state["query"], AIMessage(content=context_message)]})
    messages= result["messages"]
    last_ai_message:AIMessage= messages[-1]

    llm_with_structure= llm.with_structured_output(ResearchData)
    structured_output: ResearchData= llm_with_structure.invoke(last_ai_message.content)

    print(f"message: {last_ai_message.content[:20]}...")
    print("Done!\n")

    return{
        "messages": messages,
        "research": structured_output.research
    }

def writer_node(state: BlogState):

    print("Starting writer agent...")

    context_message= f"Title: {state['title']}\nSections: {', '.join(state['sections'])}\nResearch:{len(state['research'])} items"

    result= writer_agent.invoke(input= {"messages": [state["query"], AIMessage(content=context_message)]})
    messages= result["messages"]
    last_ai_message: AIMessage= messages[-1]

    llm_with_structure= llm.with_structured_output(ContentDraft)
    structured_output: ContentDraft= llm_with_structure.invoke(last_ai_message)

    print(f"message: {last_ai_message.content[:20]}...")
    print("Done!\n")
    return {
        "messages": messages,
        "draft_markdown": structured_output.draft_markdown,
        "references_markdown": structured_output.references_markdown,
    }

def reviewer_node(state: BlogState) -> BlogState:
    """Node for the reviewer agent to edit and optimize."""
    print("Starting review agent...")

    context_message = f"Draft: {state['draft_markdown']}...\nKeywords: {', '.join(state['keywords'])}"
    
    result = reviewer_agent.invoke(
        input={
            "messages": [
                state["query"],
                AIMessage(content=context_message)
            ]
        }
    )
    messages = result["messages"]
    last_ai_message:AIMessage = messages[-1]

    llm_with_structure = llm.with_structured_output(ReviewedContent)
    structured_output:ReviewedContent = llm_with_structure.invoke(last_ai_message.content)
    
    print(f"message: {last_ai_message.content[:20]}...")
    print("Done!\n")
    
    return {
        "messages": messages,
        "final_markdown": structured_output.final_markdown,
        "seo_score": structured_output.seo_score,
    }

def publisher_node(state: BlogState) -> BlogState:
    """Node for the publisher agent to finalize and publish."""
    print("Starting publisher agent...")
    
    context_message = f"Title: {state['title']}\nContent: {state['final_markdown']}"
    
    result = publisher_agent.invoke(
        input={
            "messages": [
                state["query"],
                AIMessage(content=context_message)
            ]
        }
    )
    messages = result["messages"]
    last_ai_message:AIMessage = messages[-1]
    
    llm_with_structure = llm.with_structured_output(PublicationStatus)
    structured_output:PublicationStatus = llm_with_structure.invoke(last_ai_message.content)
    
    print(f"message: {last_ai_message.content[:20]}...")
    print("Done!\n")
    
    return {
        "messages": messages,
        "publish_status": structured_output.publish_status,
    }

# WORKFLOW DE LANGGRAPH

workflow= StateGraph(BlogState)

workflow.add_node("intake", intake_node)
workflow.add_node("research", research_node)
workflow.add_node("writer", writer_node)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("publisher", publisher_node)

workflow.add_edge(START, "intake")
workflow.add_edge("intake", "research")
workflow.add_edge("research", "writer")
workflow.add_edge("writer", "reviewer")
workflow.add_edge("reviewer", "publisher")
workflow.add_edge("publisher", END)

graph= workflow.compile()

workflow_png= graph.get_graph().draw_mermaid_png()
#with open("workflow.png", "wb") as f:
    #f.write(workflow_png)

user_message= HumanMessage(content= "I want a blog post about Kingdom Hearts 2")

result= graph.invoke(input={"query": user_message})

result["publish_status"]
result["final_markdown"]
for message in result["messages"]:
    message.pretty_print()
```