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



