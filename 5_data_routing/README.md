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
  - compilar el workflow de fintech