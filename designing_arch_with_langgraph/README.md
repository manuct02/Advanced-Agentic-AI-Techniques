# Ejericio: Multi-Agent Architecture Design

En este ejercicio diseñaremos e implementaremos un sistema multi-agente para un escenario de negocios en el mundo real. Esta práctica refuerza la toma de decisiones de la arquitectura, la definición de roles del agente, el diseño de comunicación...

### Objetivos:
- Definir un escenario de negocio concreto y extraer los requisitos precisos del agente
- Diseñar los roles del agente y seleccionar un patrón de atquitectura apropiado (orchestrator vs perr-to-peer)
- Implementar un protótipo de workflow multiagéntico usando LangGraph, StateGraph y los message handlers
- Visualizar el workflow y reflejar las futuras modificaciones

### **Pasos:**

#### **1. Set up del entorno y los imports**

#### **2. Definir el escenario de negocio**

- Elegimos un escenario de la lista

- Escribimos una statement breve sobre el escenario a tratar que incluya:
  - La meta principal
  - Los requerimientos funcionales (pasos y restricciones)
  - Los requisiton no funcionales (latencia, escalabilidad,...)

#### **3. Especificar los roles del agente y las responsibilidades**

- Definir de 4 a 6 agentes con resposibilidades individuales (por ejemplo, para un caso de e-commerce (compras online)):
  - IngestAgent: parsea las comandas entrantas al formato deseado
  - FraudAgent: califica los potenciales fraudes
  - InventoryAgent: reserva o libera stock
  - PaymentAgent: procesa las transacciones
  - NotificationAgent: informa al cliente
- Listar para cada agente sus respectivos inputs, outputs, comportamiento de éxito/fracaso...

#### **4. Elegir el patrón de arquitectura y documentar el razonamiento detrás de la elección**

- Comparar los patrones contra los requerimientos del escenario:
  - Usar **orchestrator** cuando se requiera un orden estricto, un estado centralizado o haga falta rollback
  - Usar **Peer-topeer**  cuando se prefiera flexibilidad, paralelismo, o montaje en cadena.

- Llenar estos campos:
  - Pattern: ORCHESTRATOR or PEER-TO-PEER
  - Reasoning: map requirements to pattern benefits
  - Data Flow: describir el formato del mensaje y el path (JSON payload con metadata, events o commands)

#### **5. Diseñar la comunicación y el flujo de datos**

- Elegir protocolo de comunicación: métodos de llamada directos (in-proc), colas de mensajes (RabbitMQ/Kafka) o HTTP callbacks.
- Definir el esquema del mensaje: el mínimo de campos como {id, type, payload, status, trace}

#### **6. Implementar agentes y graph**

- Usar las funciones starter como plantilla.
- Para el orquestador:
  - Implementar una función supervisora que lea el estado y devuelva el nombre del siguiente agente o `END`.
  - Implementar cada agente para procesar el estado y devolverlo al supervisor o al `END`.
- Para el peer-to-peer:
  - Implementar agentes que directamente devuelvan el nombre del siguiente agente
- Crear un objeto StateGraph(MessageState), añadirle nodos, y conectarlos con edges para mostrar el camino elegido
- Compilar y visualizar el graph con lo de siempre

#### **7. Probar el prototipo**
- Crear mensajes de muestra que pongan a prueba casos normales y extremos (fallos, timeouts)
- Correr el workflow y validar transiciones y outputs
- Logea las transiciones de estado y usa random o stubs deterministas para llamadas externas para simular respuestas.

#### **8. Producir el diagrama de la arquitectura**

- Exportar el visual producido por LangGraph o dibujar con exclaidraw un diagrama simple mostrando los agentes, flujos y protocolos.


# SOLUCIÓN

## 1. Definir el escenario de negocio

### **Sistema de citas de centros de salud**

Este sistema es capaz de gestionar el ciclo de vida completo de citas médicas desde la petición inicial hasta el seguimiento post-cita. El flujo de acción debe estar estructurado y ser secuencial para asegurar la seguridad dle paciente, la adjudicación adecuada, y el cumplimiento de las regulaciones sanitarias.

**Requisitos clave**:
  - Orden estricto en el flujo de acción para/con la seguridad del paciente.
  - Vista general centralizada para prevenir conflictos de horarios.
  - Verificación del paciente.
  - Manejo de la disponibilidad del doctor.
  - Recordatorios y notificaciones notificados.
  - Registro de la consulta para el cumplimiento de las regulaciones sanitarias.

**Pasos del workflow**:
1. Solicitud de cita recibida
2. Disponibilidad del doctor chequeada
3. Verificación y eligibilidad del paciente confirmada
4. Cita agendada y confirmada
5. Recordatorios mandados al paciente y al doctor


## 2. Definir los roles y responsabilidades de cada agente

#### **Agente 1: Coordinador de citas (el supervisor)**

Responsabilidades:
- Recibe la solicitud de cita inicial
- Orquesta el workflow entero
- Toma las decisiones de routing basándose en el tipo de request y urgencia
- Mantiene el estado del workflow general
- Se encarga de las excepciones y las mejoras

#### **Agente 2: Agente de disponibilidad de los doctores**

Responsabilidades:
- Chequea los horarios del doctor y su disponibilidad
- Identifica espacios de tiempo apropiados para la cita
- Maneja los conflictos de horario
- Actualiza la disponibilidad en tiempo real
- Aporta recomendaciones de disponibilidad (alternativa de doctores, centros...)

#### **Agente 3: Agente de verificación del paciente**

Responsabilidades:
- Valida la identidad y el seguro del paciente
- Chequea si es elegible para la solicitud
- Verifica el historial del paciente y sus preferencias
- Maneja la pre-autorización del seguro
- Maneja los registros del paciente

#### **Agente 4: Agente de horarios**

Responsabilidades:
- Reserva la cita
- Manda la confirmación al paciente y al doctor
- Actualiza los calendarios pertinentes
- Maneja las solicitudes de cambio de hora
- Maneja los conflictos de citas

#### **Agente 5: Agente de notificaciones**

Responsabilidades:
- Envía los recordatorios de las citas
- Maneja las preferencias de comunicación
- Gestiona los SMS, los emails y las notificaciones telefónicas
- Manda instrucciones pre-cita
- Se encarga de las notificaciones de cancelación

## 3. Modelo de Arquitectura

### Patrón de Orquestador (Supervisor)

- Razonamiento: las citas sanitarias requieren un flujo de acción estricto para la seguridad del paciente. EL patrón de supervisor aporta control centralizado para asegurar que cada paso se complete en el orden correcto, previene los conflictos de horario y mantiene los registros de auditoría requeridos para las regulaciones sanitarias.

- Flujo de datos: el coordinador de citas (supervisor) recibe la request inicial y marca la ruta secuenciada entre agentes especializados Cada agente reporta su acción al supervisor, que decide el siguiente paso basándose en los resultados.

## 4. Dibujar el Sistema