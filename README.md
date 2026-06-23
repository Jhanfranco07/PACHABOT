# PACHABOT - Asistente Virtual Inteligente Conversacional Municipal

PachaBot es un Asistente Virtual Inteligente Conversacional disenado para orientar a ciudadanos y emprendedores en tramites municipales. Utiliza modelos de lenguaje, recuperacion aumentada por generacion (RAG), memoria conversacional y consulta de fuentes documentales oficiales para entregar respuestas claras, trazables y sustentadas.

El proyecto esta orientado a los tramites municipales gestionados por la Gerencia de Licenciasy Desarrollo Economico de la Municipalidad Distrital de Pachacamac. 

Aunque **inicialmente esta enfocado en comercio ambulatorio**, la arquitectura esta disenada para escalar e incorporar proximamente bases de conocimiento para **licencias de funcionamiento** y **anuncios publicitarios**, cubriendo asi los permisos principales de dicha gerencia.

Funciona localmente en consola, FastAPI y simulador web. El despliegue para el usuario final se orientara a WhatsApp o un portal web institucional. Su arquitectura queda preparada para integrar herramientas especializadas mediante MCP.

## Objetivo

El proyecto no busca ser un chatbot basico de preguntas frecuentes. Busca orientar conversacionalmente al ciudadano: interpreta su consulta, identifica la intencion, usa memoria si hay seguimiento, recupera evidencia documental, valida si la informacion alcanza, genera una respuesta clara y advierte limites cuando corresponde.

Fuentes normativas principales:

- Ordenanza 108-2012-MDP/C
- Ordenanza 227-2019-MDP/C
- Base de Conocimiento Simplificada (Lenguaje Ciudadano)

Prioridades del prototipo:

- responder en espanol claro y ciudadano
- limitarse al dominio de comercio ambulatorio
- basarse en la documentacion cargada
- citar ordenanza y articulo cuando sea posible
- permitir preguntas libres y preguntas de seguimiento
- seguir funcionando incluso si el LLM externo no esta disponible
- preparar una futura integracion MCP con herramientas municipales especializadas

## Diferencia entre chatbot basico y PachaBot

Un chatbot basico suele responder con reglas, menus o preguntas frecuentes fijas. PachaBot funciona como asistente conversacional porque:

- interpreta la intencion del ciudadano, por ejemplo requisitos, costos, zonas, sanciones o normativa;
- usa memoria para entender repreguntas como "y cuanto cuesta?" o "y si es en Manchay?";
- decide que fuente consultar: fichas de tramite, FAQ, chunks normativos, norma consolidada o fuentes estructuradas;
- recupera evidencia con metadatos, score, tipo de fuente y vigencia;
- valida si la evidencia es suficiente antes de generar una respuesta;
- evita inventar cuando falta respaldo documental;
- deriva al area municipal competente cuando la consulta requiere verificacion humana;
- deja trazabilidad para debug y futura auditoria;
- esta preparado para conectar herramientas externas mediante MCP.

## Arquitectura conversacional inteligente

El proyecto esta organizado como un agente conversacional con recuperacion documental local:

- `channels/`: adapta los canales de comunicacion (web, WhatsApp, etc.) al formato interno del asistente
- `memory/`: guarda historial, modo, ultima intencion, fuentes usadas y advertencias
- `tools/`: concentra herramientas documentales para reescribir consultas y recuperar evidencia
- `services/`: contiene el orquestador inteligente, router, rewriter, retrieval, evidencia y LLM
- `api/`: expone un endpoint HTTP para interactuar con el asistente

Flujo:

```text
Entrada del usuario
  -> canal mensajeria/web/consola
  -> normalizacion
  -> router de intencion
  -> memoria conversacional
  -> reformulacion de consulta
  -> recuperacion documental
  -> validacion de evidencia
  -> generacion con LLM
  -> respuesta con fuente o advertencia
  -> registro en historial/debug
```

`EvidenceService` conserva para cada soporte utilizado el tipo de fuente, score,
extracto base y advertencias de revision. Si no existe soporte suficiente, el
LLM no recibe contexto para improvisar: el sistema responde con un fallback
controlado y deriva al area municipal competente.

## Por que PachaBot no debe funcionar con respuestas quemadas

Las respuestas fijas sirven para menus simples, pero no para orientacion municipal conversacional. Una misma duda ciudadana puede escribirse como "permiso", "autorizacion", "quiero vender", "puesto", "stand", "no cumplo" o "me pueden quitar mi modulo". Por eso PachaBot no debe crecer agregando respuestas pegadas en codigo.

El flujo correcto es:

- detectar una intencion general, sin imponer una respuesta;
- reconstruir la consulta si depende del historial;
- ampliar la busqueda con terminos relacionados;
- recuperar evidencia desde tramites, FAQ, chunks, norma consolidada y JSON estructurados;
- validar si la evidencia alcanza;
- pedir al LLM que explique la evidencia en lenguaje ciudadano;
- usar fallback solo cuando realmente no haya respaldo suficiente.

## Recuperacion antes del fallback

Antes de decir que no encontro informacion suficiente, PachaBot intenta varias busquedas:

- consulta original del ciudadano;
- consulta reformulada con memoria conversacional;
- consulta expandida por sinonimos municipales;
- busqueda por intencion, por ejemplo obligaciones, sanciones, zonas, ferias o requisitos;
- busqueda en fuentes estructuradas como tramites, FAQ y zonas restringidas;
- busqueda en chunks normativos y norma consolidada.

Ejemplos de expansion:

- "no cumple" se relaciona con incumplimiento, sancion, revocacion, retiro y fiscalizacion;
- "puesto" se relaciona con modulo, stand, mobiliario y espacio autorizado;
- "permiso" se relaciona con autorizacion municipal y resolucion;
- "zona prohibida" se relaciona con zona rigida, zona restringida y ubicacion no autorizada;
- "feria" se relaciona con feriante, recinto ferial, stand y autorizacion de feria.

El fallback se mantiene por seguridad, pero queda como ultimo recurso. Si existe evidencia parcial, el asistente responde lo que si esta respaldado y aclara que parte debe validarse.

## Requisitos diferenciados de comercio ambulatorio

PachaBot separa los requisitos de comercio ambulatorio en dos casos para no
mezclar documentos:

- tramite nuevo / ingreso al padron municipal: aplica cuando la persona quiere
  vender por primera vez, inscribirse o pedir permiso para vender en la via
  publica.
- renovacion: aplica cuando la persona ya tiene autorizacion, su permiso esta
  por vencer o quiere seguir vendiendo con autorizacion municipal.

La fuente principal de esta separacion es
`data/tramites/requisitos_comercio_ambulatorio.json`. La ordenanza funciona como
respaldo normativo general, pero la orientacion ciudadana prioriza la ficha
interna diferenciada.

Reglas de respuesta:

- si el ciudadano pide permiso por primera vez, el asistente usa la seccion
  `nuevo_ingreso_padron`;
- si pregunta por renovar, usa la seccion `renovacion`;
- si no queda claro, pregunta si es primera vez o renovacion;
- no mezcla fotos carne o voucher como requisitos principales del tramite nuevo;
- no inventa costos: el monto exacto se valida con el TUPA vigente;
- cita la fuente al final de forma breve, sin mostrar estados internos ni
  metadatos del RAG.

## Preparacion futura para MCP

MCP permitira conectar este asistente conversacional con herramientas externas o internas sin cambiar la arquitectura principal. Posibles herramientas:

- consulta de requisitos estructurados por tramite;
- consulta de costos o TUPA vigente;
- busqueda normativa con filtros por articulo y vigencia;
- consulta de expedientes, si existiera autorizacion futura;
- conexion con bases internas o APIs simuladas para evaluacion academica;
- exposicion de recursos documentales para panel administrativo o auditoria.

Por ahora, estas integraciones no son obligatorias: PachaBot funciona con OpenAI API como proveedor principal, conserva Ollama como alternativa local opcional y mantiene el RAG sobre archivos JSON/TXT.

## Decision tecnica del RAG

Para este prototipo se usa un motor local con `scikit-learn`, `TF-IDF` y similitud coseno. Se eligio asi porque:

- es facil de correr en local
- no depende de embeddings externos
- es mas simple para un prototipo universitario
- deja el proyecto listo para migrar despues a FAISS, ChromaDB o embeddings externos

Encima de esa base ya existe una capa mas inteligente:

- segmentacion juridica por titulo y articulo
- busqueda hibrida con prioridad legal
- reescritura de consultas de seguimiento
- memoria por sesion
- respuesta sintetizada por LLM o fallback local

## Estructura

```text
project_root/
|-- app/
|   |-- api/
|   |   |-- __init__.py
|   |   `-- schemas.py
|   |-- channels/
|   |   |-- __init__.py
|   |   |-- schemas.py
|   |   `-- web.py
|   |-- core/
|   |   |-- __init__.py
|   |   |-- logger.py
|   |   `-- prompts.py
|   |-- memory/
|   |   |-- __init__.py
|   |   `-- conversation_store.py
|   |-- models/
|   |   |-- __init__.py
|   |   |-- domain.py
|   |   `-- schemas.py
|   |-- services/
|   |   |-- __init__.py
|   |   |-- assistant_service.py
|   |   |-- document_service.py
|   |   |-- evidence_service.py
|   |   |-- llm_service.py
|   |   |-- query_rewriter.py
|   |   |-- query_router.py
|   |   `-- retrieval_service.py
|   |-- tools/
|   |   |-- __init__.py
|   |   `-- document_toolkit.py
|   |-- utils/
|   |   |-- __init__.py
|   |   |-- chunking.py
|   |   |-- docx_extractor.py
|   |   |-- helpers.py
|   |   `-- text_cleaner.py
|   |-- __init__.py
|   |-- config.py
|   `-- main.py
|-- data/
|   |-- raw/
|   |   |-- base_conocimiento_comercio_ambulatorio_pachacamac.txt
|   |   |-- ordenanza_108_2012.txt
|   |   `-- ordenanza_227_2019.txt
|   |-- cleaned/
|   |   `-- base_conocimiento_comercio_ambulatorio_pachacamac.cleaned.txt
|   |-- consolidated/
|   |   `-- norma_consolidada.json
|   |-- processed/
|   |   |-- conversations/
|   |   `-- chunks.json
|   |-- tramites/
|   |   |-- comercio_ambulatorio.json
|   |   |-- zonas_restringidas_comercio_ambulatorio.json
|   |   `-- ordenanza_227_articulos_57_64.json
|   |-- faq/
|   |   `-- comercio_ambulatorio_faq.json
|   |-- vectorstore/
|   `-- runtime/
|       |-- chat_modes/
|       |-- conversations/
|       `-- debug/
|-- scripts/
|   |-- import_docx_documents.py
|   |-- ingest_documents.py
|   `-- reset_vectorstore.py
|-- tests/
|   |-- test_assistant_conversation.py
|   |-- test_chunking.py
|   |-- test_retrieval.py
|   `-- test_router.py
|-- .env.example
|-- README.md
|-- run_console.py
|-- requirements.txt
`-- run.py
```

## Instalacion paso a paso

### 1. Crear entorno virtual

En PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Si PowerShell bloquea la activacion:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### 2. Instalar dependencias

```powershell
pip install -r requirements.txt
```

### 3. Configurar variables de entorno

```powershell
Copy-Item .env.example .env
```

Completa al menos:

- `OPENAI_API_KEY`

Por defecto PachaBot usa OpenAI API:

- `LLM_PROVIDER=openai`
- `OPENAI_API_KEY=tu_api_key`
- `OPENAI_MODEL=gpt-5.4-mini`
- `OPENAI_MAX_OUTPUT_TOKENS=500`
- `OPENAI_TEMPERATURE=0.2`
- `LLM_MODE=auto`

Tambien puedes usar proveedores remotos compatibles:

- `LLM_PROVIDER=openrouter` con `OPENROUTER_API_KEY` y `CHAT_MODEL`
- `LLM_PROVIDER=groq` con `GROQ_API_KEY`
- `GROQ_BASE_URL=https://api.groq.com/openai/v1`

Si vas a usar xAI/Grok:

- `GROK_API_KEY`
- `GROK_BASE_URL=https://api.x.ai/v1`

Si todavia no tienes un LLM operativo para pruebas de interfaz, puedes dejar:

```env
LLM_MODE=mock
```

### Uso de OpenAI como proveedor principal y Ollama como opcion local

PachaBot queda configurado para usar OpenAI API como proveedor principal. Ollama sigue disponible como alternativa local, pero esta apagado por defecto para evitar consumo de memoria y lentitud del equipo.

Con OpenAI activo, el sistema no consulta `http://localhost:11434`, no lista modelos locales y no intenta iniciar Ollama:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=tu_api_key
OPENAI_MODEL=gpt-5.4-mini
OPENAI_MAX_OUTPUT_TOKENS=500
OPENAI_TEMPERATURE=0.2
OPENAI_BASE_URL=https://api.openai.com/v1
OLLAMA_ENABLED=false
```

Para activar Ollama manualmente algun dia:

```env
LLM_PROVIDER=ollama
OLLAMA_ENABLED=true
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
```

Antes de usarlo, inicia Ollama por tu cuenta y confirma que el modelo exista:

```powershell
ollama list
```

Para volver a OpenAI:

```env
LLM_PROVIDER=openai
OLLAMA_ENABLED=false
```

## Variables principales

Ejemplo base:

```env
LLM_PROVIDER=openai
LLM_MODE=auto
OPENAI_API_KEY=
OPENAI_MODEL=gpt-5.4-mini
OPENAI_MAX_OUTPUT_TOKENS=500
OPENAI_TEMPERATURE=0.2
OPENAI_BASE_URL=https://api.openai.com/v1
OLLAMA_ENABLED=false
OLLAMA_MODEL=llama3.1
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT=120
OLLAMA_THINK=false
OLLAMA_TEMPERATURE=0.2
OLLAMA_MAX_TOKENS=400
OLLAMA_KEEP_ALIVE=5m
OPENROUTER_API_KEY=
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_HTTP_REFERER=http://localhost
OPENROUTER_APP_NAME=PachaBot
GROK_API_KEY=
GROK_BASE_URL=https://api.x.ai/v1
GROQ_API_KEY=
GROQ_BASE_URL=https://api.groq.com/openai/v1
APP_ENV=development
LOG_LEVEL=INFO
EMBEDDING_MODEL=local-tfidf
CHAT_MODEL=gpt-5.4-mini
CHAT_MODEL_FALLBACKS=
MODEL_RETRY_COOLDOWN_SECONDS=180
RETRIEVAL_TOP_K=4
RETRIEVAL_MIN_SCORE=0.35
RETRIEVAL_MAX_RESULTS=5
CHUNK_SIZE=700
CHUNK_OVERLAP=120
CONFIDENCE_THRESHOLD=0.05
MEMORY_HISTORY_LIMIT=12
MEMORY_MAX_TURNS=40
ASSISTANT_MAX_SOURCES=3
RAG_DEBUG_TRACE=false
```

Notas:

- `openai` es el proveedor principal por defecto y usa `OPENAI_MODEL`.
- `ollama` usa la API local `POST /api/generate`; no requiere API key, pero solo se consulta si `LLM_PROVIDER=ollama` y `OLLAMA_ENABLED=true`.
- `OLLAMA_THINK=false` evita razonamiento extenso en modelos Qwen para respuestas ciudadanas; si la ejecucion solo usa CPU, aumenta `OLLAMA_TIMEOUT` localmente.
- `data/processed/` contiene artefactos de RAG; las conversaciones y modos nuevos se guardan en `data/runtime/`.
- La aplicacion puede leer temporalmente sesiones antiguas en `data/processed/conversations/` y `data/processed/chat_modes/`.
- `RAG_DEBUG_TRACE=true` escribe trazas JSONL por sesion en `data/runtime/debug/`, con consulta reformulada, confianza y evidencia; dejalo desactivado para uso cotidiano.
- `openrouter` y `groq` funcionan bien con la arquitectura actual porque exponen APIs compatibles con el cliente de OpenAI.
- Este proyecto todavia no trae busqueda web automatica. Un proveedor externo mejora mucho la conversacion general, pero no equivale por si solo a navegar internet.

## Cargar documentos desde Word

Si tienes las ordenanzas en `.docx`, puedes importarlas a `data/raw/` asi:

```powershell
python scripts/import_docx_documents.py
```

Ese script extrae el texto y lo deja listo para la siguiente etapa.

## Construir el indice documental

Despues de tener los `.txt` en `data/raw/`:

```powershell
python scripts/ingest_documents.py
```

Este proceso:

- limpia texto
- segmenta por secciones y articulos
- incorpora metadatos de fuente, articulo, vigencia, tipo de contenido y exclusion de retrieval
- genera la norma consolidada y sus advertencias de validacion
- genera `chunks.json`
- construye un indice local conjunto con tramites, FAQ, chunks y norma consolidada en `data/vectorstore/`

Si quieres reiniciar el indice:

```powershell
python scripts/reset_vectorstore.py
python scripts/ingest_documents.py
```

## Agregar conocimiento municipal

- Ordenanzas fuente: agrega texto extraido y revisado a `data/raw/`; conserva originales fuera del indice si aun requieren OCR o correccion.
- Tramites: completa `data/tramites/comercio_ambulatorio.json` con datos validados por el area responsable y TUPA vigente.
- Requisitos diferenciados: actualiza `data/tramites/requisitos_comercio_ambulatorio.json` cuando el area municipal cambie documentos para tramite nuevo o renovacion.
- FAQ: edita `data/faq/comercio_ambulatorio_faq.json` manteniendo `fuentes` y marcas de revision.
- Regeneracion: despues de cambiar ordenanzas, ejecuta `python scripts/ingest_documents.py`.

No cargues un monto, plazo o restriccion como definitivo sin una fuente vigente.

## Probar en consola de VS Code

Para conversar con el asistente localmente sin depender de plataformas de mensajeria externas, usa el canal local interactivo.
Este canal utiliza el mismo RAG, memoria y proveedor LLM configurado que el asistente.

```powershell
.\.venv\Scripts\python.exe run_console.py
```

La consola mantiene una sola conversación natural. Cuando preguntas por comercio
ambulatorio, el asistente recupera automáticamente la documentación municipal.
Ejemplo:

```text
Tu: Que requisitos necesito para vender en la via publica?
PachaBot: ...
```

Comandos disponibles:

- `/reset` - borrar la memoria de la sesion local.
- `/estado` - ver proveedor, modelo y cantidad de chunks.
- `/ayuda` - listar comandos.
- `/salir` - terminar.

Para ver fuentes, intent y si la respuesta uso IA externa:

```powershell
.\.venv\Scripts\python.exe run_console.py --debug
```

Si decides activar Ollama y aparece un error de memoria al cargar un modelo local,
instala y selecciona un modelo mas liviano para desarrollo:

```powershell
ollama pull qwen3.5:0.8b
```

Luego cambia en `.env`:

```env
LLM_PROVIDER=ollama
OLLAMA_ENABLED=true
OLLAMA_MODEL=qwen3.5:0.8b
```

Vuelve a `LLM_PROVIDER=openai` y `OLLAMA_ENABLED=false` si quieres evitar consumo local.

## Simulador web tipo chat

Para probar el asistente en una interfaz web institucional similar a mensajeria:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Abre en el navegador:

```text
http://127.0.0.1:8000/simulator
```

Desde la pantalla puedes:

- conversar con OpenAI cuando `LLM_PROVIDER=openai`;
- elegir entre modelos Ollama instalados solo si `LLM_PROVIDER=ollama` y `OLLAMA_ENABLED=true`;
- usar `Veloz`, que envía `think=false` y es el recomendado para probar el asistente;
- usar `Pensamiento`, que envía `think=true` y puede tardar bastante mas;
- reiniciar la conversación conservando el mismo servidor local.

La seleccion web de Ollama es temporal: solo se mantiene mientras la API esta ejecutandose.
Para la ejecucion principal, deja el proveedor definitivo configurado en `.env`.

## Probar por HTTP

Tambien puedes levantar FastAPI:

```powershell
uvicorn app.main:app --reload
```

Endpoints disponibles:

- `GET /health`
- `GET /info`
- `GET /simulator`
- `GET /ollama/config`
- `POST /ollama/config`
- `POST /chat`
- `POST /chat/reset`

`POST /chat` retorna tambien `evidence`, `evidence_warning` y
`confidence_level`, utiles para auditoria del prototipo y para una futura
pantalla administrativa.

Ejemplo de prueba:

```json
POST /chat
{
  "channel": "api",
  "session_id": "demo-1",
  "user_id": "usuario-demo",
  "text": "Que dice el articulo 7"
}
```

## Ejemplos de preguntas

- `Como saco mi permiso de comercio ambulatorio`
- `Que necesito para vender`
- `Como renuevo mi permiso`
- `Requisitos de comercio ambulatorio`
- `Que necesito y cuanto cuesta`
- `Cuanto mide un modulo`
- `Cuanto se paga de SISA`
- `Que es el SISA`
- `Que zonas son rigidas`
- `Que dice el articulo 7`
- `Explicame la autorizacion municipal`
- `Y cuanto dura?`

## Comportamiento esperado

- si la consulta esta dentro del dominio, el sistema busca evidencia y responde con lenguaje natural
- soporta el cambio de contexto ("Modo General" y "Modo Comercio") para separar conversaciones libres del asesoramiento normativo
- si la consulta es de seguimiento, intenta aprovechar el contexto del chat
- si la evidencia es debil, responde con honestidad y evita inventar
- las respuestas municipales solicitan brevedad y evitan agregar datos no preguntados
- si no se recupera evidencia para un dato municipal, el asistente responde con fallback controlado sin inventar datos
- si la consulta esta fuera del dominio, limita la conversacion a comercio ambulatorio
- si el LLM externo falla, el asistente sigue respondiendo con un fallback local

## Pruebas

```powershell
pytest
```

Las pruebas cubren intencion, reformulacion de seguimientos, retrieval,
vigencia legal, respuesta sin evidencia, evaluacion/traza de evidencia,
ingesta documental y simulador web.
`pytest.ini` limita la recoleccion a `tests/`, de modo que los repositorios
descargados en `referencias-chatbots/` no se mezclen con las pruebas del sistema.

## Referentes revisados y decisiones

- Repositorios de asistentes RAG: es valiosa su separacion de canales y capa `core/`, junto con pruebas de limpieza, split estructural, reformulacion y versionado. PachaBot mantiene esa separacion modular mediante `channels/`, `services/`, `documents/` y sus reglas de vigencia.
- `AI-RAG-Assistant-Chatbot`: aporta historial, fuentes persistidas, degradacion controlada y preparacion MCP. Sus dependencias cloud (Pinecone, Neo4j y servicios de despliegue) no se adoptan porque PachaBot debe seguir local.
- `pathwaycom/llm-app`: aporta la idea de ingesta observable, entradas consultables y servidor MCP documental. Se toma como direccion futura; la ingesta actual permanece simple con archivos y TF-IDF.
- `datvodinh/rag-chatbot`: confirma un flujo local con Ollama, PDFs, memoria y combinacion de recuperadores. Chroma/BM25/embeddings quedan como evolucion opcional despues de estabilizar el corpus juridico.

Las mejoras aplicadas conservan la estructura existente: conocimiento
ciudadano primero, norma vigente como respaldo, evidencia auditable antes de
generacion y servicios externos solo opcionales.

## Limitaciones actuales

- la calidad sigue dependiendo de la calidad del texto cargado
- el indice TF-IDF es adecuado para el corpus inicial, pero pierde sinonimos y variaciones semanticas amplias
- no hay OCR ni extractor PDF general integrado; la ingesta vigente parte de texto o DOCX previamente extraido
- no hay panel admin
- el canal definitivo (WhatsApp o portal web) aun no esta conectado, aunque la arquitectura modular en `channels/` ya lo soporta
- si xAI devuelve `403` por falta de creditos, el proyecto trabajara en fallback local
- los datos de area responsable, TUPA y ubicaciones deben revisarse con la municipalidad antes de publicacion

## Proximos pasos sugeridos

- agregar parser PDF/OCR con metadatos de pagina y limpieza juridica
- exponer retrieval documental mediante MCP manteniendo archivos locales
- integrar el canal definitivo para el usuario final (WhatsApp o un portal web) reutilizando `channels/`
- evaluar ChromaDB o FAISS con embeddings locales como indice semantico opcional
- ampliar la base documental con tramites de licencias de funcionamiento y anuncios publicitarios
- crear panel administrativo para revisar fuentes, conflictos y trazas
- ejecutar evaluacion de usabilidad SUS con usuarios y pruebas de exactitud documental

## Arquitectura actual del prototipo

PachaBot esta organizado como un Asistente Virtual Inteligente Conversacional Municipal. No funciona como un menu fijo ni como una lista de respuestas quemadas: interpreta la consulta del vecino, revisa memoria conversacional, recupera evidencia documental y usa el LLM para explicar la respuesta en lenguaje ciudadano.

Flujo principal:

1. El ciudadano escribe por Telegram, consola, API o simulador web.
2. `AssistantService` recibe el mensaje y coordina todo el flujo.
3. `QueryRouter` detecta una primera intencion con reglas flexibles.
4. `IntentInterpreterService` usa el LLM como apoyo cuando la intencion es ambigua o de baja confianza.
5. `QueryRewriter` reformula seguimientos como "y cuanto cuesta", "el segundo requisito" o "art 36" sin perder el contexto.
6. `DocumentToolkit` prepara la busqueda documental y ordena la evidencia.
7. `RetrievalService` busca en tramites, FAQ, chunks normativos, norma consolidada y datos estructurados.
8. El validador de evidencia decide si hay sustento suficiente, parcial o si corresponde fallback.
9. `LLMService` genera la respuesta final con OpenAI por defecto u Ollama si se activa manualmente.
10. La memoria guarda historial, tema, ultima intencion, fuentes usadas y opciones de continuidad.

Estructura de capas:

```text
app/
|-- bot/                  # Canal Telegram
|-- channels/             # Adaptadores de canal
|-- web/                  # Simulador local profesional
|-- services/
|   |-- assistant_service.py
|   |-- intent_interpreter.py
|   |-- query_router.py
|   |-- query_rewriter.py
|   |-- retrieval_service.py
|   |-- llm_service.py
|   `-- document_service.py
|-- tools/
|   `-- document_toolkit.py
|-- memory/               # Historial y estado conversacional
|-- prompts/              # Prompt institucional y reglas de tono
|-- models/               # Esquemas internos
`-- main.py               # FastAPI y simulador

data/
|-- raw/                  # Ordenanzas originales
|-- cleaned/              # Texto limpio
|-- consolidated/         # Norma vigente consolidada
|-- processed/            # Chunks para RAG
|-- tramites/             # Fichas estructuradas
|-- faq/                  # Preguntas frecuentes orientativas
|-- vectorstore/          # Indice local
`-- runtime/              # Conversaciones, modos y debug
```

Componentes clave:

- `AssistantService`: orquestador conversacional. Evita responder de forma aislada y decide cuando usar memoria, RAG, LLM o fallback.
- `IntentInterpreterService`: segunda capa inteligente para consultas ambiguas. Si el router no esta seguro, el LLM ayuda a interpretar o pedir aclaracion.
- `QueryRouter`: clasifica intenciones como requisitos, renovacion, costos, zonas, sanciones, definiciones, rubros y articulos normativos.
- `QueryRewriter`: convierte seguimientos en consultas completas, por ejemplo "y el costo?" o "que dice el art. 36".
- `RetrievalService`: recupera evidencia por fuente, intencion, sinonimos, articulos y metadatos.
- `DocumentToolkit`: arma el contexto limpio para que el LLM no reciba JSON crudo ni informacion contradictoria.
- `LLMService`: usa OpenAI como proveedor principal. Ollama queda disponible, pero no se consulta si `LLM_PROVIDER=openai`.
- `app/web/index.html`: prototipo visual para demostracion local, con panel de parametros, estado del modelo y chat ciudadano.

## Diferencia con un chatbot basico

Un chatbot basico suele responder por coincidencia de palabras, menus o FAQ rigidas. PachaBot busca comportarse como un orientador municipal:

- entiende preguntas escritas con errores o lenguaje informal
- distingue tramite nuevo, renovacion, costo, zona, rubro, SISA, articulo y seguimiento
- usa memoria para continuar una conversacion
- recupera evidencia antes de responder
- explica en lenguaje claro, sin sonar legalista
- pregunta al ciudadano cuando falta un dato importante
- no inventa costos, requisitos, zonas, sanciones ni articulos

## Uso de OpenAI y Ollama

Por defecto se recomienda usar OpenAI para que el equipo no consuma recursos locales:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=tu_api_key
OPENAI_MODEL=gpt-5.4-mini
OPENAI_MAX_OUTPUT_TOKENS=500
OPENAI_TEMPERATURE=0.2
OLLAMA_ENABLED=false
```

Para usar Ollama manualmente:

```env
LLM_PROVIDER=ollama
OLLAMA_ENABLED=true
OLLAMA_MODEL=qwen3.5:0.8b
OLLAMA_BASE_URL=http://localhost:11434
```

Ollama no se inicia automaticamente. Si `LLM_PROVIDER=openai`, el sistema no consulta `http://localhost:11434`.

## Demo con ngrok

Ngrok sirve para compartir temporalmente el simulador local con otra persona durante la exposicion. Debes mantener abiertas la terminal de FastAPI y la terminal de ngrok.

Terminal 1, levantar PachaBot:

```powershell
cd "C:\Users\PC\Documents\2 - PROYECTOS DEV\BOT"
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Verifica en tu navegador:

```text
http://127.0.0.1:8000/simulator
```

Terminal 2, abrir el puente:

```powershell
ngrok http 8000
```

Ngrok mostrara una URL parecida a:

```text
https://xxxx-xxxx.ngrok-free.app
```

Comparte con tu compañero:

```text
https://xxxx-xxxx.ngrok-free.app/simulator
```

Notas para la demo:

- no cierres ninguna de las dos terminales
- no compartas tu archivo `.env` ni tu `OPENAI_API_KEY`
- en el plan gratuito la URL de ngrok cambia cada vez que reinicias el tunel
- si aparece una pantalla de aviso de ngrok, tu compañero solo debe continuar al sitio
- para cerrar la demo, presiona `Ctrl+C` en la terminal de ngrok y luego en la de FastAPI
