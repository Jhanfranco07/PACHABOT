# PACHABOT - Asistente Inteligente para Orientacion Ciudadana

Prototipo local en Python de un asistente conversacional para orientacion ciudadana en tramites municipales, enfocado inicialmente en comercio ambulatorio de la Municipalidad Distrital de Pachacamac. La ficha de tramite cargada identifica como area responsable a la Gerencia de Turismo y Desarrollo Economico / subgerencia competente; esta denominacion debe validarse institucionalmente antes de publicacion. Funciona en Telegram, consola y simulador web.

## Objetivo

El proyecto responde consultas normativas e informativas usando como base principal:

- Ordenanza 108-2012-MDP/C
- Ordenanza 227-2019-MDP/C

Prioridades del prototipo:

- responder en espanol claro y ciudadano
- limitarse al dominio de comercio ambulatorio
- basarse en la documentacion cargada
- citar ordenanza y articulo cuando sea posible
- permitir preguntas libres y preguntas de seguimiento
- seguir funcionando incluso si el LLM externo no esta disponible

## Arquitectura actual

El proyecto ya no es solo un bot con busqueda simple. Ahora tiene estas capas:

- `channels/`: adapta Telegram al formato interno del asistente
- `memory/`: guarda el historial reciente por chat
- `tools/`: concentra herramientas documentales para reescribir consultas y recuperar evidencia
- `services/`: contiene el orquestador, el router, la recuperacion y la capa LLM
- `api/`: expone un endpoint HTTP para probar el asistente fuera de Telegram

## Flujo de consulta

```text
Mensaje del ciudadano
  -> normalizacion y deteccion de intencion
  -> reescritura si es una pregunta de seguimiento
  -> retrieval local: tramites -> FAQ -> chunks -> norma consolidada
  -> evaluacion de evidencia y nivel de confianza
  -> Ollama solo si existe evidencia suficiente
  -> respuesta ciudadana breve con fuentes
  -> historial y, opcionalmente, traza de depuracion
```

`EvidenceService` conserva para cada soporte utilizado el tipo de fuente, score,
extracto base y advertencias de revision. Si no existe soporte suficiente, el
LLM no recibe contexto para improvisar: el sistema responde con un fallback
controlado y deriva al area municipal competente.

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
|   |-- bot/
|   |   |-- __init__.py
|   |   |-- handlers.py
|   |   |-- keyboards.py
|   |   `-- telegram_bot.py
|   |-- channels/
|   |   |-- __init__.py
|   |   |-- schemas.py
|   |   `-- telegram.py
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
|   |   |-- ordenanza_108_2012.txt
|   |   `-- ordenanza_227_2019.txt
|   |-- cleaned/
|   |-- consolidated/
|   |   `-- norma_consolidada.json
|   |-- processed/
|   |   `-- chunks.json
|   |-- tramites/
|   |   `-- comercio_ambulatorio.json
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

- `TELEGRAM_BOT_TOKEN`
- `LLM_PROVIDER`

Opcion recomendada para pruebas locales con Ollama, sin API key:

- `LLM_PROVIDER=ollama`
- `OLLAMA_BASE_URL=http://localhost:11434`
- `OLLAMA_MODEL=qwen3.5:4b`
- `OLLAMA_TIMEOUT=120`
- `OLLAMA_THINK=false`
- `OLLAMA_TEMPERATURE=0.2`
- `OLLAMA_MAX_TOKENS=400`
- `OLLAMA_KEEP_ALIVE=5m`
- `LLM_MODE=auto`

Confirma que el modelo local esta disponible antes de iniciar el bot:

```powershell
ollama list
Invoke-RestMethod http://localhost:11434/api/tags
```

Tambien puedes usar proveedores remotos:

- `LLM_PROVIDER=openrouter` con `OPENROUTER_API_KEY` y `CHAT_MODEL`
- `LLM_PROVIDER=groq`
- `GROQ_API_KEY`
- `GROQ_BASE_URL=https://api.groq.com/openai/v1`

Si vas a usar xAI/Grok:

- `GROK_API_KEY`
- `GROK_BASE_URL=https://api.x.ai/v1`

Si todavia no tienes un LLM operativo, puedes dejar:

```env
LLM_MODE=mock
```

## Variables principales

Ejemplo base:

```env
TELEGRAM_BOT_TOKEN=
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3.5:4b
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
OPENAI_API_KEY=
OPENAI_BASE_URL=
APP_ENV=development
LOG_LEVEL=INFO
EMBEDDING_MODEL=local-tfidf
CHAT_MODEL=mistralai/mistral-small-3.1-24b-instruct:free
CHAT_MODEL_FALLBACKS=google/gemma-3-12b-it:free
MODEL_RETRY_COOLDOWN_SECONDS=180
LLM_MODE=auto
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

- `ollama` usa la API local `POST /api/generate`; no requiere API key y queda encapsulado en `app/services/llm_service.py`.
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
- FAQ: edita `data/faq/comercio_ambulatorio_faq.json` manteniendo `fuentes` y marcas de revision.
- Regeneracion: despues de cambiar ordenanzas, ejecuta `python scripts/ingest_documents.py`.

No cargues un monto, plazo o restriccion como definitivo sin una fuente vigente.

## Ejecutar el bot de Telegram

Con la aplicacion de Ollama ejecutandose en Windows y un modelo instalado:

```powershell
python run.py
```

Comandos disponibles en Telegram:

- `/start`
- `/help`
- `/reset`
- `/estado`

## Probar en consola de VS Code

Para conversar con el asistente sin abrir Telegram, usa el canal local interactivo.
Este canal utiliza el mismo RAG, memoria y proveedor Ollama que el bot.

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

Para ver fuentes, intent y si la respuesta uso Ollama:

```powershell
.\.venv\Scripts\python.exe run_console.py --debug
```

Si aparece un error de memoria al cargar `qwen3.5:4b`, instala y selecciona el
modelo liviano para desarrollo:

```powershell
ollama pull qwen3.5:0.8b
```

Luego cambia en `.env`:

```env
OLLAMA_MODEL=qwen3.5:0.8b
```

Conserva `qwen3.5:4b` para equipos con mas memoria disponible o pruebas de mayor calidad.

## Simulador web tipo chat

Para probar el bot en una interfaz similar a mensajeria, sin abrir Telegram:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Abre en el navegador:

```text
http://127.0.0.1:8000/simulator
```

Desde la pantalla puedes:

- elegir entre los modelos Ollama instalados, por ejemplo `qwen3.5:0.8b` o `qwen3.5:4b`;
- usar `Veloz`, que envía `think=false` y es el recomendado para probar el bot;
- usar `Pensamiento`, que envía `think=true` y puede tardar bastante mas;
- reiniciar la conversación conservando el mismo servidor local.

La seleccion web es temporal: solo se mantiene mientras la API esta ejecutandose.
Para Telegram, deja el modelo definitivo configurado en `.env`.

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

- `Que requisitos necesito para una autorizacion`
- `Cuanto mide un modulo`
- `Cuanto se paga de SISA`
- `Que zonas son rigidas`
- `Que dice el articulo 7`
- `Explicame la autorizacion municipal`
- `Y cuanto dura?`

## Comportamiento esperado

- si la consulta esta dentro del dominio, el sistema busca evidencia y responde con lenguaje natural
- si la consulta es de seguimiento, intenta aprovechar el contexto del chat
- si la evidencia es debil, responde con honestidad y evita inventar
- las respuestas municipales solicitan brevedad y evitan agregar datos no preguntados
- si no se recuperan chunks para un dato municipal, Ollama recibe la instruccion de indicar que no existe evidencia suficiente, sin inventar datos
- si la consulta esta fuera del dominio, limita la conversacion a comercio ambulatorio
- si el LLM externo falla, el bot sigue respondiendo con un fallback local

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

- `RAG-Telegram-Bot-LangChain-OpenAI`: es valiosa su separacion `bot/` y `core/`, junto con pruebas de limpieza, split estructural, reformulacion y versionado. PachaBot mantiene esa separacion mediante `channels/`, `services/`, `documents/` y sus reglas de vigencia.
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
- WhatsApp aun no esta conectado, aunque la arquitectura ya esta separada por canal
- si xAI devuelve `403` por falta de creditos, el proyecto trabajara en fallback local
- los datos de area responsable, TUPA y ubicaciones deben revisarse con la municipalidad antes de publicacion

## Proximos pasos sugeridos

- agregar parser PDF/OCR con metadatos de pagina y limpieza juridica
- exponer retrieval documental mediante MCP manteniendo archivos locales
- sumar un canal WhatsApp reutilizando `channels/`
- evaluar ChromaDB o FAISS con embeddings locales como indice semantico opcional
- crear panel administrativo para revisar fuentes, conflictos y trazas
- ejecutar evaluacion de usabilidad SUS con usuarios y pruebas de exactitud documental
