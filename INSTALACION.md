# Guía de Instalación — Motor de Decisión Crediticia

Sistema multi-agente de análisis crediticio para SOFIPO, basado en FastAPI, LangGraph y Ollama (modelo Mistral). Todo el procesamiento es **local**: ningún dato sale de la máquina.

---

## Índice

1. [Requisitos del sistema](#1-requisitos-del-sistema)
2. [Instalación en macOS](#2-instalación-en-macos-apple-silicon--intel)
3. [Instalación en Linux](#3-instalación-en-linux-ubuntu-2204--2404-y-debian)
4. [Instalación en Windows](#4-instalación-en-windows)
5. [Dependencias Python](#5-dependencias-python)
6. [Configuración del modelo Ollama](#6-configuración-del-modelo-ollama)
7. [Verificación de la instalación](#7-verificación-de-la-instalación)
8. [Solución de problemas comunes](#8-solución-de-problemas-comunes)

---

## 1. Requisitos del sistema

### Hardware mínimo

| Componente | Mínimo | Recomendado |
|---|---|---|
| RAM | 8 GB | 16 GB |
| Almacenamiento libre | 10 GB | 20 GB |
| CPU | x86_64 / ARM64 / Apple Silicon | Apple M-series o AMD Ryzen 7+ |
| GPU (opcional) | — | Apple Metal / NVIDIA con CUDA 11.8+ |

> El modelo Mistral 7B requiere aproximadamente 4.1 GB de RAM en modo cuantizado (Q4). Con 8 GB de RAM el sistema es funcional; con 16 GB la respuesta es notablemente más rápida.

### Software requerido

| Componente | Versión mínima | Notas |
|---|---|---|
| Python | 3.12 | Requerido exactamente 3.12 o superior |
| Git | 2.x | Para clonar el repositorio |
| Ollama | 0.3+ | Servidor de modelos LLM local |
| uv | 0.4+ | Gestor de entornos y dependencias Python |

---

## 2. Instalación en macOS (Apple Silicon e Intel)

### 2.1 Instalar Homebrew

Si aún no tienes Homebrew instalado:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Verifica la instalación:

```bash
brew --version
```

### 2.2 Instalar Python 3.12 y Git

```bash
brew install python@3.12 git
```

Agrega Python 3.12 al PATH si es necesario (Apple Silicon):

```bash
echo 'export PATH="/opt/homebrew/opt/python@3.12/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

Para Intel Mac:

```bash
echo 'export PATH="/usr/local/opt/python@3.12/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### 2.3 Instalar uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.zshrc
```

Verifica:

```bash
uv --version
```

### 2.4 Instalar Ollama

```bash
brew install ollama
```

Inicia el servicio de Ollama en segundo plano:

```bash
ollama serve &
```

O bien, para que inicie automáticamente al arrancar el sistema:

```bash
brew services start ollama
```

### 2.5 Descargar el modelo Mistral

```bash
ollama pull mistral
```

Este paso descarga aproximadamente 4.1 GB. Requiere conexión a internet solo la primera vez.

### 2.6 Clonar el repositorio y preparar el entorno Python

```bash
cd ~/proyectos
git clone <URL_DEL_REPOSITORIO> agente-credito
cd agente-credito
```

Crea el entorno virtual con uv y Python 3.12:

```bash
uv venv --python 3.12
source .venv/bin/activate
```

Instala las dependencias:

```bash
uv pip install fastapi uvicorn langgraph langchain-core langchain-ollama pydantic python-multipart
```

### 2.7 Iniciar la API

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Accede al frontend en: [http://localhost:8000/static/index.html](http://localhost:8000/static/index.html)

---

## 3. Instalación en Linux (Ubuntu 22.04 / 24.04 y Debian)

### 3.1 Actualizar paquetes del sistema

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y curl git build-essential
```

### 3.2 Instalar Python 3.12

En Ubuntu 22.04 (Python 3.12 no está en los repos por defecto):

```bash
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev
```

En Ubuntu 24.04 y Debian 12+, Python 3.12 está disponible directamente:

```bash
sudo apt install -y python3.12 python3.12-venv python3.12-dev
```

### 3.3 Instalar uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

### 3.4 Instalar Ollama

Usa el script oficial:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

El instalador crea automáticamente el usuario `ollama` y un servicio systemd. Verifica que el servicio esté activo:

```bash
sudo systemctl status ollama
```

Si no está activo:

```bash
sudo systemctl enable ollama
sudo systemctl start ollama
```

### 3.5 Descargar el modelo Mistral

```bash
ollama pull mistral
```

### 3.6 Configurar el entorno Python

```bash
cd ~/proyectos
git clone <URL_DEL_REPOSITORIO> agente-credito
cd agente-credito

uv venv --python python3.12
source .venv/bin/activate

uv pip install fastapi uvicorn langgraph langchain-core langchain-ollama pydantic python-multipart
```

### 3.7 Iniciar la API

Para pruebas:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Para producción con systemd, crea el archivo de servicio:

```bash
sudo nano /etc/systemd/system/agente-credito.service
```

Contenido del servicio:

```ini
[Unit]
Description=Motor de Decisión Crediticia — FastAPI
After=network.target ollama.service
Requires=ollama.service

[Service]
Type=simple
User=tu_usuario
WorkingDirectory=/home/tu_usuario/proyectos/agente-credito
ExecStart=/home/tu_usuario/proyectos/agente-credito/.venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Activa e inicia el servicio:

```bash
sudo systemctl daemon-reload
sudo systemctl enable agente-credito
sudo systemctl start agente-credito
```

---

## 4. Instalación en Windows

### Opción A — WSL2 (recomendada)

WSL2 (Windows Subsystem for Linux) es la ruta recomendada porque ofrece mejor compatibilidad con Ollama y el ecosistema Python.

**Paso 1:** Instala WSL2 desde PowerShell con privilegios de administrador:

```powershell
wsl --install
```

Reinicia el equipo cuando se solicite.

**Paso 2:** Instala Ubuntu 22.04 desde Microsoft Store o con:

```powershell
wsl --install -d Ubuntu-22.04
```

**Paso 3:** Abre la terminal de Ubuntu y sigue exactamente los pasos de la sección 3 (Linux Ubuntu 22.04).

**Paso 4:** Accede al frontend desde el navegador de Windows:

```
http://localhost:8000/static/index.html
```

WSL2 expone automáticamente los puertos al host de Windows.

---

### Opción B — Windows Nativo (sin WSL2)

> Esta opción requiere más configuración manual y tiene menor compatibilidad probada.

**Paso 1 — Instalar Python 3.12:**

Descarga el instalador desde [python.org/downloads](https://www.python.org/downloads/). Durante la instalación, marca la casilla **"Add Python to PATH"**.

Verifica en PowerShell:

```powershell
python --version
```

**Paso 2 — Instalar uv:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Reinicia PowerShell y verifica:

```powershell
uv --version
```

**Paso 3 — Instalar Git:**

Descarga desde [git-scm.com](https://git-scm.com/download/win) e instala con las opciones por defecto.

**Paso 4 — Instalar Ollama:**

Descarga el instalador `.exe` desde [ollama.com/download](https://ollama.com/download). Ollama se instala como servicio de Windows y se inicia automáticamente.

**Paso 5 — Descargar Mistral:**

```powershell
ollama pull mistral
```

**Paso 6 — Clonar y configurar el proyecto:**

```powershell
cd C:\proyectos
git clone <URL_DEL_REPOSITORIO> agente-credito
cd agente-credito

uv venv --python 3.12
.venv\Scripts\activate

uv pip install fastapi uvicorn langgraph langchain-core langchain-ollama pydantic python-multipart
```

**Paso 7 — Iniciar la API:**

```powershell
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

---

## 5. Dependencias Python

Las siguientes son las dependencias directas del sistema:

| Paquete | Versión mínima probada | Función |
|---|---|---|
| `fastapi` | 0.111+ | Framework web y API REST |
| `uvicorn` | 0.29+ | Servidor ASGI para FastAPI |
| `langgraph` | 0.1+ | Orquestación del grafo de agentes |
| `langchain-core` | 0.2+ | Cadenas de prompts y abstracciones LLM |
| `langchain-ollama` | 0.1+ | Integración de Ollama con LangChain |
| `pydantic` | 2.x | Validación y serialización de modelos de datos |
| `python-multipart` | 0.0.9+ | Soporte para `multipart/form-data` en FastAPI |

Instala todas con un solo comando:

```bash
uv pip install fastapi uvicorn langgraph langchain-core langchain-ollama pydantic python-multipart
```

Para fijar versiones exactas en producción, genera un `requirements.txt`:

```bash
uv pip freeze > requirements.txt
```

Y para instalar desde ese archivo en otro entorno:

```bash
uv pip install -r requirements.txt
```

---

## 6. Configuración del modelo Ollama

### Modelo principal: Mistral 7B

```bash
ollama pull mistral
```

- Tamaño en disco: ~4.1 GB (cuantización Q4_0)
- RAM requerida en ejecución: ~5–6 GB
- Tiempo de respuesta típico: 5–15 segundos por análisis (sin GPU)

### Modelo alternativo: Llama 3.2

Si Mistral tiene problemas de rendimiento en tu hardware, puedes usar Llama 3.2 (3B parámetros, más ligero):

```bash
ollama pull llama3.2
```

Para cambiar el modelo, edita la línea de inicialización en `agentes.py`:

```python
# Línea actual:
llm = OllamaLLM(model="mistral", temperature=0.1)

# Alternativa más ligera:
llm = OllamaLLM(model="llama3.2", temperature=0.1)
```

### Aceleración por GPU

**Apple Silicon (Metal):**

Ollama detecta automáticamente el chip Apple M-series y usa Metal para aceleración. No se requiere configuración adicional. El tiempo de respuesta baja a 2–5 segundos por análisis en M2/M3.

Verifica que Ollama reconoce el GPU:

```bash
ollama run mistral "hola"
# Observa los logs: debe mencionar "Metal" o "GPU layers"
```

**NVIDIA (CUDA):**

Ollama detecta automáticamente CUDA si los drivers están instalados. Requiere:
- Driver NVIDIA 525+
- CUDA Toolkit 11.8+

Verifica la detección:

```bash
ollama run mistral "hola"
# Los logs deben mencionar "CUDA" y el nombre de tu GPU
```

Para forzar el número de capas en GPU (útil si la VRAM es limitada):

```bash
OLLAMA_GPU_LAYERS=24 ollama serve
```

**Sin GPU:**

El sistema funciona correctamente en modo CPU. El tiempo de respuesta es mayor (15–45 segundos por análisis completo) pero los resultados son idénticos.

### Verificar que Ollama está corriendo

```bash
curl http://localhost:11434/api/tags
```

Respuesta esperada:

```json
{
  "models": [
    {
      "name": "mistral:latest",
      "size": 4109865280,
      ...
    }
  ]
}
```

---

## 7. Verificación de la instalación

### 7.1 Verificar el health check de la API

Con la API corriendo (`uvicorn api:app --reload`):

```bash
curl http://localhost:8000/health
```

Respuesta esperada:

```json
{
  "status": "ok",
  "timestamp": "2026-03-22T10:30:00.123456"
}
```

### 7.2 Verificar el endpoint raíz

```bash
curl http://localhost:8000/
```

Respuesta esperada:

```json
{
  "status": "ok",
  "version": "2.0.0",
  "mensaje": "Motor de decisión activo"
}
```

### 7.3 Ejecutar los casos de prueba

Asegúrate de estar en el directorio del proyecto con el entorno virtual activado:

```bash
cd /Users/fernandosierra/proyectos/agente-credito
source .venv/bin/activate
python prueba.py
```

Esto ejecuta tres casos de prueba predefinidos:
- **Caso 1** (`47491944`): Lourdes Priscila — caso con alertas HAWK y juicios
- **Caso 2** (`47465130`): Cristel Martinez — happy path, BC Score 662, historial limpio
- **Caso 3** (`47019195`): Roberto García — BC Score sin cuentas (reporte incompleto)

Cada caso imprime el expediente en consola y registra el resultado en `auditoria.jsonl`.

### 7.4 Verificar el frontend

Abre en el navegador:

```
http://localhost:8000/static/index.html
```

Deberías ver el formulario de captura de solicitud con las pestañas: Condiciones, Identificación, Personales, Negocio, Ingresos/Egresos, Buró de Crédito, Resultado.

---

## 8. Solución de problemas comunes

### Error: JSONDecodeError al procesar respuesta del modelo

**Síntoma:** La API devuelve error 500 con mensaje similar a `Expecting value: line 1 column 1`.

**Causa:** El modelo Ollama devolvió texto en formato inesperado (markdown, texto libre) en lugar de JSON puro.

**Solución:**
1. Verifica que Ollama está corriendo: `curl http://localhost:11434/api/tags`
2. Prueba el modelo directamente: `ollama run mistral "responde solo: {\"ok\": true}"`
3. Si el problema persiste, reinicia Ollama: `ollama stop mistral && ollama serve &`
4. El sistema tiene manejo de errores con fallback: en `agente_decision`, si el JSON no se puede parsear, devuelve `ESCALAR_EJECUTIVO` automáticamente.

---

### Error: Ollama no está corriendo

**Síntoma:** `ConnectionRefusedError` o `httpx.ConnectError` al llamar `/analizar`.

**Diagnóstico:**

```bash
curl http://localhost:11434/api/tags
# Si falla, Ollama no está corriendo
```

**Solución en macOS:**

```bash
brew services restart ollama
# o manualmente:
ollama serve &
```

**Solución en Linux:**

```bash
sudo systemctl restart ollama
sudo systemctl status ollama
```

**Solución en Windows:**

Abre el Administrador de Tareas, busca "Ollama" en la lista de aplicaciones y verifica que esté corriendo. Si no, ábrelo desde el menú de inicio.

---

### Error: Puerto 8000 en uso

**Síntoma:** `OSError: [Errno 98] Address already in use` al iniciar uvicorn.

**Diagnóstico:**

```bash
# macOS / Linux
lsof -i :8000

# Windows PowerShell
netstat -ano | findstr :8000
```

**Solución — Cambiar el puerto:**

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8001
```

Recuerda actualizar la URL del frontend si cambias el puerto (no es necesario si sirves el frontend desde la misma API).

**Solución — Terminar el proceso que usa el puerto:**

```bash
# macOS / Linux (reemplaza PID con el número del proceso)
kill -9 <PID>
```

---

### Error: Modelo no encontrado

**Síntoma:** `model 'mistral' not found` en los logs de la API.

**Solución:**

```bash
ollama pull mistral
ollama list  # verifica que "mistral:latest" aparece en la lista
```

Si el modelo está descargado pero sigue sin encontrarse, verifica que Ollama está corriendo en el puerto por defecto (11434):

```bash
OLLAMA_HOST=http://localhost:11434 ollama list
```

---

### Error: ImportError o ModuleNotFoundError

**Síntoma:** `ModuleNotFoundError: No module named 'fastapi'` al iniciar.

**Causa:** El entorno virtual no está activado.

**Solución:**

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

# Verifica que el entorno está activo (debe aparecer "(. venv)" en el prompt)
which python  # debe apuntar a .venv/bin/python
```

---

### Respuestas lentas del modelo

**Síntoma:** El endpoint `/analizar` tarda más de 60 segundos.

**Causas y soluciones:**
- En modo CPU con poca RAM: normal en hardware con 8 GB. Considera llama3.2 como modelo alternativo.
- Ollama usando swap: cierra otras aplicaciones que consuman RAM.
- Apple Silicon sin Metal: verifica que Ollama detecta el GPU con `ollama run mistral "hola"`.
- Linux sin CUDA: instala los drivers NVIDIA actualizados y el toolkit CUDA.

---

### El frontend no carga estilos o muestra errores de CORS

**Síntoma:** El frontend en `index.html` no se ve correctamente o la consola del navegador muestra errores CORS.

**Causa:** Acceder al archivo `index.html` directamente desde el sistema de archivos (`file://`) en lugar de desde el servidor.

**Solución:** Siempre accede a través del servidor FastAPI:

```
# Correcto:
http://localhost:8000/static/index.html

# Incorrecto (no usar):
file:///Users/.../static/index.html
```
