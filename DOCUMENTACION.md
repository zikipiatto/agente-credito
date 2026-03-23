# Documentación Técnica — Motor de Decisión Crediticia

**Versión:** 2.0.0
**Institución:** SOFIPO (Sociedad Financiera Popular)
**Entorno:** Local / Demo — sin conexión a servicios externos

---

## Índice

1. [Descripción general del sistema](#1-descripción-general-del-sistema)
2. [Arquitectura local](#2-arquitectura-local)
3. [Estructura de archivos del proyecto](#3-estructura-de-archivos-del-proyecto)
4. [Modelos de datos](#4-modelos-de-datos)
5. [Sistema multi-agente](#5-sistema-multi-agente)
6. [API REST](#6-api-rest)
7. [Frontend](#7-frontend)
8. [Reglas de negocio consolidadas](#8-reglas-de-negocio-consolidadas)
9. [Auditoría y trazabilidad](#9-auditoría-y-trazabilidad)
10. [Limitaciones y consideraciones regulatorias](#10-limitaciones-y-consideraciones-regulatorias)

---

## 1. Descripción general del sistema

El **Motor de Decisión Crediticia** es un sistema de apoyo a la toma de decisiones para el otorgamiento de crédito en una Sociedad Financiera Popular (SOFIPO) regulada por la CNBV. El sistema automatiza el análisis de solicitudes de crédito mediante una cadena de agentes especializados que evalúan la identidad del solicitante, su capacidad de pago y su historial crediticio en buró, para emitir una recomendación estructurada al ejecutivo de crédito.

### Propósito y alcance

- Analizar solicitudes de crédito productivo y de consumo para personas físicas con actividad empresarial.
- Estandarizar y documentar el proceso de análisis crediticio, reduciendo el tiempo de revisión y la variabilidad en los criterios de decisión.
- Generar un expediente de decisión auditable que cumple con los requerimientos de trazabilidad de la CNBV.
- Operar completamente en infraestructura local sin enviar datos personales a servicios externos en la nube.

### Contexto regulatorio

| Marco | Relevancia para el sistema |
|---|---|
| **CNBV** — Ley de Ahorro y Crédito Popular (LACP) | El sistema es una herramienta de apoyo. La decisión final debe ser validada y firmada por un ejecutivo humano. |
| **SOFIPO** | Institución de tecnología financiera regulada, sujeta a límites de exposición por acreditado y a reglas de diversificación de cartera. |
| **PLD** (Prevención de Lavado de Dinero) | El agente KYC verifica alertas de identidad. Las alertas HAWK del buró son evaluadas. Se requiere integración con listas OFAC/SAT en ambientes productivos. |
| **LFPDPPP** | Los datos personales del solicitante (RFC, CURP, nombre, biométricos) se procesan localmente y no se transmiten a terceros. |
| **Buró de Crédito** | El sistema recibe e interpreta el reporte de buró previamente consultado por la institución. No consulta directamente buró. |

> **Nota importante:** Este sistema es una herramienta de análisis y apoyo. Ninguna decisión del sistema tiene efecto legal por sí sola. Todo resultado debe ser revisado, validado y aprobado por el ejecutivo de crédito responsable.

---

## 2. Arquitectura local

```
┌─────────────────────────────────────────────────────────────────────┐
│                        NAVEGADOR WEB                                │
│                   http://localhost:8000/static/                     │
│           (index.html — Single Page Application)                    │
└─────────────────────────┬───────────────────────────────────────────┘
                          │  POST /analizar (JSON)
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FASTAPI (api.py)                                │
│                    uvicorn — puerto 8000                            │
│   GET /health   GET /   POST /analizar   GET /static/*             │
└─────────────────────────┬───────────────────────────────────────────┘
                          │  analizar(SolicitudCompleta)
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   LANGGRAPH (agentes.py)                            │
│              StateGraph — EstadoSolicitud compartido                │
│                                                                     │
│  ┌──────────┐   ┌─────────────┐   ┌───────────┐                   │
│  │ agente   │──▶│  agente     │──▶│  agente   │                   │
│  │  kyc     │   │  financiero │   │   buro    │                   │
│  │(reglas)  │   │  (reglas)   │   │ (reglas)  │                   │
│  └──────────┘   └─────────────┘   └─────┬─────┘                   │
│                                         │                          │
│  ┌──────────┐   ┌─────────────┐   ┌─────▼─────┐                   │
│  │ agente   │◀──│  agente     │◀──│  agente   │                   │
│  │expediente│   │  decision   │   │analisis   │                   │
│  │(reglas)  │   │   (LLM)     │   │  _buro    │                   │
│  └──────────┘   └─────────────┘   │(hybrid)   │                   │
│        │               │          └─────┬─────┘                   │
└────────┼───────────────┼────────────────┼────────────────────────┘
         │               │  LLM calls     │
         │               ▼                ▼
         │   ┌─────────────────────────────────┐
         │   │       OLLAMA (puerto 11434)      │
         │   │       Modelo: Mistral 7B         │
         │   │    (local, sin internet)         │
         │   └─────────────────────────────────┘
         │
         ▼
  auditoria.jsonl  (registro de auditoría local)
```

### Flujo de datos

1. El ejecutivo captura la solicitud en el frontend (7 pestañas de formulario).
2. Al enviar, el frontend hace `POST /analizar` con el JSON completo de la solicitud.
3. FastAPI valida el JSON contra el modelo Pydantic `SolicitudCompleta`.
4. Se ejecuta el grafo LangGraph: 6 agentes en secuencia, cada uno lee y escribe en el estado compartido.
5. Los agentes `analisis_buro` y `decision` invocan el modelo Mistral vía Ollama.
6. FastAPI construye la respuesta estructurada y la devuelve al frontend.
7. El frontend renderiza el resultado en la pestaña "Resultado".

---

## 3. Estructura de archivos del proyecto

```
agente-credito/
│
├── api.py                  # Servidor FastAPI: endpoints, CORS, montaje de estáticos
├── agentes.py              # Sistema multi-agente LangGraph: los 6 agentes y el grafo
├── modelos.py              # Modelos Pydantic: todos los tipos de datos de entrada
├── prueba.py               # Tres casos de prueba con datos representativos
│
├── static/
│   └── index.html          # Frontend SPA: formulario de captura y visualización
│
├── auditoria.jsonl         # Registro de auditoría (generado en ejecución)
│
├── Ejemplos Solicitud y Buro/
│   └── ...                 # Ejemplos de JSON para pruebas manuales
│
├── INSTALACION.md          # Esta guía de instalación por sistema operativo
├── DOCUMENTACION.md        # Documentación técnica completa del sistema
└── ARQUITECTURA_AWS.md     # Plan de migración y arquitectura cloud objetivo
```

| Archivo | Responsabilidad |
|---|---|
| `api.py` | Punto de entrada HTTP. Valida entrada, invoca el grafo, construye y devuelve la respuesta estructurada. |
| `agentes.py` | Toda la lógica de negocio del análisis crediticio. Define `EstadoSolicitud`, los 6 nodos del grafo, y la función `construir_grafo()`. |
| `modelos.py` | Contratos de datos. Define los 8 modelos Pydantic que estructuran la entrada de la solicitud. |
| `prueba.py` | Casos de prueba ejecutables directamente con Python, sin necesidad de la API REST. Escribe en `auditoria.jsonl`. |
| `static/index.html` | Interfaz de usuario de página única. No tiene dependencias externas (sin npm, sin framework). |
| `auditoria.jsonl` | Registro append-only de cada solicitud analizada. Cada línea es un objeto JSON con la decisión completa. |

---

## 4. Modelos de datos

Todos los modelos se definen en `modelos.py` usando Pydantic v2. Son los contratos de datos para la entrada del sistema.

### 4.1 `Condiciones`

Parámetros financieros del producto crediticio solicitado.

| Campo | Tipo | Descripción |
|---|---|---|
| `numero_solicitud` | `str` | Folio único de la solicitud (ej: `"47491944"`) |
| `fecha` | `str` | Fecha de la solicitud en formato `DD/MM/AAAA` |
| `monto` | `float` | Monto total del crédito solicitado en MXN |
| `tipo` | `str` | Tipo de crédito: `"PRODUCTIVO"` o `"CONSUMO"` |
| `producto` | `str` | Nombre del producto (ej: `"Capital de Trabajo"`) |
| `tasa` | `float` | Tasa de interés anual en porcentaje (ej: `131.0` para 131% anual) |
| `cuota` | `float` | Monto de cada pago periódico en MXN |
| `frecuencia` | `str` | Frecuencia de pago (ej: `"MENSUAL DE 28 DÍAS"`) |
| `plazo` | `int` | Número total de pagos |

### 4.2 `Identificacion`

Resultados de la verificación biométrica digital del solicitante. Todos los scores son valores numéricos en escala 0–100.

| Campo | Tipo | Descripción |
|---|---|---|
| `verificacion_id` | `float` | Score de verificación del documento de identidad (INE/pasaporte) |
| `reconocimiento_facial` | `float` | Score de coincidencia entre la selfie y el documento |
| `deteccion_vida` | `float` | Score de prueba de vida (liveness detection) — detecta suplantación |
| `validacion_gobierno` | `float` | Score de validación contra registros del gobierno (RENAPO/INE). `0` = no consultado |
| `video_selfie` | `float` | Score de calidad del video selfie. `0` = no capturado |
| `distancia_km` | `float` | Distancia en kilómetros entre la geolocalización del dispositivo y el domicilio declarado |

### 4.3 `DatosCliente`

Información personal del solicitante.

| Campo | Tipo | Descripción |
|---|---|---|
| `nombre` | `str` | Nombre completo en mayúsculas |
| `rfc` | `str` | RFC del solicitante (mínimo 12 caracteres: personas físicas 13, morales 12) |
| `curp` | `str` | CURP del solicitante (exactamente 18 caracteres) |
| `fecha_nacimiento` | `str` | Fecha de nacimiento en formato `DD/MM/AAAA` |
| `genero` | `str` | Género declarado: `"MASCULINO"` o `"FEMENINO"` |
| `estado_civil` | `str` | Estado civil (ej: `"SOLTERO(A)"`, `"CASADO"`) |
| `tipo_vivienda` | `str` | Tipo de vivienda: `"PROPIA"`, `"FAMILIAR"`, `"RENTADA"` |
| `dependientes_economicos` | `int` | Número de personas que dependen económicamente del solicitante |
| `antiguedad_domicilio_anios` | `int` | Años de residencia en el domicilio actual |

### 4.4 `Negocio`

Información del negocio o actividad económica del solicitante.

| Campo | Tipo | Descripción |
|---|---|---|
| `nombre` | `str` | Nombre del negocio o actividad |
| `giro` | `str` | Giro de negocio: `"COMERCIO"`, `"SERVICIOS"`, `"PRODUCCIÓN"`, etc. |
| `antiguedad_fecha` | `str` | Fecha desde que opera el negocio en formato `DD/MM/AAAA` |

### 4.5 `IngresosEgresos`

Flujo de efectivo mensual declarado.

| Campo | Tipo | Descripción |
|---|---|---|
| `total_ingresos` | `float` | Total de ingresos mensuales en MXN (ventas, salarios, rentas, etc.) |
| `total_egresos` | `float` | Total de egresos mensuales en MXN (gastos operativos, gastos del hogar, otras deudas) |

### 4.6 `CuentaBuro`

Detalle de una cuenta individual del reporte de Buró de Crédito. El historial de pagos usa la escala MOP (Manner of Payment) estándar de la industria.

| Campo | Tipo | Default | Descripción |
|---|---|---|---|
| `numero` | `int` | — | Número de cuenta dentro del reporte |
| `tipo_credito` | `str` | — | Tipo: `"PRÉSTAMO PERSONAL"`, `"TARJETA DE CRÉDITO"`, `"HIPOTECARIO"`, etc. |
| `otorgante` | `str` | — | Nombre del otorgante: `"BANCO"`, `"MICROFINANCIERA"`, `"SOFOM"`, etc. |
| `estado` | `str` | — | Estado de la cuenta: `"ABIERTA"` o `"CERRADA"` |
| `fecha_apertura` | `Optional[str]` | `None` | Fecha de apertura de la cuenta |
| `fecha_cierre` | `Optional[str]` | `None` | Fecha de cierre (solo si estado = `"CERRADA"`) |
| `credito_maximo` | `float` | `0.0` | Monto máximo autorizado de la línea de crédito en MXN |
| `saldo_actual` | `float` | `0.0` | Saldo vigente al momento del reporte en MXN |
| `saldo_vencido` | `float` | `0.0` | Saldo vencido (en mora) en MXN |
| `mop_actual` | `int` | `0` | MOP (Manner of Payment) actual: `0`=sin info, `1`=puntual, `2–9`=días de atraso |
| `peor_mop` | `int` | `0` | Peor MOP registrado en el historial de la cuenta |
| `pagos_puntuales` | `int` | `0` | Conteo histórico de pagos con MOP=1 |
| `pagos_atrasados` | `int` | `0` | Conteo histórico de pagos con MOP>1 |

**Escala MOP:**

| Valor | Significado |
|---|---|
| 0 | Sin información |
| 1 | Puntual (al corriente) |
| 2 | 1–29 días de atraso |
| 3 | 30–59 días de atraso |
| 4 | 60–89 días de atraso |
| 5 | 90–119 días de atraso |
| 6 | 120–149 días de atraso |
| 7 | Cartera vencida / 150+ días |
| 9 | Deuda desconocida o en litigio |

### 4.7 `Buro`

Resumen del reporte de Buró de Crédito del solicitante.

| Campo | Tipo | Default | Descripción |
|---|---|---|---|
| `score` | `Optional[float]` | `None` | Score crediticio numérico. Puede ser `-8` o `-9` (códigos de exclusión) |
| `icc` | `Optional[str]` | `None` | Código ICC del reporte (ej: `"0005"`) — indica el tipo de consulta |
| `tipo_score` | `Optional[str]` | `None` | Nombre del modelo de score: `"BC SCORE"`, `"FICO"`, etc. |
| `causas_score` | `list[str]` | `[]` | Razones textuales que explican el valor del score |
| `etiqueta_riesgo` | `Optional[str]` | `None` | Etiqueta cualitativa de riesgo del reporte |
| `alertas_hawk` | `list[str]` | `[]` | Alertas HAWK del buró (inconsistencias de identidad, posibles fraudes) |
| `tiene_juicios` | `bool` | `False` | Indica si existen juicios registrados en buró |
| `creditos_activos` | `int` | `0` | Número de cuentas abiertas en el reporte |
| `creditos_cerrados` | `int` | `0` | Número de cuentas cerradas en el reporte |
| `creditos_vencidos` | `int` | `0` | Número de cuentas con saldo vencido |
| `peor_atraso_dias` | `int` | `0` | Peor atraso histórico en días (campo independiente del reporte) |
| `saldo_actual` | `float` | `0.0` | Saldo total vigente en todas las cuentas |
| `saldo_vencido` | `float` | `0.0` | Saldo total vencido en todas las cuentas |
| `cuentas` | `list[CuentaBuro]` | `[]` | Lista del detalle de cuentas individuales |

### 4.8 `SolicitudCompleta`

Modelo raíz que agrupa todos los datos de una solicitud. Este es el tipo que recibe el endpoint `POST /analizar`.

| Campo | Tipo | Descripción |
|---|---|---|
| `condiciones` | `Condiciones` | Parámetros del producto crediticio |
| `identificacion` | `Identificacion` | Resultados biométricos |
| `cliente` | `DatosCliente` | Datos personales del solicitante |
| `negocio` | `Negocio` | Información del negocio |
| `finanzas` | `IngresosEgresos` | Flujo de efectivo mensual |
| `buro` | `Buro` | Reporte de Buró de Crédito |

---

## 5. Sistema multi-agente

### 5.1 Grafo de estado LangGraph

El sistema utiliza LangGraph para orquestar los agentes como un **grafo de estado dirigido acíclico (DAG)**. LangGraph garantiza que:

- El estado (`EstadoSolicitud`) se pasa entre nodos por referencia.
- Cada nodo solo puede ejecutarse cuando el nodo anterior ha completado.
- El grafo se compila una vez (`construir_grafo()`) y se puede invocar múltiples veces.

### 5.2 `EstadoSolicitud` — Estado compartido

```python
class EstadoSolicitud(TypedDict):
    solicitud:          dict   # SolicitudCompleta serializada como dict
    resultado_kyc:      dict   # Salida del agente_kyc
    resultado_financiero: dict # Salida del agente_financiero
    resultado_buro:     dict   # Salida del agente_buro
    analisis_buro:      dict   # Salida del agente_analisis_buro
    decision_final:     dict   # Salida del agente_decision
    expediente:         str    # Texto del expediente generado
```

El estado se inicializa con la solicitud y los demás campos vacíos. Cada agente lee del estado y escribe su resultado en el campo correspondiente.

### 5.3 Diagrama del flujo de agentes

```
 ENTRADA
 SolicitudCompleta
       │
       ▼
 ┌─────────────┐
 │  agente_kyc │  Agente 1
 │  (reglas)   │  Biométricos + RFC + CURP
 └──────┬──────┘
        │  resultado_kyc → {aprobado, score_biometrico, alertas}
        ▼
 ┌─────────────────┐
 │agente_financiero│  Agente 2
 │   (reglas)      │  Ingreso neto + ratio cuota/ingreso
 └────────┬────────┘
          │  resultado_financiero → {aprobado, ingreso_neto, ratio_cuota_ingreso, ...}
          ▼
 ┌──────────────┐
 │  agente_buro │  Agente 3
 │   (reglas)   │  Score, juicios, vencidos, atraso
 └──────┬───────┘
        │  resultado_buro → {aprobado, score, alertas, reporte_incompleto, ...}
        ▼
 ┌────────────────────┐
 │agente_analisis_buro│  Agente 3B
 │  (reglas + LLM)    │  Métricas cuantitativas + análisis Mistral
 └──────────┬─────────┘
            │  analisis_buro → {métricas + comportamiento_pago + resumen_ejecutivo, ...}
            ▼
 ┌──────────────────┐
 │  agente_decision │  Agente 4
 │      (LLM)       │  Decisión final vía Mistral
 └────────┬─────────┘
          │  decision_final → {decision, score_riesgo, razon_principal, condiciones}
          ▼
 ┌────────────────────┐
 │ agente_expediente  │  Agente 5
 │    (reglas)        │  Genera texto del expediente formateado
 └────────┬───────────┘
          │  expediente → str (texto formateado con todos los resultados)
          ▼
       SALIDA
    EstadoSolicitud
    (completo)
```

### 5.4 Agente 1: `agente_kyc`

**Propósito:** Verificar la identidad del solicitante mediante análisis de los resultados biométricos del proceso de onboarding digital.

**Tipo:** Basado en reglas (determinístico).

**Entradas del estado:** `solicitud.identificacion`, `solicitud.cliente`

**Lógica de procesamiento:**

1. Evalúa los tres scores biométricos críticos (`verificacion_id`, `reconocimiento_facial`, `deteccion_vida`):
   - Score < 50: Genera alerta de rechazo y establece `aprobado = False`
   - Score 50–69: Genera alerta informativa (no rechaza)
   - Excepción: `deteccion_vida` < 80 ya genera alerta (umbral más estricto)
2. Valida `validacion_gobierno` y `video_selfie`: si ambos son `0`, genera alerta informativa.
3. Valida formato de RFC (mínimo 12 caracteres) y CURP (exactamente 18 caracteres).
4. Calcula `score_biometrico` como promedio de los tres scores críticos.

**Salida:**

```json
{
  "aprobado": true,
  "score_biometrico": 96.0,
  "alertas": []
}
```

**Reglas de rechazo:** Score biométrico crítico < 50 en cualquiera de los tres campos.

---

### 5.5 Agente 2: `agente_financiero`

**Propósito:** Evaluar la capacidad de pago del solicitante comparando sus ingresos y egresos con la cuota propuesta.

**Tipo:** Basado en reglas (determinístico).

**Entradas del estado:** `solicitud.finanzas`, `solicitud.condiciones`

**Lógica de procesamiento:**

1. Calcula métricas financieras fundamentales:
   - `ingreso_neto = total_ingresos - total_egresos`
   - `ratio_cuota_ingreso = (cuota / total_ingresos) × 100`
   - `capacidad_pago = ingreso_neto - cuota`
   - `ratio_monto_ingreso = monto / total_ingresos`
2. Evalúa condiciones de rechazo y alerta.

**Salida:**

```json
{
  "aprobado": true,
  "ingreso_neto": 27000.0,
  "ratio_cuota_ingreso": 4.8,
  "capacidad_pago": 24857.14,
  "ratio_monto_ingreso": 0.33,
  "alertas": []
}
```

**Reglas de rechazo:** `ratio_cuota_ingreso > 60%` o `capacidad_pago < 0` o `total_ingresos <= 0`.

---

### 5.6 Agente 3: `agente_buro`

**Propósito:** Analizar el reporte de Buró de Crédito para identificar indicadores de riesgo crediticio.

**Tipo:** Basado en reglas (determinístico).

**Entradas del estado:** `solicitud.buro`

**Lógica de procesamiento:**

1. **Códigos de exclusión BC Score:** Si `score` es `-8` o `-9`, no rechaza. Son códigos que indican "sin historial registrado" — situación normal en SOFIPO para primer crédito. Solo genera alerta informativa.
2. **Reporte incompleto:** Si el tipo es "BC SCORE" y el score existe pero no hay cuentas reportadas (`creditos_activos + creditos_cerrados = 0`), el reporte es inconsistente. Rechaza y escala para consulta manual.
3. **Juicios:** Si `tiene_juicios = True`, rechaza automáticamente.
4. **Créditos vencidos:** Si `creditos_vencidos > 0`, rechaza.
5. **Atraso grave:** Si `peor_atraso_dias > 90`, rechaza.
6. **Saldo vencido:** Si `saldo_vencido > 0`, rechaza.
7. **Alertas HAWK:** Se registran como alertas informativas (no rechazan por sí solas).
8. **Evaluación del score numérico:**
   - Score < 400: Rechaza por alto riesgo.
   - Score 400–549: Alerta, escala a ejecutivo (no rechaza automáticamente).
   - Score >= 550: Sin alerta por score.

**Salida:**

```json
{
  "aprobado": true,
  "score": 662,
  "icc": "0005",
  "tipo_score": "BC SCORE",
  "causas_score": ["Última cuenta nueva aperturada recientemente"],
  "total_creditos": 35,
  "creditos_activos": 2,
  "creditos_cerrados": 33,
  "reporte_incompleto": false,
  "alertas": ["Alertas HAWK: TELEFONO NO CORRESPONDE A ZONA POSTAL"]
}
```

---

### 5.7 Agente 3B: `agente_analisis_buro`

**Propósito:** Calcular métricas cuantitativas del historial crediticio y generar una interpretación cualitativa usando el modelo de lenguaje Mistral.

**Tipo:** Híbrido — reglas para métricas, LLM para interpretación.

**Entradas del estado:** `solicitud.buro` (campo `cuentas`), `resultado_buro`

**Lógica de procesamiento:**

**Fase 1 — Métricas calculadas (reglas):**

| Métrica | Cálculo |
|---|---|
| `utilizacion_pct` | `(saldo_actual_abiertas / credito_maximo_abiertas) × 100` |
| `tasa_puntualidad_pct` | `(pagos_puntuales / total_pagos) × 100` |
| `cuentas_con_atraso` | Conteo de cuentas con `peor_mop > 1` |
| `peor_mop_historico` | Máximo `peor_mop` entre todas las cuentas |
| `desglose_otorgante` | Agregación por otorgante: abiertas, cerradas, saldo, máximo |
| `detalle_parcial` | `True` si `total_reportado > len(cuentas)` (el JSON no incluye todas las cuentas del reporte) |

**Fase 2 — Análisis LLM (Mistral vía Ollama):**

El agente construye un resumen textual del historial y lo envía al modelo con el siguiente prompt estructurado, solicitando respuesta en JSON:

```
Campos solicitados al modelo:
- comportamiento_pago: descripción cualitativa
- nivel_endeudamiento: "bajo" | "medio" | "alto"
- experiencia_crediticia: descripción
- tendencia_reciente: descripción
- areas_oportunidad: lista de puntos
- fortalezas: lista de puntos
- recomendacion_monto: texto con recomendación
- resumen_ejecutivo: párrafo de síntesis
```

Si no hay cuentas en el reporte, el agente devuelve valores por defecto sin invocar el LLM.

**Salida:** Combinación de métricas calculadas + campos del análisis LLM.

---

### 5.8 Agente 4: `agente_decision`

**Propósito:** Emitir la decisión crediticia final integrando los resultados de todos los agentes anteriores.

**Tipo:** LLM (Mistral vía Ollama).

**Entradas del estado:** `resultado_kyc`, `resultado_financiero`, `resultado_buro`, `analisis_buro`, `solicitud.condiciones`, `solicitud.negocio`, `solicitud.cliente`

**Lógica de procesamiento:**

El agente construye un resumen completo de la solicitud (condiciones del crédito, datos del cliente, resultados de los tres agentes anteriores y el análisis de buró) y lo envía al modelo Mistral con instrucciones explícitas sobre las reglas de decisión.

**Reglas de decisión instruidas al modelo:**

| Resultado | Condición |
|---|---|
| `RECHAZADO` | Juicios, créditos vencidos, score < 400, reporte incompleto |
| `ESCALAR_EJECUTIVO` | Alertas menores, casos borderline, datos faltantes, score 400–549 |
| `APROBADO` | KYC, finanzas y buró todos aprobados sin alertas críticas |

**Fallback:** Si el modelo no devuelve JSON válido, el sistema devuelve `ESCALAR_EJECUTIVO` automáticamente.

**Salida:**

```json
{
  "decision": "APROBADO",
  "score_riesgo": 3,
  "razon_principal": "Solicitante con historial crediticio sólido, 35 créditos con alta puntualidad, buena capacidad de pago.",
  "condiciones": "Monto aprobado hasta $15,000 MXN. Plazo máximo 7 pagos. Requiere aval.",
  "recomendacion_ejecutivo": null
}
```

---

### 5.9 Agente 5: `agente_expediente`

**Propósito:** Generar el expediente de decisión en formato de texto estructurado, listo para imprimir o mostrar al ejecutivo.

**Tipo:** Basado en reglas (determinístico).

**Entradas del estado:** Todo el estado completo.

**Salida:** Texto formateado con secciones:
- Encabezado con fecha/hora de generación
- Datos de la solicitud y condiciones del crédito
- Resultados por agente (KYC, Financiero, Buró)
- Métricas de buró (utilización, puntualidad, desglose por otorgante)
- Tabla de detalle de cuentas individuales
- Análisis profundo del buró (interpretación LLM)
- Decisión final con score de riesgo y condiciones
- Leyenda de validación obligatoria del ejecutivo

---

## 6. API REST

La API está implementada en `api.py` con FastAPI. Todos los endpoints devuelven JSON con `Content-Type: application/json`.

### 6.1 `GET /`

Endpoint de estado del sistema.

**Respuesta:**

```json
{
  "status": "ok",
  "version": "2.0.0",
  "mensaje": "Motor de decisión activo"
}
```

### 6.2 `GET /health`

Health check para monitoreo. Útil para load balancers y scripts de verificación.

**Respuesta:**

```json
{
  "status": "ok",
  "timestamp": "2026-03-22T10:30:00.123456"
}
```

### 6.3 `POST /analizar`

Endpoint principal. Recibe una solicitud crediticia completa y devuelve el resultado del análisis multi-agente.

**Content-Type de la solicitud:** `application/json`

**Esquema de la solicitud:** `SolicitudCompleta` (ver sección 4.8)

**Tiempo de respuesta típico:** 15–60 segundos (depende del hardware y del modelo).

**Esquema de respuesta:**

```json
{
  "status": "ok",
  "timestamp": "2026-03-22T10:30:00.123456",

  "solicitud": {
    "numero": "47465130",
    "fecha": "20/03/2026",
    "cliente": "CRISTEL MARTINEZ MARQUEZ",
    "rfc": "MAMC8203097X2",
    "producto": "Capital de Trabajo",
    "tipo": "PRODUCTIVO",
    "monto": 15000.0,
    "plazo": 7,
    "cuota": 2142.86,
    "tasa": 131.0,
    "frecuencia": "MENSUAL DE 28 DÍAS"
  },

  "kyc": {
    "aprobado": true,
    "score_biometrico": 98.3,
    "alertas": []
  },

  "financiero": {
    "aprobado": true,
    "ingreso_mensual": 45000.0,
    "egreso_mensual": 18000.0,
    "ingreso_neto": 27000.0,
    "cuota_sobre_ingreso_pct": 4.8,
    "capacidad_pago": 24857.14,
    "alertas": []
  },

  "buro": {
    "aprobado": true,
    "score": 662.0,
    "icc": "0005",
    "tipo_score": "BC SCORE",
    "causas_score": ["Última cuenta nueva aperturada recientemente"],
    "reporte_incompleto": false,
    "total_cuentas": 35,
    "cuentas_abiertas": 2,
    "cuentas_cerradas": 33,
    "alertas": ["Alertas HAWK: TELEFONO NO CORRESPONDE A ZONA POSTAL"]
  },

  "analisis_buro": {
    "metricas": {
      "utilizacion_credito_pct": 74.2,
      "tasa_puntualidad_pct": 97.4,
      "pagos_puntuales": 67,
      "pagos_atrasados": 1,
      "peor_mop_historico": 2,
      "cuentas_con_atraso": 1,
      "saldo_actual_abiertas": 103465.0,
      "credito_maximo_abiertas": 139882.0,
      "credito_maximo_historico": 449253.0,
      "saldo_vencido": 0.0,
      "otorgantes": ["BANCO", "MICROFINANCIERA"]
    },
    "interpretacion": {
      "comportamiento_pago": "Historial de pago muy sólido con 97.4% de puntualidad en 35 créditos.",
      "nivel_endeudamiento": "medio",
      "experiencia_crediticia": "Amplia experiencia con múltiples tipos de crédito a lo largo de varios años.",
      "tendencia_reciente": "Comportamiento reciente positivo, sin atrasos en cuentas activas.",
      "fortalezas": ["Alta tasa de puntualidad", "Amplio historial crediticio"],
      "areas_oportunidad": ["Reducir utilización de crédito activo"],
      "recomendacion_monto": "Monto solicitado de $15,000 es conservador respecto al historial.",
      "resumen_ejecutivo": "Solicitante con perfil crediticio sólido. Recomendable para aprobación."
    },
    "detalle_cuentas": [...]
  },

  "decision": {
    "resultado": "APROBADO",
    "score_riesgo": 2,
    "razon_principal": "Historial crediticio excelente y capacidad de pago holgada.",
    "condiciones": "Aprobar $15,000 MXN a 7 pagos mensuales. Sin garantía adicional requerida.",
    "recomendacion_ejecutivo": null
  },

  "expediente_texto": "╔══════════════ EXPEDIENTE ══════..."
}
```

**Manejo de errores:**

| Código HTTP | Causa |
|---|---|
| `422 Unprocessable Entity` | JSON de solicitud no válido según el esquema Pydantic |
| `500 Internal Server Error` | Error en el procesamiento (Ollama no disponible, etc.) |

### 6.4 Ejemplo de invocación con cURL

```bash
curl -X POST http://localhost:8000/analizar \
  -H "Content-Type: application/json" \
  -d '{
    "condiciones": {
      "numero_solicitud": "TEST001",
      "fecha": "22/03/2026",
      "monto": 10000.00,
      "tipo": "PRODUCTIVO",
      "producto": "Capital de Trabajo",
      "tasa": 131.0,
      "cuota": 1666.67,
      "frecuencia": "MENSUAL DE 28 DÍAS",
      "plazo": 6
    },
    "identificacion": {
      "verificacion_id": 95,
      "reconocimiento_facial": 88,
      "deteccion_vida": 97,
      "validacion_gobierno": 100,
      "video_selfie": 85,
      "distancia_km": 0.5
    },
    "cliente": {
      "nombre": "JUAN PEREZ GARCIA",
      "rfc": "PEGJ850312AB3",
      "curp": "PEGJ850312HDFRCN02",
      "fecha_nacimiento": "12/03/1985",
      "genero": "MASCULINO",
      "estado_civil": "CASADO",
      "tipo_vivienda": "PROPIA",
      "dependientes_economicos": 2,
      "antiguedad_domicilio_anios": 5
    },
    "negocio": {
      "nombre": "TAQUERIA EL BUEN SABOR",
      "giro": "COMERCIO",
      "antiguedad_fecha": "01/01/2018"
    },
    "finanzas": {
      "total_ingresos": 25000,
      "total_egresos": 10000
    },
    "buro": {
      "score": 620,
      "tipo_score": "BC SCORE",
      "causas_score": [],
      "alertas_hawk": [],
      "tiene_juicios": false,
      "creditos_activos": 1,
      "creditos_cerrados": 5,
      "creditos_vencidos": 0,
      "peor_atraso_dias": 0,
      "saldo_actual": 5000,
      "saldo_vencido": 0,
      "cuentas": []
    }
  }'
```

---

## 7. Frontend

El frontend es una **Single Page Application (SPA)** implementada en un solo archivo HTML (`static/index.html`) sin dependencias externas (sin npm, sin frameworks, sin CDN). Funciona completamente offline una vez que el servidor está corriendo.

### 7.1 Estructura de pestañas

| # | Pestaña | Sección del formulario | Modelo Pydantic |
|---|---|---|---|
| 0 | **Condiciones** | Número de solicitud, monto, producto, tasa, cuota, plazo, frecuencia, tipo | `Condiciones` |
| 1 | **Identificación** | Scores biométricos (verificación ID, reconocimiento facial, detección de vida, validación gobierno, video selfie), distancia km | `Identificacion` |
| 2 | **Personales** | Nombre, RFC, CURP, fecha nacimiento, género, estado civil, tipo vivienda, dependientes, antigüedad domicilio | `DatosCliente` |
| 3 | **Negocio** | Nombre del negocio, giro, fecha de inicio de operaciones | `Negocio` |
| 4 | **Ingresos/Egresos** | Total ingresos mensuales, total egresos mensuales | `IngresosEgresos` |
| 5 | **Buró de Crédito** | Score, tipo, ICC, causas, alertas HAWK, juicios, cuentas vencidas, saldo vencido, tabla de cuentas individuales | `Buro` |
| 6 | **Resultado** | Visualización de la respuesta completa del análisis | — (solo lectura) |

### 7.2 Lógica de validación del formulario

El frontend realiza validaciones básicas antes de enviar:
- Los campos de score biométrico cambian de color según el valor: verde (>=70), naranja (50–69), rojo (<50).
- Los campos numéricos de montos aceptan valores decimales.
- La pestaña activa se marca visualmente. Las pestañas completadas muestran indicador verde.
- El botón "Analizar Solicitud" está habilitado desde cualquier pestaña del formulario.

### 7.3 Lógica de renderizado de resultados

Al recibir la respuesta del servidor, el frontend renderiza en la pestaña **Resultado**:

1. **Badge de decisión:** Muestra `APROBADO` (verde), `RECHAZADO` (rojo) o `ESCALAR_EJECUTIVO` (naranja) con estilos visuales diferenciados.

2. **Cards de resumen por agente:** Tres tarjetas lado a lado mostrando el resultado de KYC, Financiero y Buró, con indicador verde/rojo según `aprobado` y las alertas de cada uno.

3. **Métricas de buró:** Grid de 4 columnas con las métricas calculadas: utilización de crédito, tasa de puntualidad, saldo vencido, peor MOP histórico.

4. **Tabla de cuentas:** Tabla HTML con el detalle de cada `CuentaBuro`. Las celdas MOP se colorean: verde (MOP=1), naranja (MOP=2), rojo (MOP>=3).

5. **Tags de fortalezas y áreas de oportunidad:** Etiquetas visuales con los resultados del análisis LLM.

6. **Expediente texto:** Área de texto monoespaciada (estilo terminal oscuro) con el expediente completo generado por `agente_expediente`. Permite scroll y puede ser copiado.

---

## 8. Reglas de negocio consolidadas

La siguiente tabla resume todas las reglas de negocio del sistema. Las reglas se aplican en el agente indicado.

| Variable | Umbral de alerta | Umbral de rechazo | Agente responsable |
|---|---|---|---|
| `verificacion_id` | < 70 | < 50 | `agente_kyc` |
| `reconocimiento_facial` | < 70 | < 50 | `agente_kyc` |
| `deteccion_vida` | < 80 | < 50 | `agente_kyc` |
| RFC formato | < 12 caracteres | — (solo alerta) | `agente_kyc` |
| CURP formato | ≠ 18 caracteres | — (solo alerta) | `agente_kyc` |
| `ratio_cuota_ingreso` | > 40% | > 60% | `agente_financiero` |
| `capacidad_pago` | — | < 0 | `agente_financiero` |
| `total_ingresos` | — | = 0 | `agente_financiero` |
| BC SCORE (numérico) | < 550 | < 400 | `agente_buro` |
| `tiene_juicios` | — | `True` | `agente_buro` |
| `creditos_vencidos` | — | > 0 | `agente_buro` |
| `peor_atraso_dias` | — | > 90 | `agente_buro` |
| `saldo_vencido` | — | > 0 | `agente_buro` |
| Código exclusión (-8/-9) | Alerta informativa | No rechaza | `agente_buro` |
| Reporte incompleto | — | BC Score sin cuentas | `agente_buro` |
| Alertas HAWK | Siempre alerta | No rechaza por sí solas | `agente_buro` |
| Decisión final integradora | — | Ver reglas del agente | `agente_decision` (LLM) |

---

## 9. Auditoría y trazabilidad

### 9.1 Expediente de decisión

El `agente_expediente` genera un texto estructurado con todos los resultados del análisis. Este texto:

- Es devuelto en el campo `expediente_texto` de la respuesta API.
- Se muestra en el frontend en la pestaña Resultado.
- Contiene una leyenda explícita indicando que la decisión está **sujeta a validación y firma de ejecutivo**.
- Incluye fecha y hora de generación.

### 9.2 Registro de auditoría en `auditoria.jsonl`

El archivo `auditoria.jsonl` (generado en el directorio raíz del proyecto cuando se ejecuta `prueba.py`) sigue el formato **JSON Lines**: cada línea es un objeto JSON independiente con la siguiente estructura:

```json
{
  "timestamp": "2026-03-22T10:30:00.123456",
  "numero_solicitud": "47465130",
  "cliente": "CRISTEL MARTINEZ MARQUEZ",
  "kyc": { "aprobado": true, "score_biometrico": 98.3, "alertas": [] },
  "financiero": { "aprobado": true, "ingreso_neto": 27000.0, ... },
  "buro": { "aprobado": true, "score": 662, ... },
  "decision": { "decision": "APROBADO", "score_riesgo": 2, ... }
}
```

**Características del registro:**
- **Append-only:** Nunca se sobreescribe. Cada análisis agrega una línea nueva.
- **Legible por líneas:** Cada línea es JSON válido independiente. Fácil de procesar con `jq`, Python, o cualquier herramienta de logs.
- **No incluye datos biométricos raw:** Solo los resultados del análisis, no los valores originales de los scores de identificación.

> **Limitación actual:** La API REST (`api.py`) no escribe en `auditoria.jsonl`. Solo `prueba.py` lo hace actualmente. Para producción, se recomienda agregar escritura en auditoría desde el endpoint `/analizar`.

---

## 10. Limitaciones y consideraciones regulatorias

### Limitaciones técnicas actuales

| Limitación | Impacto | Recomendación |
|---|---|---|
| Sin autenticación en la API | Cualquier proceso en la red local puede llamar `/analizar` | Agregar API Key o JWT antes de exponer en red |
| `auditoria.jsonl` no se escribe desde la API | Las solicitudes vía frontend no quedan registradas | Mover la lógica de auditoría a `api.py` |
| Sin persistencia de sesiones ni historial en el frontend | No se puede ver el historial de solicitudes previas | Agregar base de datos o caché |
| El modelo LLM puede generar respuestas variables | La misma solicitud puede dar decisiones distintas en distintas ejecuciones | Usar `temperature=0` para mayor determinismo, o agregar validación de consistencia |
| Sin validación real de RFC/CURP contra RENAPO | Solo se valida el formato, no la autenticidad | Integrar consulta RENAPO en producción |
| Sin consulta directa a Buró de Crédito | El sistema recibe el reporte ya consultado | Integrar API de Buró de Crédito (Circle of Trust) |

### Consideraciones regulatorias

**CNBV y LACP:**
- El sistema es una **herramienta de apoyo a la decisión** (Decision Support System). La decisión crediticia definitiva, con sus implicaciones legales y regulatorias, debe ser tomada y firmada por un ejecutivo humano debidamente acreditado.
- Se debe mantener evidencia de que el ejecutivo revisó el expediente y tomó la decisión de forma informada.
- Los parámetros de riesgo (umbrales de score, ratios máximos de cuota/ingreso) deben estar alineados con las políticas de crédito aprobadas por el Comité de Crédito de la institución y documentadas ante la CNBV.

**PLD (Prevención de Lavado de Dinero):**
- Las alertas HAWK del buró son evaluadas pero no se comparan con listas de sanciones (OFAC, SAT, ONU).
- En producción, se recomienda integrar una consulta a listas negras como trigger antes de `agente_kyc`.
- El sistema no realiza Debida Diligencia reforzada para clientes PEP (Personas Políticamente Expuestas).

**LFPDPPP (Protección de Datos Personales):**
- En la versión actual (local), los datos del solicitante no salen de la infraestructura de la institución.
- En una implementación en la nube, se deben establecer políticas de retención, consentimiento explícito del titular y procedimientos de derecho al olvido conforme a la ley.
- Los datos biométricos (`verificacion_id`, `reconocimiento_facial`, `deteccion_vida`) son datos sensibles bajo la LFPDPPP y requieren tratamiento especial.
