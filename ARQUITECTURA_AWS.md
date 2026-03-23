# Arquitectura AWS — Motor de Decisión Crediticia

**Documento:** Plan de migración y arquitectura cloud objetivo
**Sistema:** Motor de Decisión Crediticia — SOFIPO
**Versión del documento:** 1.0
**Fecha:** Marzo 2026

---

## Índice

1. [Arquitectura actual (local/demo)](#1-arquitectura-actual-localdemo)
2. [Arquitectura objetivo en AWS](#2-arquitectura-objetivo-en-aws)
3. [Componentes y justificación](#3-componentes-y-justificación)
4. [Estimación de costos AWS](#4-estimación-de-costos-aws)
5. [Implicaciones regulatorias y de cumplimiento](#5-implicaciones-regulatorias-y-de-cumplimiento)
6. [Plan de migración](#6-plan-de-migración)
7. [Consideraciones de seguridad](#7-consideraciones-de-seguridad)
8. [Escalabilidad y performance](#8-escalabilidad-y-performance)

---

## 1. Arquitectura actual (local/demo)

La versión actual es una aplicación monolítica ejecutada en una sola máquina local:

```
[Navegador] → [FastAPI :8000] → [LangGraph] → [Ollama :11434 / Mistral]
                                               ↓
                                        auditoria.jsonl
```

**Características:**
- Sin autenticación ni autorización.
- Sin persistencia de base de datos: auditoría en archivo local (JSONL).
- Sin alta disponibilidad: un solo proceso, sin redundancia.
- Sin cifrado en tránsito (HTTP plano en red local).
- El modelo LLM corre en la misma máquina que la API.
- Capacidad de procesamiento: una solicitud a la vez (sin concurrencia).

Esta arquitectura es válida para demo y pruebas internas. No es apta para producción regulada.

---

## 2. Arquitectura objetivo en AWS

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              INTERNET                                        │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │ HTTPS (TLS 1.3)
                                   ▼
                    ┌──────────────────────────┐
                    │      AMAZON CLOUDFRONT   │
                    │  CDN + SSL Termination   │
                    │  (Frontend estático S3)  │
                    └──────────────┬───────────┘
                                   │
                    ┌──────────────▼───────────┐
                    │        AWS WAF           │
                    │  (Rate limiting, OWASP   │
                    │   top 10, IP allowlist)  │
                    └──────────────┬───────────┘
                                   │
                    ┌──────────────▼───────────┐
                    │      API GATEWAY         │
                    │  (REST API, throttling,  │
                    │   Cognito Authorizer)    │
                    └──────────────┬───────────┘
                                   │
          ┌────────────────────────▼─────────────────────────┐
          │                  VPC PRIVADA                      │
          │           (10.0.0.0/16 — us-east-1)              │
          │                                                   │
          │    Subnet Pública        Subnet Privada           │
          │   ┌────────────┐        ┌──────────────────────┐ │
          │   │ NAT Gateway│        │  ECS FARGATE CLUSTER │ │
          │   │ (salida)   │        │  FastAPI + LangGraph  │ │
          │   └─────┬──────┘        │  Task: 2 vCPU / 4 GB  │ │
          │         │               │  Auto Scaling: 1–10   │ │
          │         │               └──────────┬───────────┘ │
          │         │                          │              │
          │         │         ┌────────────────┼─────────┐   │
          │         │         │                │         │   │
          │         │         ▼                ▼         ▼   │
          │         │  ┌────────────┐  ┌──────────┐  ┌─────┐│
          │         │  │EC2 GPU     │  │   RDS    │  │  S3 ││
          │         │  │g4dn.xlarge │  │PostgreSQL│  │Docs ││
          │         │  │Ollama +    │  │(Auditoría│  │PDFs ││
          │         │  │Mistral 7B  │  │, Expedi- │  │Img  ││
          │         │  │            │  │entes)    │  │     ││
          │         │  └─────┬──────┘  └──────────┘  └─────┘│
          │         │        │                               │
          │         │        ▼                               │
          │         │  ┌──────────────┐                      │
          │         │  │ ELASTICACHE  │                      │
          │         │  │ Redis        │                      │
          │         │  │ (caché buró  │                      │
          │         │  │  por RFC)    │                      │
          │         │  └──────────────┘                      │
          │         │                                        │
          └─────────┼────────────────────────────────────────┘
                    │
          ┌─────────▼──────────────────────────────────────────┐
          │              SERVICIOS TRANSVERSALES                │
          │                                                     │
          │  ┌─────────────┐  ┌──────────────┐  ┌──────────┐  │
          │  │   COGNITO   │  │  CLOUDWATCH  │  │   SQS    │  │
          │  │ (Auth ejec- │  │ (Logs, Met-  │  │(Análisis │  │
          │  │  utivos)    │  │  rics, Alarm)│  │ asínc.)  │  │
          │  └─────────────┘  └──────────────┘  └──────────┘  │
          │                                                     │
          │  ┌───────────────────────────────────────────────┐ │
          │  │          SECRETS MANAGER                       │ │
          │  │  (DB password, API keys, Cognito secrets)     │ │
          │  └───────────────────────────────────────────────┘ │
          │                                                     │
          │  ┌───────────────────────────────────────────────┐ │
          │  │          AWS KMS                               │ │
          │  │  (Cifrado en reposo: RDS, S3, ElastiCache)    │ │
          │  └───────────────────────────────────────────────┘ │
          └────────────────────────────────────────────────────┘
```

---

## 3. Componentes y justificación

### 3.1 API Gateway + ECS Fargate vs. EC2 directo

**Opción seleccionada: API Gateway + ECS Fargate**

| Aspecto | API Gateway + Fargate | EC2 directo |
|---|---|---|
| Gestión de servidores | Sin servidores que administrar | Requiere parcheo, actualizaciones del SO |
| Escalado | Automático, horizontal, por solicitud | Manual o con Auto Scaling Groups más complejos |
| Alta disponibilidad | Integrada: tareas en múltiples AZs | Requiere configurar ELB + múltiples instancias |
| Costo en baja demanda | Paga por uso (Fargate Spot disponible) | La instancia corre aunque no haya tráfico |
| Integración con WAF y Cognito | Nativa en API Gateway | Requiere configuración adicional |
| Tiempo de arranque | 20–60 segundos (cold start) | Segundos si la instancia ya está corriendo |
| Control del entorno | Contenedor Docker (reproducible) | Control total del SO |

**Recomendación para este sistema:** API Gateway + ECS Fargate es la opción más adecuada para el volumen de transacciones típico de una SOFIPO (decenas a cientos de solicitudes diarias). El tiempo de respuesta del LLM (5–15 segundos) domina la latencia total, haciendo irrelevante el overhead de Fargate.

**Definición del Task de Fargate:**

```
CPU: 2 vCPU
Memoria: 4 GB
Imagen: Python 3.12 + FastAPI + LangGraph
Variables de entorno desde Secrets Manager:
  - OLLAMA_BASE_URL (endpoint interno EC2 GPU)
  - DATABASE_URL (RDS)
  - REDIS_URL (ElastiCache)
```

---

### 3.2 Ollama en EC2 GPU vs. Amazon Bedrock

Esta es la decisión de arquitectura más relevante para el sistema.

#### Opción A: Ollama en EC2 GPU (g4dn.xlarge)

| Especificación | Valor |
|---|---|
| GPU | NVIDIA T4 (16 GB VRAM) |
| vCPU | 4 |
| RAM | 16 GB |
| Precio bajo demanda | ~$0.526/hora |
| Precio Spot | ~$0.16–0.21/hora |
| Precio reservado 1 año | ~$0.34/hora |

**Ventajas:**
- Privacidad total: los datos nunca salen de la infraestructura propia.
- Modelo fijo: comportamiento consistente y auditable.
- Sin costo por token: el modelo corre localmente.
- Compatible con el código actual sin modificaciones (solo cambia la URL de Ollama).

**Desventajas:**
- La instancia debe correr 24/7 para evitar tiempos de arranque del modelo (>2 min de carga del modelo en memoria).
- Requiere gestión de la instancia EC2 (actualizaciones, monitoreo, backups).
- Costo fijo independiente del volumen de solicitudes.

**Costo mensual estimado (instancia activa 24/7):**

| Modalidad | Costo/hora | Costo/mes |
|---|---|---|
| On-Demand | $0.526 | ~$379 |
| Spot (varía) | ~$0.18 | ~$130 |
| Reserved 1 año | ~$0.34 | ~$245 |

#### Opción B: Amazon Bedrock (Claude Haiku / Sonnet)

Reemplazar Ollama/Mistral con un modelo administrado de Bedrock.

| Modelo | Costo input | Costo output | Notas |
|---|---|---|---|
| Claude Haiku 3 | $0.00025/1K tokens | $0.00125/1K tokens | Más rápido, menos preciso |
| Claude Sonnet 3.5 | $0.003/1K tokens | $0.015/1K tokens | Mayor calidad de análisis |
| Llama 3 70B (Bedrock) | $0.00265/1K tokens | $0.0035/1K tokens | Open source en infraestructura AWS |

**Estimación de costo por solicitud (análisis completo):**
- Tokens promedio por solicitud: ~2,000 input + ~500 output
- Costo por solicitud con Claude Haiku: `(2,000 × $0.00025) + (500 × $0.00125)` ≈ $0.00113

**Costo mensual con Bedrock (500 solicitudes/mes):**
- Claude Haiku: ~$0.56/mes
- Claude Sonnet 3.5: ~$10.50/mes

**Ventajas de Bedrock:**
- Sin infraestructura GPU que administrar.
- Modelos de mayor calidad disponibles.
- Alta disponibilidad garantizada por AWS.
- Escalado instantáneo.

**Desventajas de Bedrock:**
- Los datos del solicitante (RFC, CURP, información financiera) se envían a AWS para procesamiento. Requiere análisis de privacidad y posiblemente Data Processing Agreement.
- No es completamente "local" — sensible para datos regulados.
- Costo variable (puede escalar con el volumen).

**Recomendación:** Para una SOFIPO con requisitos estrictos de privacidad y cumplimiento PLD, **EC2 GPU con Ollama es la opción preferida** porque garantiza que ningún dato personal se transmite fuera de la infraestructura controlada por la institución. Para uso en desarrollo y staging, Bedrock es más económico y ágil.

---

### 3.3 RDS PostgreSQL — Reemplazar `auditoria.jsonl`

La base de datos PostgreSQL en RDS reemplaza el archivo `auditoria.jsonl` con una solución estructurada, consultable y con respaldos automáticos.

**Esquema sugerido:**

```sql
-- Tabla principal de solicitudes
CREATE TABLE solicitudes (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    numero_solicitud VARCHAR(20) NOT NULL,
    timestamp        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    cliente_nombre   VARCHAR(200),
    rfc              VARCHAR(15),
    monto            DECIMAL(12,2),
    producto         VARCHAR(100),

    -- Resultados por agente (JSON para flexibilidad)
    resultado_kyc        JSONB,
    resultado_financiero JSONB,
    resultado_buro       JSONB,
    analisis_buro        JSONB,
    decision_final       JSONB,

    -- Decisión
    decision       VARCHAR(20) CHECK (decision IN ('APROBADO','RECHAZADO','ESCALAR_EJECUTIVO')),
    score_riesgo   INT CHECK (score_riesgo BETWEEN 1 AND 10),

    -- Auditoría del ejecutivo
    ejecutivo_id         VARCHAR(50),
    ejecutivo_decision   VARCHAR(20),
    ejecutivo_timestamp  TIMESTAMPTZ,
    ejecutivo_notas      TEXT,

    -- Metadatos
    version_sistema  VARCHAR(10) DEFAULT '2.0.0',
    ambiente         VARCHAR(20) DEFAULT 'produccion'
);

CREATE INDEX idx_solicitudes_rfc ON solicitudes(rfc);
CREATE INDEX idx_solicitudes_timestamp ON solicitudes(timestamp DESC);
CREATE INDEX idx_solicitudes_decision ON solicitudes(decision);
```

**Configuración RDS recomendada:**
- Instancia: `db.t4g.small` (2 vCPU, 2 GB RAM) — suficiente para el volumen de una SOFIPO mediana.
- Motor: PostgreSQL 16
- Multi-AZ: Sí (alta disponibilidad y failover automático)
- Backup automático: 7 días de retención
- Cifrado en reposo: Habilitado con KMS

---

### 3.4 S3 — Almacenamiento de documentos del expediente

**Bucket `sofipo-expedientes-prod`:**

```
sofipo-expedientes-prod/
├── expedientes/
│   └── {AAAA}/{MM}/
│       └── {numero_solicitud}/
│           ├── expediente.txt          # Texto del expediente
│           ├── expediente.pdf          # PDF generado (futuro)
│           ├── ine_frente.jpg          # Foto del documento de ID
│           ├── ine_reverso.jpg
│           └── selfie_video.mp4        # Video selfie
└── backups/
    └── auditoria/                      # Export periódico de la BD
```

**Configuración:**
- Versionado habilitado (requerido para auditoría regulatoria).
- Política de ciclo de vida: mover a S3 Glacier después de 90 días.
- Cifrado del lado del servidor con KMS (SSE-KMS).
- Bloqueo de objetos (Object Lock) en modo COMPLIANCE para expedientes (retención mínima 5 años conforme a normativa).
- Acceso privado: sin acceso público. Solo el rol IAM de la aplicación puede escribir; los ejecutivos acceden mediante URLs pre-firmadas con tiempo de expiración.

---

### 3.5 ElastiCache Redis — Caché de análisis de buró

Los análisis de buró son costosos (requieren llamada al LLM) y los reportes de buró tienen validez de 30 días. Se puede cachear el resultado del análisis por RFC.

**Llave de caché:**

```
buro:analisis:{rfc}:{hash_sha256_del_reporte}
TTL: 24 horas (el reporte puede actualizarse en el día)
```

**Flujo con caché:**

```
1. Recibir solicitud con RFC y reporte de buró
2. Calcular hash del reporte de buró
3. Buscar en Redis: buro:analisis:{rfc}:{hash}
4a. HIT → devolver resultado cacheado (evita llamada al LLM)
4b. MISS → ejecutar agente_analisis_buro → guardar en Redis → continuar
```

**Ahorro estimado:** Si el 30% de solicitudes son re-análisis (mismo cliente, mismo reporte), el caché reduce el tiempo de respuesta en esos casos de ~15s a ~2s y elimina esas llamadas al modelo GPU.

**Configuración:**
- Nodo: `cache.t4g.small` (~$0.025/hora, ~$18/mes)
- Motor: Redis 7
- Cifrado en tránsito y en reposo habilitado
- Sin disponibilidad multi-AZ para este caso (el dato es regenerable)

---

### 3.6 SQS — Análisis asíncrono con callback

Para integraciones donde el sistema origen no puede esperar 15–60 segundos (core bancario, plataformas de originación), se puede implementar un flujo asíncrono.

**Flujo asíncrono:**

```
1. Cliente llama POST /analizar/async con solicitud + webhook_url
2. API devuelve inmediatamente: {"job_id": "uuid", "status": "en_proceso"}
3. API publica la solicitud en SQS
4. Worker ECS (tarea separada) consume el mensaje de SQS
5. Worker ejecuta el grafo LangGraph completo
6. Worker guarda resultado en RDS
7. Worker hace POST al webhook_url con el resultado completo
```

**Configuración SQS:**
- Cola estándar (no FIFO — el orden no es crítico aquí)
- `VisibilityTimeout`: 120 segundos (más que el tiempo máximo de análisis)
- `MessageRetentionPeriod`: 4 días
- Dead Letter Queue: para mensajes que fallaron 3 veces

---

### 3.7 CloudFront — Frontend estático

El archivo `static/index.html` se despliega en S3 y se sirve a través de CloudFront.

**Configuración:**
- Origen: S3 bucket `sofipo-frontend-prod` (acceso solo vía OAC — Origin Access Control)
- Certificado SSL: ACM (AWS Certificate Manager) en us-east-1 (requerido para CloudFront)
- Cache policy: `CACHING_OPTIMIZED` con TTL de 1 día (el frontend cambia raramente)
- Dominio personalizado: `credito.sofipo.com.mx`
- Geo-restriction: Solo México (MX) — bloquear otros países

---

### 3.8 Cognito — Autenticación de ejecutivos

**User Pool para ejecutivos de crédito:**

```
Grupos:
  - ejecutivos_credito     (acceso de solo lectura + envío de solicitudes)
  - coordinadores_credito  (acceso a reportes y estadísticas)
  - administradores        (gestión de usuarios y configuración)

Configuración de seguridad:
  - MFA: TOTP obligatorio (Google Authenticator / Authy)
  - Contraseña: mínimo 12 caracteres, caducidad 90 días
  - Sesión: token JWT con duración de 8 horas
  - IP allowlist: solo IPs corporativas de la SOFIPO
```

**Integración con API Gateway:**

```
API Gateway → Cognito Authorizer → valida JWT
                                 → verifica grupo del usuario
                                 → adjunta claims al request context
```

---

## 4. Estimación de costos AWS

Estimación para una SOFIPO mediana con ~500 solicitudes/mes, en región `us-east-1`.

| Componente | Servicio / Instancia | Costo mensual estimado | Notas |
|---|---|---|---|
| Frontend estático | S3 + CloudFront | ~$3 | 1 GB transferencia, 10K requests |
| WAF | AWS WAF | ~$15 | Regla base OWASP + rate limiting |
| API Gateway | REST API | ~$5 | 500 solicitudes + 50K health checks |
| Aplicación FastAPI | ECS Fargate (2 vCPU / 4 GB) | ~$60 | 730 hrs/mes, 1 tarea activa |
| Auto Scaling adicional | ECS Fargate (escala esporádica) | ~$15 | Estimado picos de demanda |
| Modelo LLM | EC2 g4dn.xlarge (24/7) | ~$379 | On-Demand. Reservado 1 año: ~$245 |
| Base de datos | RDS PostgreSQL db.t4g.small Multi-AZ | ~$55 | 20 GB almacenamiento |
| Caché de buró | ElastiCache Redis cache.t4g.small | ~$18 | Sin Multi-AZ |
| Cola asíncrona | SQS (500 mensajes) | < $1 | Dentro del free tier |
| Almacenamiento documentos | S3 (10 GB/mes acumulado) | ~$2 | Estándar + Glacier después de 90d |
| Autenticación | Cognito (50 usuarios activos) | ~$3 | Free tier cubre primeros 50K MAU |
| Gestión de secretos | Secrets Manager (5 secretos) | ~$2 | $0.40/secreto/mes |
| Cifrado | KMS (5 claves, 100K ops) | ~$1 | $1/clave/mes |
| Monitoreo | CloudWatch (logs + metrics) | ~$15 | 5 GB logs, dashboards básicos |
| Transferencia de datos | Data Transfer Out | ~$5 | Estimado |
| **TOTAL (On-Demand)** | | **~$579/mes** | |
| **TOTAL (EC2 Reserved 1 año)** | | **~$445/mes** | EC2 GPU con descuento |

> **Nota:** Si se adopta Amazon Bedrock en lugar de EC2 GPU, el costo del LLM baja a ~$1–15/mes (según volumen y modelo elegido), reduciendo el total a ~$200/mes. Sin embargo, esto implica que los datos del solicitante se procesan fuera de la infraestructura controlada.

---

## 5. Implicaciones regulatorias y de cumplimiento

### 5.1 Región AWS

**Situación actual:** AWS no tiene región en México. La región más cercana disponible es `us-east-1` (Virginia) o `us-east-2` (Ohio). La región `mx-central-1` (Querétaro) fue anunciada pero aún no está en disponibilidad general a la fecha de este documento.

| Región | Latencia desde CDMX | Disponibilidad | Recomendación |
|---|---|---|---|
| `us-east-1` (Virginia) | ~60–80 ms | Disponible | Usar mientras mx-central-1 no esté disponible |
| `us-east-2` (Ohio) | ~70–90 ms | Disponible | Alternativa con menor carga |
| `mx-central-1` (Querétaro) | <10 ms | Pendiente | Migrar cuando esté disponible |

**Consideraciones legales de la región:** El artículo 36 Bis de la LFPDPPP permite la transferencia de datos personales a terceros (incluyendo nubes internacionales) bajo contratos que garanticen niveles equivalentes de protección. AWS tiene el **AWS Data Processing Addendum** disponible para este propósito.

---

### 5.2 Cifrado

**En tránsito:**
- TLS 1.3 en todos los endpoints externos (CloudFront, API Gateway).
- TLS 1.2+ en comunicación interna entre servicios VPC.
- Certificados administrados por ACM con renovación automática.

**En reposo:**
- RDS: Cifrado con KMS (`aws/rds` o clave gestionada por el cliente).
- S3: SSE-KMS para todos los buckets con datos personales.
- ElastiCache: Cifrado habilitado (Redis at-rest encryption).
- EBS/volúmenes del EC2 GPU: Cifrado con KMS.

**Gestión de claves:**
- Usar **Customer Managed Keys (CMK)** en KMS para datos sensibles (no las claves administradas por AWS `aws/rds`).
- Política de rotación automática de CMK cada 365 días.
- Auditoría de uso de claves en CloudTrail.

---

### 5.3 LFPDPPP — Datos personales

Los datos procesados por el sistema que constituyen datos personales bajo la LFPDPPP incluyen:

| Dato | Categoría LFPDPPP | Medida requerida |
|---|---|---|
| Nombre completo | Dato personal | Consentimiento en solicitud de crédito |
| RFC | Dato personal | Consentimiento en solicitud de crédito |
| CURP | Dato personal | Consentimiento en solicitud de crédito |
| Fecha de nacimiento | Dato personal | Consentimiento en solicitud de crédito |
| Datos biométricos (scores) | **Dato sensible** | Consentimiento expreso y por escrito |
| Historial crediticio | Dato financiero (LFPIOR) | Consulta autorizada en contrato |
| Foto/video del solicitante | **Dato sensible** | Consentimiento expreso |

**Política de retención:** La CNBV requiere que los expedientes de crédito se conserven mínimo **5 años** después del vencimiento del crédito o del rechazo. En S3, implementar Object Lock en modo COMPLIANCE con período de retención de 6 años (margen de seguridad).

**Derecho al olvido:** Para solicitudes rechazadas o no ejercidas, el titular puede solicitar la eliminación de sus datos. En la arquitectura propuesta, esto implica:
1. Eliminar el registro de `solicitudes` en RDS (o marcar como eliminado lógicamente).
2. Eliminar los archivos del expediente en S3 (Object Lock debe respetar los plazos mínimos primero).
3. Invalidar el caché en Redis si existe.

---

### 5.4 CNBV — Trazabilidad de decisiones y human-in-the-loop

La CNBV exige que los sistemas de decisión crediticia mantengan una pista de auditoría completa y que exista intervención humana en el proceso de otorgamiento.

**Trazabilidad en la arquitectura propuesta:**

Cada solicitud genera un registro inmutable en RDS con:
- Timestamp de cada etapa del análisis.
- Resultado de cada agente con sus alertas específicas.
- Versión del sistema que generó la decisión (para reproducibilidad).
- Identidad del ejecutivo que revisó y aprobó/rechazó la solicitud.
- Timestamp de la acción del ejecutivo.

**Human-in-the-loop obligatorio:**

El sistema debe implementar un flujo de aprobación explícito:

```
1. Sistema emite: APROBADO / RECHAZADO / ESCALAR_EJECUTIVO
2. Ejecutivo revisa el expediente en el frontend
3. Ejecutivo registra su decisión definitiva:
   - Confirma la recomendación del sistema, o
   - La modifica con justificación documentada
4. La decisión del ejecutivo queda registrada en RDS con su ID y timestamp
5. Solo entonces se puede proceder al desembolso
```

El campo `decision` del sistema es una **recomendación**, no una resolución. La resolución es el campo `ejecutivo_decision` en la base de datos.

---

### 5.5 PLD — Prevención de Lavado de Dinero

**Mejoras requeridas para producción:**

1. **Lista OFAC:** Antes de `agente_kyc`, agregar una función Lambda que consulte las listas de sanciones de OFAC (Office of Foreign Assets Control), SAT (lista negra de contribuyentes) y ONU.

```
[SQS] → [Lambda: PLD Check] → {
    GET https://api.ofac.treasury.gov/v1/...
    GET https://sat.gob.mx/...
    Si hit → rechazar automáticamente + escalar a Oficial de Cumplimiento
    Si no → continuar flujo normal
}
```

2. **PEP (Personas Políticamente Expuestas):** Integrar consulta a bases de datos de PEP para aplicar Debida Diligencia Reforzada (DDR).

3. **Reporte de Operaciones Inusuales (ROU):** El sistema debe generar alertas cuando detecte patrones inusuales (múltiples solicitudes del mismo RFC en corto tiempo, montos inconsistentes con el giro del negocio).

4. **Archivo de Operaciones Monitoreadas:** Configurar CloudWatch Events para alertar al Oficial de Cumplimiento ante ciertos patrones de decisión.

---

### 5.6 Backup y Disaster Recovery

| Componente | RPO (Recovery Point Objective) | RTO (Recovery Time Objective) | Estrategia |
|---|---|---|---|
| RDS PostgreSQL | 5 minutos (Point-in-Time Recovery) | 15–30 minutos | Multi-AZ + backup automático diario |
| S3 (documentos) | 0 (replicación síncrona interna) | Inmediato | Versionado habilitado + Object Lock |
| EC2 GPU (Ollama) | 4 horas (snapshot diario) | 30–60 minutos | AMI con Ollama + modelo pre-instalado |
| ECS Fargate (API) | N/A (sin estado) | <5 minutos | Nueva tarea desde imagen ECR |
| ElastiCache Redis | N/A (caché regenerable) | <5 minutos | Dato es regenerable, no crítico |

**Estrategia de backup para RDS:**
- Backup automático diario en ventana de baja carga (02:00–04:00 UTC).
- Retención: 30 días de backups diarios.
- Snapshot manual mensual: retención permanente (requerido para auditoría CNBV).
- Cross-region snapshot a `us-west-2` para DR extremo.

---

## 6. Plan de migración

### Fase 1 — Demo local (estado actual)

**Duración:** Actualmente en uso

**Características:**
- Monolito local en una sola máquina (Mac / laptop).
- Sin autenticación, sin base de datos, sin alta disponibilidad.
- Útil para: validación del modelo de negocio, pruebas de concepto, presentaciones internas.

**Criterio de salida:** Validación del flujo de análisis crediticio con casos reales de la institución. Aprobación del Comité de Crédito para piloto controlado.

---

### Fase 2 — EC2 simple (piloto controlado)

**Duración estimada:** 2–4 semanas de implementación + 4–8 semanas de piloto

**Objetivo:** Llevar el sistema a un ambiente accesible para ejecutivos de crédito, con seguridad básica y auditoría persistente.

**Arquitectura:**
- 1 instancia EC2 `g4dn.xlarge` con Ollama + FastAPI + PostgreSQL.
- Nginx como proxy reverso con HTTPS (certificado Let's Encrypt o ACM).
- Autenticación básica con API Key en los headers.
- PostgreSQL local en la misma instancia (no RDS aún).
- Acceso solo desde IPs corporativas (Security Group).

**Pasos de implementación:**

```bash
# 1. Lanzar instancia g4dn.xlarge con Ubuntu 22.04 + NVIDIA drivers
# 2. Instalar Docker, Ollama, uv
# 3. Pull del modelo Mistral
# 4. Clonar repositorio, crear imagen Docker
# 5. Configurar Nginx + SSL
# 6. Configurar Security Group: solo puerto 443 desde IPs corporativas
# 7. Configurar backup diario de PostgreSQL a S3
```

**Criterio de salida:** Piloto exitoso con 50–100 solicitudes reales. Tiempo de respuesta <30s. Sin incidentes de seguridad.

---

### Fase 3 — Arquitectura completa AWS

**Duración estimada:** 6–10 semanas de implementación

**Objetivo:** Arquitectura production-ready con alta disponibilidad, cumplimiento regulatorio completo y monitoreo.

**Orden de implementación:**

| Semana | Componente | Descripción |
|---|---|---|
| 1 | VPC, subnets, Security Groups | Red base privada |
| 1–2 | EC2 GPU + Ollama | Servidor del modelo LLM |
| 2 | RDS PostgreSQL Multi-AZ | Base de datos de auditoría |
| 2–3 | ECS Fargate + ECR | Contenedorizar la API y desplegar |
| 3 | Cognito | Autenticación de ejecutivos |
| 3–4 | API Gateway + WAF | Exponer API con seguridad |
| 4 | CloudFront + S3 | Frontend estático |
| 4–5 | ElastiCache Redis | Caché de análisis de buró |
| 5 | SQS + Worker asíncrono | Análisis async para integraciones |
| 5–6 | Secrets Manager + KMS | Gestión de secretos y cifrado |
| 6 | CloudWatch dashboards + alertas | Observabilidad |
| 6+ | Integración PLD (Lambda + OFAC) | Cumplimiento PLD |

**Herramientas de infraestructura como código (IaC) recomendadas:**
- **Terraform** o **AWS CDK (Python)** para todos los recursos.
- Repositorio Git separado para infraestructura.
- Ambientes separados: `dev`, `staging`, `produccion`.

---

## 7. Consideraciones de seguridad

### 7.1 IAM — Principio de mínimo privilegio

**Roles IAM definidos:**

| Rol | Servicios permitidos | Uso |
|---|---|---|
| `ecs-task-role` | S3 (expedientes), RDS (conexión), Secrets Manager (lectura), ElastiCache, SQS | Tarea Fargate de la API |
| `ec2-ollama-role` | CloudWatch Logs (escritura), Systems Manager (SSM para acceso sin SSH) | Instancia EC2 GPU |
| `lambda-pld-role` | SQS (lectura/escritura), RDS (escritura), Secrets Manager (lectura) | Lambda de verificación PLD |
| `developer-role` | ECR (push imágenes), ECS (despliegue), CloudWatch (lectura) | Equipo de desarrollo (asumible) |

**Política crítica:** Ningún rol de producción tiene permisos `iam:*`, `ec2:*` irrestrictos o `s3:*` sobre todos los buckets. Los accesos son específicos por recurso con ARN explícitos.

---

### 7.2 VPC y segmentación de red

```
VPC: 10.0.0.0/16

Subnets públicas (NAT Gateway, Load Balancer):
  10.0.1.0/24  — us-east-1a
  10.0.2.0/24  — us-east-1b

Subnets privadas (ECS, EC2 GPU, RDS, ElastiCache):
  10.0.10.0/24 — us-east-1a
  10.0.11.0/24 — us-east-1b
```

**Security Groups:**

| Grupo | Inbound | Outbound | Recursos |
|---|---|---|---|
| `sg-api-fargate` | TCP 8000 desde `sg-api-gateway` | TCP 11434 a `sg-ollama`, TCP 5432 a `sg-rds`, TCP 6379 a `sg-redis` | Tareas ECS |
| `sg-ollama` | TCP 11434 desde `sg-api-fargate` | TCP 443 (NAT, para pull de modelos) | EC2 g4dn.xlarge |
| `sg-rds` | TCP 5432 desde `sg-api-fargate` | — | RDS PostgreSQL |
| `sg-redis` | TCP 6379 desde `sg-api-fargate` | — | ElastiCache |
| `sg-api-gateway` | — | TCP 8000 a `sg-api-fargate` | VPC Link de API Gateway |

**No hay SSH (puerto 22) abierto** hacia ningún grupo. El acceso al EC2 GPU se realiza exclusivamente a través de AWS Systems Manager Session Manager.

---

### 7.3 AWS WAF — Reglas aplicadas

| Regla | Tipo | Acción |
|---|---|---|
| AWSManagedRulesCommonRuleSet | Managed (OWASP Top 10) | Block |
| AWSManagedRulesKnownBadInputsRuleSet | Managed | Block |
| Rate limiting | Custom: >100 req/5min por IP | Block |
| Geo restriction | Custom: solo MX | Block |
| Body size limit | Custom: max 1 MB para `/analizar` | Block |
| SQL injection | AWSManagedRulesSQLiRuleSet | Block |

---

### 7.4 Secrets Manager y rotación

Los siguientes secretos se almacenan en Secrets Manager (nunca en variables de entorno en texto plano ni en código):

| Secreto | Descripción | Rotación |
|---|---|---|
| `prod/rds/credentials` | Usuario y contraseña de RDS | Automática cada 30 días |
| `prod/cognito/client-secret` | Client secret de Cognito | Manual al regenerar |
| `prod/ollama/endpoint` | URL interna del servidor Ollama | Manual al cambiar instancia |
| `prod/redis/auth-token` | Token de autenticación de ElastiCache | Automática cada 30 días |

La rotación automática de credenciales RDS usa una función Lambda de rotación nativa de Secrets Manager, compatible con PostgreSQL.

---

## 8. Escalabilidad y performance

### 8.1 Auto Scaling de ECS Fargate

El cluster de Fargate escala horizontalmente basándose en métricas:

| Métrica | Umbral de scale-out | Umbral de scale-in |
|---|---|---|
| CPU promedio | > 70% por 2 minutos | < 30% por 5 minutos |
| Memoria promedio | > 75% por 2 minutos | < 40% por 5 minutos |
| Request count (SQS) | > 10 mensajes en cola | < 2 mensajes en cola |

**Configuración:**
- Mínimo: 1 tarea (para evitar cold starts)
- Máximo: 10 tareas
- Cooldown de scale-out: 60 segundos
- Cooldown de scale-in: 300 segundos

**Nota importante:** El cuello de botella no es la API de FastAPI sino el servidor Ollama. Con una sola instancia EC2 GPU, el throughput máximo es ~1 solicitud concurrente procesándose en el LLM. Para mayor concurrencia, se necesitan múltiples instancias EC2 GPU o un balanceador de carga de inferencia (ver nota sobre Bedrock).

---

### 8.2 RDS Read Replicas

Para consultas de auditoría y reportes (que no requieren datos en tiempo real):

- 1 read replica en `us-east-1b` para consultas de dashboards.
- Las escrituras van siempre a la instancia primaria.
- Las lecturas de reportes históricos se dirigen a la réplica.

---

### 8.3 Caché de análisis de buró

Con ElastiCache Redis, las solicitudes para clientes con análisis reciente en caché reducen drásticamente el tiempo de respuesta:

| Escenario | Tiempo estimado sin caché | Tiempo estimado con caché (HIT) |
|---|---|---|
| Análisis completo nuevo (Apple M2) | 8–15 segundos | — |
| Análisis completo nuevo (EC2 g4dn) | 5–10 segundos | — |
| Solicitud con análisis cacheado | 5–10 segundos | 1–3 segundos |

El `agente_decision` no se puede cachear porque recibe el resumen completo de todos los agentes y varía con cada solicitud.

---

### 8.4 Tiempos estimados por etapa (EC2 g4dn.xlarge)

| Etapa | Tiempo estimado | Notas |
|---|---|---|
| FastAPI — validación Pydantic | < 10 ms | Determinístico |
| `agente_kyc` | < 5 ms | Solo reglas |
| `agente_financiero` | < 5 ms | Solo reglas |
| `agente_buro` | < 10 ms | Solo reglas |
| `agente_analisis_buro` — métricas | < 20 ms | Solo reglas |
| `agente_analisis_buro` — LLM | 2–5 segundos | Mistral en T4 GPU |
| `agente_decision` — LLM | 3–8 segundos | Mistral en T4 GPU |
| `agente_expediente` | < 30 ms | Solo formateo de texto |
| FastAPI — construcción de respuesta | < 10 ms | Serialización JSON |
| **TOTAL (con GPU)** | **~6–14 segundos** | |
| **TOTAL (sin GPU, CPU only)** | **~25–60 segundos** | |
| **TOTAL (con caché de buró)** | **~4–10 segundos** | Solo `agente_decision` invoca LLM |

---

### 8.5 Capacidad estimada del sistema

Con la arquitectura propuesta (1 EC2 g4dn.xlarge + ECS Fargate auto-scaling):

| Métrica | Valor estimado |
|---|---|
| Solicitudes simultáneas procesando LLM | 1 |
| Solicitudes en cola (SQS buffer) | Ilimitado |
| Throughput sostenido | ~4–6 solicitudes/minuto |
| Solicitudes diarias (8 hrs hábiles) | ~2,000–3,000 |
| Solicitudes mensuales | ~40,000–60,000 |

Este throughput es más que suficiente para una SOFIPO con decenas a cientos de ejecutivos activos. Si la demanda supera estos límites, la solución es agregar más instancias EC2 GPU y un balanceador de carga Nginx entre ellas.
