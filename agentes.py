"""
Sistema Multi-Agente de Decisión Crediticia
Orquestado con LangGraph + Ollama local
"""
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from modelos import SolicitudCompleta
import json
import re

# Modelo ML XGBoost — scoring por decil relativo
try:
    from modelo_nuevos.predecir import predecir_desde_solicitud as _predecir_ml
    _ML_DISPONIBLE = True
except Exception as _e:
    _ML_DISPONIBLE = False
    print(f"[AVISO] Modelo ML no disponible: {_e}")

try:
    from base_datos import guardar_caso, buscar_casos_similares
    _DB_DISPONIBLE = True
except Exception as _db_e:
    _DB_DISPONIBLE = False
    print(f"[AVISO] Base de datos no disponible: {_db_e}")

# ── Modelos locales ──────────────────────────────────────────────────────────
# qwen2.5:14b   — análisis estructurado, JSON, narrativa (mejor capacidad general)
# gemma4:e4b    — deliberación con thinking mode integrado (reemplaza deepseek-r1:8b)
# Ambos corren 100% local en Apple Silicon via Ollama. Sin conexión a internet.
llm          = OllamaLLM(model="qwen2.5:14b", temperature=0.1)
llm_razonador = OllamaLLM(model="gemma4:e4b", temperature=1.0, top_p=0.95)

def _parsear_score_buro(score_raw) -> tuple:
    """
    Parsea el campo buro.score que puede venir como:
      - "588/0004"  → (bc_score=588.0, icc=4)   — formato BC SCORE / ICC del reporte
      - "588"       → (bc_score=588.0, icc=None)
      - 588         → (bc_score=588.0, icc=None)
      - None        → (None, None)
    Devuelve (bc_score: float|None, icc: int|None).
    """
    if score_raw is None:
        return None, None
    try:
        s = str(score_raw).strip()
        if "/" in s:
            partes = s.split("/", 1)
            bc  = float(partes[0].strip())
            try:
                icc = int(partes[1].strip())
            except (ValueError, TypeError):
                icc = None
            return bc, icc
        else:
            return float(s), None
    except (ValueError, TypeError):
        return score_raw, None


def _parsear_json(texto: str, fallback: dict) -> dict:
    """Extrae y parsea JSON de la respuesta del modelo de forma robusta."""
    # Limpiar bloques markdown
    texto = re.sub(r'```json\s*', '', texto)
    texto = re.sub(r'```\s*', '', texto)
    texto = texto.strip()

    # Intentar parsear directo
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        pass

    # Extraer primer objeto JSON con regex
    match = re.search(r'\{[\s\S]*?\}(?=\s*$|\s*\n)', texto)
    if not match:
        match = re.search(r'\{[\s\S]*\}', texto)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            # Intentar reparar comillas simples y trailing commas
            fragmento = match.group()
            fragmento = re.sub(r',\s*([}\]])', r'\1', fragmento)  # trailing comma
            try:
                return json.loads(fragmento)
            except json.JSONDecodeError:
                pass

    return fallback

# ─────────────────────────────────────────
# Estado compartido entre todos los agentes
# ─────────────────────────────────────────
class EstadoSolicitud(TypedDict):
    solicitud: dict
    resultado_kyc: dict
    resultado_financiero: dict
    resultado_buro: dict
    analisis_buro: dict
    analisis_buro_completo: str  # narrativa completa del lector de buró (qwen2.5:14b)
    deliberacion_ia: str         # análisis libre del agente deliberador (gemma4:e4b)
    decision_final: dict
    expediente: str
    analisis_riesgo_buro: str
    analisis_perfil_negocio: str
    analisis_alertas: str
    patrones_historicos: str

# ─────────────────────────────────────────
# AGENTE 1: KYC / Identificación
# ─────────────────────────────────────────
def agente_kyc(estado: EstadoSolicitud) -> EstadoSolicitud:
    s = estado["solicitud"]
    ident = s["identificacion"]
    cliente = s["cliente"]

    alertas = []
    aprobado = True

    # ── Biométricos: solo rechaza si es crítico (< 50), entre 50-70 alerta
    if ident["verificacion_id"] < 50:
        alertas.append("Verificación de ID muy baja — posible fraude")
        aprobado = False
    elif ident["verificacion_id"] < 70:
        alertas.append("Verificación de ID baja — revisar documento")

    if ident["reconocimiento_facial"] < 50:
        alertas.append("Reconocimiento facial fallido")
        aprobado = False
    elif ident["reconocimiento_facial"] < 70:
        alertas.append("Reconocimiento facial bajo — revisar")

    if ident["deteccion_vida"] < 50:
        alertas.append("Detección de vida fallida — posible suplantación")
        aprobado = False
    elif ident["deteccion_vida"] < 80:
        alertas.append("Detección de vida baja — revisar")

    # Validación gobierno y selfie: alerta individual por cada canal faltante, no bloquea
    if ident["validacion_gobierno"] == 0:
        alertas.append("Sin validación gobierno — revisar manualmente")
    if ident["video_selfie"] == 0:
        alertas.append("Sin video selfie — revisar manualmente")

    # RFC y CURP: formato básico, solo alerta si es inválido
    if len(cliente.get("rfc", "")) < 12:
        alertas.append("RFC con formato inválido — verificar")
    if len(cliente.get("curp", "")) != 18:
        alertas.append("CURP con formato inválido — verificar")

    estado["resultado_kyc"] = {
        "aprobado": aprobado,
        "score_biometrico": round(
            (ident["verificacion_id"] + ident["reconocimiento_facial"] + ident["deteccion_vida"]) / 3, 1
        ),
        "alertas": alertas
    }
    return estado

# ─────────────────────────────────────────
# AGENTE 2: Análisis Financiero
# ─────────────────────────────────────────
def agente_financiero(estado: EstadoSolicitud) -> EstadoSolicitud:
    s = estado["solicitud"]
    finanzas = s["finanzas"]
    condiciones = s["condiciones"]

    ingresos = finanzas["total_ingresos"]
    egresos = finanzas["total_egresos"]
    cuota = condiciones["cuota"]
    monto = condiciones["monto"]
    pago_buro = s["buro"].get("pago_a_realizar", 0) or 0  # compromisos mensuales existentes en buró

    ingreso_neto = ingresos - egresos
    # Ratio incluye cuota nueva + pagos existentes en buró para reflejar carga real de deuda
    total_compromisos = cuota + pago_buro
    ratio_cuota_ingreso = (total_compromisos / ingreso_neto * 100) if ingreso_neto > 0 else 100
    capacidad_pago = ingreso_neto - total_compromisos
    ratio_monto_ingreso = (monto / ingresos) if ingresos > 0 else 999

    # ── Monto ajustado por capacidad de pago (reducción forzada) ──
    # Perfil holgado (BC ≥ 630, ICC ≥ 5, sin vencidos): límite 40% — mayor tolerancia demostrada históricamente
    # Perfil estándar: límite 35%
    _score_bc, _icc_parsed = _parsear_score_buro(s["buro"].get("score"))
    _score_bc = _score_bc or 0
    _icc_num  = 0
    icc_explicito = s["buro"].get("icc")
    try:
        _icc_num = int(str(icc_explicito if icc_explicito is not None else (_icc_parsed or 0)).strip())
    except (ValueError, TypeError):
        pass
    _sin_vencidos = (s["buro"].get("saldo_vencido", 0) == 0 and s["buro"].get("creditos_vencidos", 0) == 0)
    _perfil_holgado = isinstance(_score_bc, (int, float)) and _score_bc >= 630 and _icc_num >= 5 and _sin_vencidos
    RATIO_MAX = 0.40 if _perfil_holgado else 0.35
    cuota_disponible = ingreso_neto * RATIO_MAX - pago_buro
    if ingreso_neto > 0 and cuota > 0 and cuota_disponible > 0 and cuota_disponible < cuota:
        factor = cuota_disponible / cuota
        monto_ajustado = int((monto * factor) / 500) * 500  # redondeo hacia abajo a $500
        cuota_ajustada = round(cuota * (monto_ajustado / monto), 2) if monto > 0 else 0
        monto_reducido = True
    else:
        monto_ajustado = monto
        cuota_ajustada = cuota
        monto_reducido = False

    alertas = []
    aprobado = True

    # Cuota/ingreso: alerta en >40%, rechaza solo si >60%
    if ratio_cuota_ingreso > 60:
        alertas.append(f"Compromisos totales representan {ratio_cuota_ingreso:.1f}% del ingreso — capacidad de pago insuficiente")
        aprobado = False
    elif ratio_cuota_ingreso > 40:
        alertas.append(f"Compromisos totales representan {ratio_cuota_ingreso:.1f}% del ingreso — revisar con ejecutivo")

    if monto_reducido:
        alertas.append(f"Monto ajustado de ${monto:,.0f} a ${monto_ajustado:,.0f} por capacidad de pago (límite 35% del ingreso neto)")

    # Capacidad de pago negativa: rechaza
    if capacidad_pago < 0:
        alertas.append("Capacidad de pago negativa después de cuota y compromisos en buró")
        aprobado = False

    if ingresos <= 0:
        alertas.append("Sin ingresos declarados")
        aprobado = False
    elif ingreso_neto <= 0:
        alertas.append("Egresos igualan o superan ingresos — ingreso neto no disponible")
        aprobado = False

    estado["resultado_financiero"] = {
        "aprobado": aprobado,
        "ingreso_neto": ingreso_neto,
        "pago_buro": round(pago_buro, 2),
        "total_compromisos": round(total_compromisos, 2),
        "ratio_cuota_ingreso": round(ratio_cuota_ingreso, 1),
        "capacidad_pago": round(capacidad_pago, 2),
        "ratio_monto_ingreso": round(ratio_monto_ingreso, 2),
        "monto_solicitado": monto,
        "monto_aprobado": monto_ajustado,
        "cuota_final": cuota_ajustada,
        "monto_reducido": monto_reducido,
        "alertas": alertas
    }
    return estado

# ─────────────────────────────────────────
# AGENTE 3: Análisis de Buró
# ─────────────────────────────────────────
def agente_buro(estado: EstadoSolicitud) -> EstadoSolicitud:
    s = estado["solicitud"]
    buro = s["buro"]

    alertas = []
    aprobado = True

    tipo_score = buro.get("tipo_score", "")
    score, icc_parsed = _parsear_score_buro(buro.get("score"))
    # Si ICC no viene explícito, usar el extraído del formato combinado
    icc = buro.get("icc") if buro.get("icc") is not None else icc_parsed

    creditos_activos = buro.get("creditos_activos", 0)
    creditos_cerrados = buro.get("creditos_cerrados", 0)
    total_creditos = creditos_activos + creditos_cerrados

    # ─── Códigos de exclusión BC Score (valores negativos — no son puntaje) ───
    # Fuente: Manual BC Score Buró de Crédito, sección de códigos de exclusión
    CODIGOS_EXCLUSION = {
        -1: "Consumidor fallecido",
        -5: "Todas las cuentas cerradas con al menos un MOP ≥ 90 días",
        -6: "Cuentas con antigüedad < 6 meses y al menos un MOP ≥ 03",
        -7: "Cuentas con antigüedad < 6 meses y al menos un MOP ≥ 02",
        -8: "Sin cuenta actualizada en el último año o sin antigüedad mínima de 6 meses",
        -9: "Sin cuentas para calcular BC Score",
    }
    try:
        es_codigo_exclusion = score is not None and int(score) in CODIGOS_EXCLUSION
    except (ValueError, TypeError):
        es_codigo_exclusion = False
    if es_codigo_exclusion:
        codigo = int(score)
        desc = CODIGOS_EXCLUSION[codigo]
        alertas.append(f"Código de exclusión BC Score {codigo}: {desc}")
        if codigo == -1:
            aprobado = False  # fallecido: rechazo inmediato
        # Los demás son informativos — situación común en SOFIPO (clientes sin historial previo)

    # ─── Validación: BC SCORE sin cuentas reportadas ───
    reporte_incompleto = (
        not es_codigo_exclusion
        and tipo_score and "BC SCORE" in tipo_score.upper()
        and total_creditos == 0
    )
    if reporte_incompleto:
        alertas.append(
            "⚠️ REPORTE INCOMPLETO: BC SCORE presente pero sin cuentas reportadas — "
            "score no confiable, escalar para consulta manual de buró"
        )
        aprobado = False

    # ─── Juicios ───
    if buro.get("tiene_juicios"):
        alertas.append("Juicios activos en buró — escalar a ejecutivo")
        aprobado = False

    # ─── Créditos vencidos ───
    if buro.get("creditos_vencidos", 0) > 0:
        alertas.append(f"{buro['creditos_vencidos']} crédito(s) vencido(s)")
        aprobado = False

    # ─── Atraso ───
    if buro.get("peor_atraso_dias", 0) > 90:
        alertas.append(f"Peor atraso: {buro['peor_atraso_dias']} días")
        aprobado = False

    # ─── Saldo vencido ───
    if buro.get("saldo_vencido", 0) > 0:
        alertas.append(f"Saldo vencido en buró: ${buro['saldo_vencido']:,.2f}")
        aprobado = False

    # ─── Alertas HAWK ───
    alertas_hawk = buro.get("alertas_hawk", [])
    if alertas_hawk:
        alertas.append(f"Alertas HAWK: {', '.join(alertas_hawk)}")

    # ─── Evaluación del score ───
    if es_codigo_exclusion:
        pass  # ya se informó arriba, no evaluar como score numérico
    elif score is not None:
        if score < 400:
            alertas.append(f"Score {tipo_score} muy bajo: {score} — alto riesgo")
            aprobado = False
        elif score < 550:
            alertas.append(f"Score {tipo_score} bajo: {score} — revisar con ejecutivo")
            # No rechaza automáticamente, escala
    else:
        alertas.append("Sin score de buró — cliente sin historial crediticio")

    estado["resultado_buro"] = {
        "aprobado": aprobado,
        "score": score,
        "icc": icc,
        "tipo_score": tipo_score,
        "causas_score": buro.get("causas_score", []),
        "total_creditos": total_creditos,
        "creditos_activos": creditos_activos,
        "creditos_cerrados": creditos_cerrados,
        "reporte_incompleto": reporte_incompleto,
        "alertas": alertas,
        "tiene_juicios":      buro.get("tiene_juicios", False),
        "creditos_vencidos":  buro.get("creditos_vencidos", 0),
        "saldo_vencido":      buro.get("saldo_vencido", 0) or 0,
        "peor_atraso_dias":   buro.get("peor_atraso_dias", 0),
    }
    return estado


# ─────────────────────────────────────────
# AGENTE 3B: Lector Completo de Buró (qwen2.5:14b)
# Lee TODO el historial y produce un informe narrativo detallado
# que ve lo que los guardrails no pueden ver.
# ─────────────────────────────────────────
PROMPT_LECTOR_BURO = PromptTemplate(
    input_variables=["score_info", "detalle_completo", "ingreso_neto"],
    template="""Eres un analista experto en buró de crédito para SOFIPO mexicanas.
Lee el historial completo y genera un informe narrativo. Sé específico con los números del caso.

Usa exactamente estas secciones (sin JSON, texto libre):

## Score e ICC
Interpreta el BC Score y el ICC en el contexto de riesgo de una SOFIPO. Explica qué implican las causas del score.

## Línea de tiempo crediticia
Construye una narrativa cronológica: desde el crédito más antiguo al más reciente. ¿Cómo ha evolucionado el comportamiento del cliente a lo largo del tiempo?

## Análisis por tipo de acreedor
¿Qué mix de acreedores tiene (bancos, microfinancieras, SOFIPO)? ¿Qué dice esto sobre el perfil y experiencia del solicitante?

## Comportamiento de pago
Analiza los MOP registrados. ¿Hay tendencia positiva o negativa? ¿Algún patrón relevante?

## Capacidad de endeudamiento
Saldo vigente total vs ingreso neto mensual de ${ingreso_neto:,.0f}. ¿Cuántos meses de ingreso representa la deuda activa? ¿Es sostenible el nuevo crédito?

## Señales clave
Lista concreta de señales de alerta y puntos positivos específicos encontrados en el historial.

## Conclusión del analista
3-4 oraciones: ¿qué dice este historial sobre el riesgo crediticio real de esta persona?

---
{score_info}

{detalle_completo}"""
)


def agente_lector_buro(estado: EstadoSolicitud) -> EstadoSolicitud:
    """
    Lee TODO el buró — score, ICC, causas, línea de tiempo, detalle cuenta por cuenta —
    y produce un informe narrativo en texto libre. Usa qwen2.5:14b, con fallback a llm.
    Este análisis enriquece al deliberador y aparece en el reporte final.
    """
    s    = estado["solicitud"]
    buro = s["buro"]
    fin  = estado["resultado_financiero"]
    rb   = estado["resultado_buro"]

    cuentas  = buro.get("cuentas", [])
    abiertas = [c for c in cuentas if c.get("estado") == "ABIERTA"]
    cerradas = [c for c in cuentas if c.get("estado") == "CERRADA"]

    if not cuentas:
        estado["analisis_buro_completo"] = (
            "Sin cuentas registradas en el buró — no es posible generar análisis narrativo. "
            "Se requiere validación manual del historial crediticio."
        )
        return estado

    # ── Armar detalle completo ──
    def _fmt_cuenta(c, idx):
        estado_c = c.get("estado", "")
        apertura = c.get("fecha_apertura", "s/f")
        cierre   = c.get("fecha_cierre", "")
        periodo  = f"{apertura} → {cierre}" if cierre else f"desde {apertura} (vigente)"
        return (
            f"  [{idx}] {c.get('tipo_credito','?')} | {c.get('otorgante','?')} | {estado_c}\n"
            f"      Periodo: {periodo}\n"
            f"      Crédito máx: ${c.get('credito_maximo',0):,.0f} | "
            f"Saldo actual: ${c.get('saldo_actual',0):,.0f} | "
            f"Saldo vencido: ${c.get('saldo_vencido',0):,.0f}\n"
            f"      MOP actual: {c.get('mop_actual',0)} | Peor MOP: {c.get('peor_mop',0)} | "
            f"Pagos puntuales: {c.get('pagos_puntuales',0)} | Con atraso: {c.get('pagos_atrasados',0)}\n"
        )

    # Ordenar por fecha de apertura para línea de tiempo
    def _sort_fecha(c):
        f = c.get("fecha_apertura") or "9999"
        return f

    todas_ordenadas = sorted(cuentas, key=_sort_fecha)
    total_saldo     = sum(c.get("saldo_actual", 0) for c in abiertas)
    total_pagos_p   = sum(c.get("pagos_puntuales", 0) for c in cuentas)
    total_pagos_a   = sum(c.get("pagos_atrasados", 0) for c in cuentas)
    peor_mop_g      = max((c.get("peor_mop", 0) for c in cuentas), default=0)

    score_info = (
        f"BC Score: {rb.get('score','N/A')} | Tipo: {rb.get('tipo_score','N/A')} | "
        f"ICC: {rb.get('icc','N/A')}/9 | "
        f"Causas del score: {', '.join(buro.get('causas_score',[]) or ['No especificadas'])}\n"
        f"HAWK: {', '.join(buro.get('alertas_hawk',[]) or ['Sin alertas'])}\n"
        f"Juicios: {'Sí' if buro.get('tiene_juicios') else 'No'} | "
        f"Créditos vencidos: {buro.get('creditos_vencidos',0)} | "
        f"Saldo vencido total: ${buro.get('saldo_vencido',0):,.0f}"
    )

    detalle = (
        f"RESUMEN: {len(cuentas)} cuentas ({len(abiertas)} abiertas, {len(cerradas)} cerradas) | "
        f"Saldo total vigente: ${total_saldo:,.0f} | "
        f"Pagos puntuales/atrasados: {total_pagos_p}/{total_pagos_a} | "
        f"Peor MOP histórico: {peor_mop_g}\n\n"
        f"HISTORIAL COMPLETO (cronológico):\n"
    )
    for i, c in enumerate(todas_ordenadas, 1):
        detalle += _fmt_cuenta(c, i)

    ingreso_neto = fin.get("ingreso_neto", 0) or 1

    try:
        chain = PROMPT_LECTOR_BURO | llm
        respuesta = chain.invoke({
            "score_info": score_info,
            "detalle_completo": detalle,
            "ingreso_neto": ingreso_neto,
        })
        analisis = _limpiar_deliberacion(respuesta)   # reutilizamos limpieza de think-tags
    except Exception as e:
        print(f"[AVISO] agente_lector_buro falló: {e}")
        analisis = f"Análisis narrativo no disponible ({e})."

    return {**estado, "analisis_buro_completo": analisis}


# ─────────────────────────────────────────
# AGENTE 3C: Análisis Profundo de Cuentas de Buró (métricas)
# ─────────────────────────────────────────
PROMPT_ANALISIS_BURO = PromptTemplate(
    input_variables=["cuentas_resumen", "score_info"],
    template="""Analiza este historial de buró de crédito. Responde SOLO el siguiente JSON, sin texto extra:

{{"comportamiento_pago":"<texto>","nivel_endeudamiento":"<bajo/medio/alto>","experiencia_crediticia":"<texto>","tendencia_reciente":"<texto>","areas_oportunidad":["<item1>","<item2>"],"fortalezas":["<item1>","<item2>"],"recomendacion_monto":"<texto>","resumen_ejecutivo":"<texto>"}}

Score: {score_info}

Datos: {cuentas_resumen}

JSON:"""
)

def agente_analisis_buro(estado: EstadoSolicitud) -> EstadoSolicitud:
    s = estado["solicitud"]
    buro = s["buro"]
    resultado_buro = estado["resultado_buro"]
    cuentas = buro.get("cuentas", [])

    # Si no hay cuentas, hacer análisis básico sin modelo
    if not cuentas:
        estado["analisis_buro"] = {
            "comportamiento_pago": "Sin cuentas reportadas — no es posible evaluar historial",
            "nivel_endeudamiento": "indeterminado",
            "experiencia_crediticia": "Sin historial registrado",
            "tendencia_reciente": "indeterminada",
            "areas_oportunidad": ["Consultar buró manualmente", "Validar identidad del solicitante"],
            "fortalezas": [],
            "recomendacion_monto": "No determinable sin historial",
            "resumen_ejecutivo": "No existen cuentas reportadas en el buró. Se requiere validación manual antes de tomar una decisión."
        }
        return estado

    # ── Métricas calculadas ──
    abiertas = [c for c in cuentas if c.get("estado") == "ABIERTA"]
    cerradas = [c for c in cuentas if c.get("estado") == "CERRADA"]

    total_saldo      = sum(c.get("saldo_actual", 0)    for c in abiertas)
    total_maximo     = sum(c.get("credito_maximo", 0)  for c in abiertas)
    total_saldo_max  = sum(c.get("credito_maximo", 0)  for c in cuentas)
    utilizacion      = (total_saldo / total_maximo * 100) if total_maximo > 0 else 0

    total_puntuales  = sum(c.get("pagos_puntuales", 0) for c in cuentas)
    total_atrasados  = sum(c.get("pagos_atrasados", 0) for c in cuentas)
    total_pagos      = total_puntuales + total_atrasados
    tasa_puntualidad = (total_puntuales / total_pagos * 100) if total_pagos > 0 else 0

    otorgantes       = sorted(set(c.get("otorgante", "") for c in cuentas))
    peor_mop         = max((c.get("peor_mop", 0) for c in cuentas), default=0)
    cuentas_con_atraso = [c for c in cuentas if c.get("peor_mop", 0) > 1]

    # ── Desglose por otorgante y tipo ──
    desglose_otorgante = {}
    for c in cuentas:
        ot = c.get("otorgante", "OTRO")
        if ot not in desglose_otorgante:
            desglose_otorgante[ot] = {"abiertas": 0, "cerradas": 0, "saldo": 0.0, "maximo": 0.0}
        if c.get("estado") == "ABIERTA":
            desglose_otorgante[ot]["abiertas"] += 1
            desglose_otorgante[ot]["saldo"] += c.get("saldo_actual", 0)
            desglose_otorgante[ot]["maximo"] += c.get("credito_maximo", 0)
        else:
            desglose_otorgante[ot]["cerradas"] += 1

    # ── Nota cuando el detalle es parcial ──
    total_reportado = buro.get("creditos_activos", 0) + buro.get("creditos_cerrados", 0)
    detalle_parcial = total_reportado > len(cuentas) and len(cuentas) > 0
    cuentas_no_mostradas = total_reportado - len(cuentas) if detalle_parcial else 0

    metricas = {
        "total_cuentas":            len(cuentas),
        "total_cuentas_reportadas": total_reportado,
        "detalle_parcial":          detalle_parcial,
        "cuentas_no_mostradas":     cuentas_no_mostradas,
        "cuentas_abiertas":         len(abiertas),
        "cuentas_cerradas":         len(cerradas),
        "cuentas_con_atraso":       len(cuentas_con_atraso),
        "saldo_actual_abiertas":    round(total_saldo, 2),
        "credito_maximo_abiertas":  round(total_maximo, 2),
        "credito_maximo_historico": round(total_saldo_max, 2),
        "utilizacion_pct":          round(utilizacion, 1),
        "pagos_puntuales":          total_puntuales,
        "pagos_atrasados":          total_atrasados,
        "tiene_datos_pagos":        total_pagos > 0,
        "tasa_puntualidad_pct":     round(tasa_puntualidad, 1) if total_pagos > 0 else None,
        "peor_mop_historico":       peor_mop,
        "otorgantes":               otorgantes,
        "desglose_otorgante":       desglose_otorgante,
        "saldo_vencido":            buro.get("saldo_vencido", 0),
    }

    # ── Texto para el modelo ──
    cuentas_resumen = f"""
Resumen cuantitativo:
- Total cuentas: {len(cuentas)} ({len(abiertas)} abiertas, {len(cerradas)} cerradas)
- Cuentas con algún atraso histórico: {len(cuentas_con_atraso)}
- Otorgantes: {', '.join(otorgantes)}
- Saldo actual (abiertas): ${total_saldo:,.2f} de ${total_maximo:,.2f} disponibles
- Utilización de crédito: {utilizacion:.1f}%
- Tasa de puntualidad: {tasa_puntualidad:.1f}% ({total_puntuales} pagos puntuales / {total_atrasados} con atraso)
- Peor MOP histórico: {peor_mop} (1=puntual, 2=1-29d, 3=30-59d, 4=60-89d, 5=90-119d, 7=cartera vencida)
- Saldo vencido: ${buro.get('saldo_vencido', 0):,.2f}

Detalle cuentas abiertas:
"""
    for c in abiertas:
        cuentas_resumen += (
            f"  [{c.get('numero')}] {c.get('tipo_credito')} | {c.get('otorgante')} | "
            f"Apertura: {c.get('fecha_apertura','')} | "
            f"Saldo: ${c.get('saldo_actual',0):,.2f} / Máx: ${c.get('credito_maximo',0):,.2f} | "
            f"MOP actual: {c.get('mop_actual',0)} | Peor MOP: {c.get('peor_mop',0)}\n"
        )

    cuentas_resumen += "\nDetalle cuentas cerradas:\n"
    for c in cerradas:
        cuentas_resumen += (
            f"  [{c.get('numero')}] {c.get('tipo_credito')} | {c.get('otorgante')} | "
            f"Apertura: {c.get('fecha_apertura','')} → Cierre: {c.get('fecha_cierre','')} | "
            f"Máx: ${c.get('credito_maximo',0):,.2f} | "
            f"Peor MOP: {c.get('peor_mop',0)} | "
            f"Puntuales: {c.get('pagos_puntuales',0)} / Atrasados: {c.get('pagos_atrasados',0)}\n"
        )

    score_info = (
        f"Tipo: {resultado_buro.get('tipo_score','N/A')} | "
        f"Score: {resultado_buro.get('score','N/A')} | "
        f"ICC: {resultado_buro.get('icc','N/A')} | "
        f"Causas: {', '.join(buro.get('causas_score',[])) or 'No especificadas'}"
    )

    chain = PROMPT_ANALISIS_BURO | llm
    respuesta = chain.invoke({
        "cuentas_resumen": cuentas_resumen,
        "score_info": score_info
    })

    analisis_ia = _parsear_json(respuesta, fallback={})

    if not analisis_ia:
        analisis_ia = {
            "comportamiento_pago": "Error al procesar análisis",
            "nivel_endeudamiento": "indeterminado",
            "experiencia_crediticia": f"{len(cuentas)} cuentas registradas",
            "tendencia_reciente": "indeterminada",
            "areas_oportunidad": [],
            "fortalezas": [],
            "recomendacion_monto": "Revisar manualmente",
            "resumen_ejecutivo": "No se pudo generar análisis automático."
        }

    estado["analisis_buro"] = {**metricas, **analisis_ia}
    return estado

# ─────────────────────────────────────────
# AGENTE 4: Decisión Final (con Ollama)
# ─────────────────────────────────────────

def _nivel_riesgo_decil(decil: int) -> str:
    if decil >= 7:   return "BAJO"
    elif decil >= 4: return "MEDIO"
    else:            return "ALTO"


def _calcular_score_cuantitativo(kyc: dict, fin: dict, buro: dict, ab: dict, ml: dict | None = None) -> tuple[int, dict]:
    """
    Calcula el score de riesgo cuantitativo (1-10) con 3 ó 4 factores ponderados.
    KYC no participa en el score crediticio (es verificación de identidad, no riesgo).
    Si hay resultado del modelo ML (decil), se incorpora como 4° factor con mayor peso.
    Retorna (score, desglose).
    """
    score_buro = buro.get("score")
    _EXCLUSION_SET = {-1, -5, -6, -7, -8, -9}
    es_exclusion = score_buro is not None and int(score_buro) in _EXCLUSION_SET

    # Escala: 10 = mejor perfil (menor riesgo) · 1 = peor perfil (mayor riesgo)

    # ── Factor 1: Score de buró ──
    # Rangos calibrados según tasas de morosidad del Manual BC Score (Buró de Crédito, jun 2021)
    if buro.get("tiene_juicios") or buro.get("creditos_vencidos", 0) > 0:
        f_buro = 1
    elif buro.get("reporte_incompleto"):
        f_buro = 4
    elif es_exclusion or score_buro is None:
        f_buro = 5  # sin historial — neutral, común en SOFIPO
    elif score_buro < 500:  f_buro = 1
    elif score_buro < 560:  f_buro = 2
    elif score_buro < 616:  f_buro = 4
    elif score_buro < 665:  f_buro = 6
    elif score_buro < 700:  f_buro = 7
    elif score_buro < 730:  f_buro = 9
    else:                   f_buro = 10

    # ── Factor 2: Cuota / Ingreso ──
    ratio = fin.get("ratio_cuota_ingreso", 0)
    if   ratio > 60: f_ci = 1
    elif ratio > 50: f_ci = 3
    elif ratio > 40: f_ci = 5
    elif ratio > 30: f_ci = 7
    elif ratio > 20: f_ci = 9
    else:            f_ci = 10

    # ── Factor 3: Señales de riesgo en buró ──
    alertas_hawk = [a for a in buro.get("alertas", []) if "HAWK" in a.upper()]
    peor_mop = ab.get("peor_mop_historico", 0)
    saldo_vencido = buro.get("saldo_vencido", 0) or ab.get("saldo_vencido", 0)

    if buro.get("tiene_juicios"):                     f_riesgo = 1
    elif buro.get("creditos_vencidos", 0) > 0:        f_riesgo = 2
    elif saldo_vencido > 0:                           f_riesgo = 3
    elif peor_mop >= 4:                               f_riesgo = 4
    elif peor_mop >= 3:                               f_riesgo = 5
    elif peor_mop >= 2:                               f_riesgo = 7
    elif alertas_hawk:                                f_riesgo = 8
    else:                                             f_riesgo = 10

    # ── Ajuste ICC (Índice de Capacidad Crediticia, Buró de Crédito) ──
    # ICC: tomar del campo explícito o extraer del score combinado "588/0004"
    _icc_raw_exp = buro.get("icc")
    _, _icc_del_score = _parsear_score_buro(buro.get("score"))
    icc_raw = _icc_raw_exp if _icc_raw_exp is not None else _icc_del_score
    icc_val = None
    if icc_raw is not None:
        try:
            icc_val = int(str(icc_raw).strip())
            if not (0 <= icc_val <= 9):
                icc_val = None
        except (ValueError, TypeError):
            icc_val = None

    if icc_val is not None:
        if icc_val <= 3:
            f_riesgo = max(1, f_riesgo - 2)
        elif icc_val >= 7:
            f_riesgo = min(10, f_riesgo + 1)

    # ── Factor 4: Modelo ML (cuando está disponible) ──
    decil = ml.get("valor_decil") if ml else None
    tiene_ml = decil is not None and isinstance(decil, int) and 1 <= decil <= 10

    if tiene_ml:
        f_ml = decil  # escala 1-10 donde 10=mejor cliente
        # ML lleva 30% — es el predictor más potente (AUC=0.85 vs combinación manual)
        score = f_buro * 0.30 + f_ci * 0.25 + f_riesgo * 0.15 + f_ml * 0.30
        pesos = {"score_buro": "30%", "cuota_ingreso": "25%",
                 "senales_riesgo": "15%", "modelo_ml": "30%"}
    else:
        f_ml = None
        # Sin ML: redistribuir entre los 3 factores disponibles
        score = f_buro * 0.45 + f_ci * 0.30 + f_riesgo * 0.25
        pesos = {"score_buro": "45%", "cuota_ingreso": "30%", "senales_riesgo": "25%"}

    score_redondeado = max(1, min(10, round(score)))

    desglose = {
        "score_buro":     f_buro,
        "cuota_ingreso":  f_ci,
        "senales_riesgo": f_riesgo,
        "modelo_ml":      f_ml,
        "tiene_ml":       tiene_ml,
        "pesos":          pesos,
    }
    return score_redondeado, desglose


_JERARQUIA = {"APROBADO": 0, "VALIDACION": 1, "ESCALAR_EJECUTIVO": 2, "RECHAZADO": 3}


def _calcular_ajuste_cualitativo(buro: dict, ab: dict, icc_val: int | None, fin: dict) -> tuple[int, str]:
    """
    Calcula el ajuste cualitativo de forma determinista (-1, 0, +1).
    El LLM no es confiable para este valor numérico — tiende a devolver 0 siempre.

    +1: perfil notablemente mejor de lo capturado por los factores cuantitativos
    -1: señales cualitativas de riesgo adicional no completamente reflejadas en el score
     0: sin señales destacadas en ninguna dirección
    """
    peor_mop      = ab.get("peor_mop_historico", 0) or 0
    nivel_endeu   = ab.get("nivel_endeudamiento", "medio") or "medio"
    total_cuentas = ab.get("total_cuentas", 0) or (
        (buro.get("creditos_activos", 0) or 0) + (buro.get("creditos_cerrados", 0) or 0)
    )
    tendencia     = ab.get("tendencia_reciente", "") or ""

    # ── Señales positivas (+1) ──
    # ICC alto + historial amplio y limpio + endeudamiento no alto
    icc_alto      = icc_val is not None and icc_val >= 7
    historial_amplio = total_cuentas >= 10
    pagos_limpios = peor_mop <= 1
    sin_endeu_alto = nivel_endeu in ("bajo", "medio")
    tendencia_pos = any(w in tendencia.lower() for w in ("positiva", "mejora", "estable"))

    if icc_alto and historial_amplio and pagos_limpios and sin_endeu_alto:
        return 1, f"ICC {icc_val}/9 alto con {total_cuentas} créditos historial, todos MOP≤1 y endeudamiento {nivel_endeu}"

    # ── Señales negativas (-1) ──
    # Tendencia reciente negativa, endeudamiento creciente, o ICC muy bajo con historial
    tendencia_neg = any(w in tendencia.lower() for w in ("negativa", "deterioro", "creciente", "empeora"))
    icc_muy_bajo  = icc_val is not None and icc_val <= 2
    saldo_vencido = (ab.get("saldo_vencido") or buro.get("saldo_vencido") or 0) > 0

    if tendencia_neg or icc_muy_bajo or saldo_vencido:
        razones = []
        if tendencia_neg:   razones.append(f"tendencia reciente: {tendencia}")
        if icc_muy_bajo:    razones.append(f"ICC {icc_val}/9 muy bajo")
        if saldo_vencido:   razones.append("saldo vencido presente")
        return -1, "; ".join(razones)

    return 0, "Perfil estándar sin señales cualitativas destacadas en ninguna dirección"

def _decision_minima(kyc: dict, fin: dict, buro: dict, ab: dict, score: int, ml: dict | None = None) -> str:
    """
    Reglas deterministas que definen la decisión mínima obligatoria.
    El LLM puede igualar o escalar, pero no puede ser más benévolo que esto.
    """
    decil       = (ml or {}).get("valor_decil")
    decision_ml = (ml or {}).get("decision_ml", "") or ""
    peor_mop    = ab.get("peor_mop_historico", 0)

    # ── Ratio deuda buró / ingreso neto mensual (meses de ingreso para cubrir deuda activa) ──
    ingreso_neto  = fin.get("ingreso_neto", 0) or 0
    saldo_buro    = ab.get("saldo_actual_abiertas", 0) or 0
    ratio_deuda   = (saldo_buro / ingreso_neto) if ingreso_neto > 0 else 999

    # ── RECHAZADO: condiciones inapelables ──
    # Juicio con cartera vencida o deuda aplastante (> 12 meses de ingreso) → rechazo
    if buro.get("tiene_juicios") and (buro.get("saldo_vencido", 0) > 0 or ratio_deuda > 12):
        return "RECHAZADO"
    if (buro.get("creditos_vencidos", 0) > 0
            or buro.get("reporte_incompleto")
            or score <= 2):
        return "RECHAZADO"
    # Score muy bajo o MOP grave: no tiene sentido mandar a mesa
    if score <= 3 or peor_mop >= 5:
        return "RECHAZADO"
    # ── ICC: parsear del campo explícito o del score combinado "588/0004" ──
    _icc_raw_exp = buro.get("icc")
    _, _icc_del_score = _parsear_score_buro(buro.get("score"))
    icc_raw = _icc_raw_exp if _icc_raw_exp is not None else _icc_del_score
    icc_val = None
    if icc_raw is not None:
        try:
            icc_val = int(str(icc_raw).strip())
            if not (0 <= icc_val <= 9):
                icc_val = None
        except (ValueError, TypeError):
            icc_val = None

    # Deciles 1-2: mora 41-63% — rechazo absoluto, no hay score que lo compense
    if decil is not None and decil <= 2:
        return "RECHAZADO"

    # ── ESCALAR_EJECUTIVO: señales graves ──
    if buro.get("tiene_juicios"):
        return "ESCALAR_EJECUTIVO"
    if (not fin["aprobado"]
            or peor_mop >= 4):
        return "ESCALAR_EJECUTIVO"

    # ── VALIDACION: el score combinado (que ya incluye el decil ML al 30%) ──
    # El decil ML NO es trigger autónomo — ya pondera al 30% en el score cuantitativo.
    # Usarlo además como trigger independiente sería doble penalización.
    # VALIDACION aplica cuando el score combinado es insuficiente o hay señales adicionales claras.
    _nivel = ab.get("nivel_endeudamiento", "")
    nivel_endeudamiento = _nivel if _nivel in ("bajo", "medio", "alto") else "medio"
    endeudamiento_alto = nivel_endeudamiento == "alto" and score <= 5
    icc_bajo_borderline = icc_val is not None and icc_val <= 3 and score <= 7
    if (score <= 5
            or endeudamiento_alto
            or peor_mop >= 3
            or icc_bajo_borderline):
        return "VALIDACION"

    # Thin file: score alto con historial delgado es poco confiable (aprendizaje caso #47184400)
    # BC Score y ICC se calculan con poca data → score inflado, riesgo real subestimado
    total_cuentas   = ab.get("total_cuentas", 0) or (buro.get("creditos_activos", 0) + buro.get("creditos_cerrados", 0))
    cuentas_abiertas = buro.get("creditos_activos", 0)
    if total_cuentas <= 4 and cuentas_abiertas <= 1:
        return "VALIDACION"

    return "APROBADO"


def _construir_resumen_caso(s, kyc, fin, buro, ab, ml, score_cuant, desglose) -> str:
    """Resumen estructurado compartido entre deliberador y decisor."""
    return f"""CONDICIONES DEL CRÉDITO:
- Producto: {s['condiciones']['producto']} | Tipo: {s['condiciones']['tipo']}
- Monto solicitado: ${s['condiciones']['monto']:,.2f} | Monto aprobado por capacidad: ${fin.get('monto_aprobado', s['condiciones']['monto']):,.2f}{' ⚠ REDUCIDO por capacidad de pago' if fin.get('monto_reducido') else ''}
- Plazo: {s['condiciones']['plazo']} pagos | Cuota: ${s['condiciones']['cuota']:,.2f} | Tasa: {s['condiciones']['tasa']}%
- Frecuencia: {s['condiciones']['frecuencia']}

CLIENTE:
- Negocio: {s['negocio']['nombre']} | Giro: {s['negocio']['giro']}
- Antigüedad negocio desde: {s['negocio']['antiguedad_fecha']}
- Dependientes: {s['cliente']['dependientes_economicos']}

RESULTADO KYC:
- Aprobado: {kyc['aprobado']}
- Score biométrico: {kyc['score_biometrico']}/100
- Alertas: {kyc['alertas'] or 'Ninguna'}

RESULTADO FINANCIERO:
- Aprobado: {fin['aprobado']}
- Ingreso neto mensual: ${fin['ingreso_neto']:,.2f}
- Pagos existentes en buró: ${fin.get('pago_buro', 0):,.2f}
- Total compromisos (cuota nueva + buró): ${fin.get('total_compromisos', 0):,.2f} ({fin['ratio_cuota_ingreso']}% del ingreso neto)
- Capacidad de pago remanente: ${fin['capacidad_pago']:,.2f}
{f"- MONTO REDUCIDO: sistema ajustó ${fin.get('monto_solicitado',0):,.2f} → ${fin.get('monto_aprobado',0):,.2f} por exceder 35% de carga sobre ingreso." if fin.get('monto_reducido') else ''}- Alertas: {fin['alertas'] or 'Ninguna'}

RESULTADO BURÓ:
- Aprobado: {buro['aprobado']}
- Score {buro.get('tipo_score','')}: {buro['score'] or 'Sin score'} | ICC: {buro.get('icc','N/A')}
- Reporte incompleto: {buro.get('reporte_incompleto', False)}
- Cuentas: {buro.get('total_creditos', 0)} ({buro.get('creditos_activos',0)} abiertas, {buro.get('creditos_cerrados',0)} cerradas)
- Alertas: {buro['alertas'] or 'Ninguna'}

ANÁLISIS PROFUNDO DE BURÓ:
- Comportamiento de pago: {ab.get('comportamiento_pago','')}
- Nivel de endeudamiento: {ab.get('nivel_endeudamiento','')}
- Experiencia crediticia: {ab.get('experiencia_crediticia','')}
- Tendencia reciente: {ab.get('tendencia_reciente','')}
- Fortalezas: {ab.get('fortalezas',[])}
- Áreas de oportunidad: {ab.get('areas_oportunidad',[])}
- Recomendación sobre monto: {ab.get('recomendacion_monto','')}

ICC: {_icc_resumen(buro.get('icc'))}

MODELO PREDICTIVO ML (XGBoost, AUC=0.85):
{f"- Decil: {ml.get('valor_decil')}/10 | Nivel de riesgo: {ml.get('nivel_riesgo','N/D')} | Prob. mora calibrada: {ml.get('probabilidad', 'N/D')}" if ml and ml.get('valor_decil') else "- No disponible (3 factores sin ML)"}

SCORE CUANTITATIVO: {score_cuant}/10
({' + '.join(f"{k}={v}×{desglose['pesos'][k]}" for k,v in desglose.items() if k not in ('pesos','tiene_ml') and v is not None)})"""


def _limpiar_deliberacion(texto: str) -> str:
    """Limpia la respuesta del deliberador: elimina bloques de razonamiento interno.
    Soporta DeepSeek R1 (<think>...</think>) y Gemma 4 (<|think|>...<|/think|>).
    """
    texto = re.sub(r'<think>[\s\S]*?</think>', '', texto)
    texto = re.sub(r'<\|think\|>[\s\S]*?<\|/think\|>', '', texto)
    return texto.strip()


# ─────────────────────────────────────────
# EQUIPO ANALISTA: 3 agentes especializados
# Cada uno tiene un prompt corto y focalizado.
# KYC se excluye — lo maneja el sistema por separado.
# ─────────────────────────────────────────

PROMPT_RIESGO_BURO = PromptTemplate(
    input_variables=["datos_buro"],
    template="""Eres un especialista en análisis de buró de crédito para SOFIPO mexicanas.
Analiza SOLO el comportamiento crediticio. En máximo 4 puntos concretos y específicos (con números del caso), responde:
¿Qué patrones de riesgo o fortaleza revela este historial que un score numérico NO captura?
Enfócate en: evolución temporal, mix de acreedores, tendencias de endeudamiento, señales de estrés o estabilidad.
Sé específico — no repitas el score ni datos obvios.

{datos_buro}"""
)

PROMPT_PERFIL_NEGOCIO = PromptTemplate(
    input_variables=["datos_negocio"],
    template="""Eres un especialista en microfinanzas productivas para SOFIPO mexicanas.
Analiza SOLO la viabilidad del negocio e ingreso. En máximo 4 puntos concretos, responde:
¿Es sostenible el flujo de caja de este negocio para absorber esta nueva deuda?
Enfócate en: antigüedad del negocio vs monto solicitado, relación ingreso/egresos/deuda total, señales de estacionalidad o vulnerabilidad del giro.
Sé específico con los números del caso.

{datos_negocio}"""
)

PROMPT_ALERTAS = PromptTemplate(
    input_variables=["datos_alertas"],
    template="""Eres un analista de riesgo de fraude y consistencia de datos para SOFIPO mexicanas.
Analiza SOLO inconsistencias, alertas y patrones inusuales. En máximo 4 puntos concretos, responde:
¿Qué señales de alerta, inconsistencias entre datos o patrones inusuales detectas en este caso?
Enfócate en: alertas HAWK, discrepancias entre datos declarados vs buró, comportamiento atípico, timing sospechoso de cuentas.
Si no hay alertas reales, dilo explícitamente en 1 línea.

{datos_alertas}"""
)


def agente_riesgo_buro(estado: EstadoSolicitud) -> EstadoSolicitud:
    """Especialista en patrones de buró — qué no captura el score."""
    s    = estado["solicitud"]
    buro = s["buro"]
    rb   = estado["resultado_buro"]
    ab   = estado["analisis_buro"]
    cuentas = buro.get("cuentas", [])

    abiertas = [c for c in cuentas if c.get("estado") == "ABIERTA"]
    cerradas = [c for c in cuentas if c.get("estado") == "CERRADA"]

    datos = f"""BC Score: {rb.get('score','N/A')} ({rb.get('tipo_score','')}) | ICC: {rb.get('icc','N/A')}/9
Total cuentas: {len(cuentas)} ({len(abiertas)} abiertas / {len(cerradas)} cerradas)
Puntualidad: {ab.get('tasa_puntualidad_pct','N/A')}% ({ab.get('pagos_puntuales',0)} puntuales / {ab.get('pagos_atrasados',0)} con atraso)
Peor MOP histórico: {ab.get('peor_mop_historico',0)} | Utilización crédito: {ab.get('utilizacion_pct',0)}%
Saldo vigente: ${ab.get('saldo_actual_abiertas',0):,.0f} | Saldo vencido: ${ab.get('saldo_vencido',0):,.0f}
Otorgantes: {', '.join(ab.get('otorgantes',[]))}
Nivel endeudamiento (LLM): {ab.get('nivel_endeudamiento','')}
Tendencia reciente (LLM): {ab.get('tendencia_reciente','')}
Causas del score: {', '.join(buro.get('causas_score',[]) or ['No especificadas'])}
Cuentas abiertas:"""

    for c in abiertas:
        datos += f"\n  • {c.get('tipo_credito','?')} {c.get('otorgante','?')} | Saldo ${c.get('saldo_actual',0):,.0f}/${c.get('credito_maximo',0):,.0f} | MOP actual: {c.get('mop_actual',0)} | Peor MOP: {c.get('peor_mop',0)}"

    if cerradas:
        datos += f"\nÚltimas cuentas cerradas ({min(3,len(cerradas))}):"
        for c in cerradas[:3]:
            datos += f"\n  • {c.get('tipo_credito','?')} {c.get('otorgante','?')} | Apertura: {c.get('fecha_apertura','')} | Peor MOP: {c.get('peor_mop',0)} | Pagos: {c.get('pagos_puntuales',0)}p/{c.get('pagos_atrasados',0)}a"

    try:
        chain = PROMPT_RIESGO_BURO | llm
        analisis = chain.invoke({"datos_buro": datos}).strip()
    except Exception as e:
        analisis = f"Análisis de riesgo de buró no disponible: {e}"

    return {**estado, "analisis_riesgo_buro": analisis}


def agente_perfil_negocio(estado: EstadoSolicitud) -> EstadoSolicitud:
    """Especialista en viabilidad del negocio e ingreso."""
    s    = estado["solicitud"]
    fin  = estado["resultado_financiero"]
    cond = s.get("condiciones", {})

    neg = s.get("negocio", {})
    cli = s.get("cliente", {})
    buro_raw = s.get("buro", {})

    datos = f"""Negocio: {neg.get('nombre','')} | Giro: {neg.get('giro','')}
Antigüedad del negocio desde: {neg.get('antiguedad_fecha','')}
Vivienda: {cli.get('tipo_vivienda','')} | Dependientes: {cli.get('dependientes_economicos',0)} | Antigüedad domicilio: {cli.get('antiguedad_domicilio_anios',0)} años

Ingresos mensuales: ${s.get('finanzas',{}).get('total_ingresos',0):,.0f}
Egresos mensuales: ${s.get('finanzas',{}).get('total_egresos',0):,.0f}
Ingreso neto: ${fin.get('ingreso_neto',0):,.0f}
Pago a realizar buró/mes: ${buro_raw.get('pago_a_realizar',0):,.0f}
Cuota nueva: ${cond.get('cuota',0):,.2f} | Ratio total compromisos/ingreso: {fin.get('ratio_cuota_ingreso',0):.1f}%
Capacidad de pago disponible: ${fin.get('capacidad_pago',0):,.0f}
Monto solicitado: ${cond.get('monto',0):,.0f} | Producto: {cond.get('producto','')} | Plazo: {cond.get('plazo','')} pagos"""

    try:
        chain = PROMPT_PERFIL_NEGOCIO | llm
        analisis = chain.invoke({"datos_negocio": datos}).strip()
    except Exception as e:
        analisis = f"Análisis de perfil de negocio no disponible: {e}"

    return {**estado, "analisis_perfil_negocio": analisis}


def agente_alertas(estado: EstadoSolicitud) -> EstadoSolicitud:
    """Especialista en inconsistencias, alertas y señales inusuales."""
    s    = estado["solicitud"]
    kyc  = estado["resultado_kyc"]
    buro = estado["resultado_buro"]
    ab   = estado["analisis_buro"]
    buro_raw = s.get("buro", {})
    cond = s.get("condiciones", {})

    cuentas  = buro_raw.get("cuentas", [])
    abiertas = [c for c in cuentas if c.get("estado") == "ABIERTA"]

    # Detectar cuentas recientes (últimos 6 meses desde apertura)
    cuentas_recientes = []
    for c in cuentas:
        f = c.get("fecha_apertura", "") or ""
        if "2025" in f or "2026" in f:
            cuentas_recientes.append(f"{c.get('tipo_credito','?')} {c.get('otorgante','?')} ({f})")

    datos = f"""ALERTAS HAWK: {', '.join(buro_raw.get('alertas_hawk',[]) or ['Ninguna'])}
Alertas buró: {', '.join(buro.get('alertas',[]) or ['Ninguna'])}
Tiene juicios: {buro_raw.get('tiene_juicios', False)}
Créditos vencidos: {buro_raw.get('creditos_vencidos', 0)}
Saldo vencido: ${buro_raw.get('saldo_vencido',0):,.0f}

Cuentas abiertas recientes (2025-2026): {', '.join(cuentas_recientes) if cuentas_recientes else 'Ninguna'}
Total cuentas reportadas: {buro_raw.get('creditos_activos',0)+buro_raw.get('creditos_cerrados',0)} vs cuentas con detalle: {len(cuentas)}
Causas del score: {', '.join(buro_raw.get('causas_score',[]) or ['No especificadas'])}

Monto solicitado: ${cond.get('monto',0):,.0f} | Saldo vigente total: ${ab.get('saldo_actual_abiertas',0):,.0f}
Ingreso neto: ${estado['resultado_financiero'].get('ingreso_neto',0):,.0f}
Utilización crédito: {ab.get('utilizacion_pct',0):.1f}%"""

    try:
        chain = PROMPT_ALERTAS | llm
        analisis = chain.invoke({"datos_alertas": datos}).strip()
    except Exception as e:
        analisis = f"Análisis de alertas no disponible: {e}"

    return {**estado, "analisis_alertas": analisis}


def agente_patrones(estado: EstadoSolicitud) -> EstadoSolicitud:
    """
    Busca casos históricos similares con feedback del analista.
    Si no hay suficientes casos, devuelve contexto vacío (no bloquea el pipeline).
    """
    if not _DB_DISPONIBLE:
        return {**estado, "patrones_historicos": ""}

    s    = estado["solicitud"]
    buro = s.get("buro", {})
    ab   = estado["analisis_buro"]
    fin  = estado["resultado_financiero"]
    ml   = s.get("modelo_ml") or {}

    # Calcular score actual para buscar similares
    kyc = estado["resultado_kyc"]
    rb  = estado["resultado_buro"]
    score_cuant, _ = _calcular_score_cuantitativo(kyc, fin, rb, ab, ml)

    bc_score_raw = buro.get("score")
    try:
        bc_score = float(bc_score_raw) if bc_score_raw else None
    except:
        bc_score = None

    dec_minima = _decision_minima(kyc, fin, rb, ab, score_cuant, ml)

    try:
        similares = buscar_casos_similares(bc_score, None, score_cuant, dec_minima, n=3)
    except Exception as e:
        print(f"[AVISO] agente_patrones falló: {e}")
        return {**estado, "patrones_historicos": ""}

    if not similares:
        return {**estado, "patrones_historicos": ""}

    lineas = ["CASOS HISTÓRICOS SIMILARES CON FEEDBACK DEL ANALISTA:"]
    for i, caso in enumerate(similares, 1):
        lineas.append(
            f"\nCaso {i} (folio {caso['folio']}, {caso['fecha'][:10]}):\n"
            f"  Score: {caso['score_sistema']}/10 | BC: {caso['bc_score']} | ICC: {caso['icc']} | "
            f"Endeudamiento: {caso['nivel_endeudamiento']} | Ratio: {caso['ratio_cuota_ingreso']:.1f}%\n"
            f"  Sistema decidió: {caso['decision_sistema']}\n"
            f"  Analista decidió: {caso['decision_analista']}\n"
            f"  Comentario: {caso['comentario_analista'] or 'Sin comentario'}"
        )

    return {**estado, "patrones_historicos": "\n".join(lineas)}


PROMPT_DELIBERADOR = PromptTemplate(
    input_variables=["decision_sistema", "score", "analisis_riesgo", "analisis_negocio",
                     "analisis_alertas", "patrones", "analisis_buro_narrativo"],
    template="""Eres el analista senior de crédito que sintetiza el trabajo del equipo para la mesa de crédito de una SOFIPO mexicana.

El motor determinista ya decidió: {decision_sistema} (Score {score}/10)
Tu trabajo NO es decidir — es entregar al analista de mesa los hallazgos que el motor no puede ver.

ANÁLISIS DEL EQUIPO:

[ESPECIALISTA BURÓ — patrones de comportamiento]
{analisis_riesgo}

[ESPECIALISTA NEGOCIO — viabilidad e ingreso]
{analisis_negocio}

[ESPECIALISTA ALERTAS — inconsistencias y señales]
{analisis_alertas}

{patrones}

{analisis_buro_narrativo}

Entrega al analista de mesa de crédito:

**SÍNTESIS:** En 2 oraciones, lo más relevante que el score no captura sobre este solicitante.

**LO QUE REFUERZA LA DECISIÓN {decision_sistema}:**
• [máximo 3 factores concretos con números]

**LO QUE EL ANALISTA DEBE VERIFICAR EN LA ENTREVISTA:**
• [máximo 3 preguntas o verificaciones específicas y accionables — no genéricas]

**RIESGO RESIDUAL NO CAPTURADO EN EL SCORE:**
[1-2 oraciones sobre lo que podría salir mal que las reglas no detectan]"""
)


# ─────────────────────────────────────────
# AGENTE 4b: Deliberador IA (análisis libre)
# ─────────────────────────────────────────
def agente_deliberador(estado: EstadoSolicitud) -> EstadoSolicitud:
    """
    Sintetiza el trabajo del equipo analista y entrega hallazgos concretos
    a la mesa de crédito. Conoce la decisión del sistema.
    """
    s    = estado["solicitud"]
    kyc  = estado["resultado_kyc"]
    fin  = estado["resultado_financiero"]
    buro = estado["resultado_buro"]
    ab   = estado["analisis_buro"]
    ml   = s.get("modelo_ml") or {}

    score_cuant, _ = _calcular_score_cuantitativo(kyc, fin, buro, ab, ml)
    dec_minima = _decision_minima(kyc, fin, buro, ab, score_cuant, ml)

    analisis_buro_narrativo = estado.get("analisis_buro_completo") or ""
    seccion_narrativo = f"[LECTURA INTEGRAL DEL BURÓ]\n{analisis_buro_narrativo}" if analisis_buro_narrativo else ""

    patrones = estado.get("patrones_historicos") or ""

    try:
        chain = PROMPT_DELIBERADOR | llm_razonador
        respuesta = chain.invoke({
            "decision_sistema":      dec_minima,
            "score":                 score_cuant,
            "analisis_riesgo":       estado.get("analisis_riesgo_buro") or "No disponible.",
            "analisis_negocio":      estado.get("analisis_perfil_negocio") or "No disponible.",
            "analisis_alertas":      estado.get("analisis_alertas") or "No disponible.",
            "patrones":              patrones,
            "analisis_buro_narrativo": seccion_narrativo,
        })
        deliberacion = _limpiar_deliberacion(respuesta)
    except Exception:
        try:
            chain = PROMPT_DELIBERADOR | llm
            respuesta = chain.invoke({
                "decision_sistema":      dec_minima,
                "score":                 score_cuant,
                "analisis_riesgo":       estado.get("analisis_riesgo_buro") or "No disponible.",
                "analisis_negocio":      estado.get("analisis_perfil_negocio") or "No disponible.",
                "analisis_alertas":      estado.get("analisis_alertas") or "No disponible.",
                "patrones":              patrones,
                "analisis_buro_narrativo": seccion_narrativo,
            })
            deliberacion = _limpiar_deliberacion(respuesta)
        except Exception as e:
            print(f"[AVISO] agente_deliberador falló: {e}")
            deliberacion = "Análisis del equipo no disponible."

    return {**estado, "deliberacion_ia": deliberacion}


def _generar_condiciones(decision: str, s: dict, fin: dict, buro: dict, ab: dict, ml: dict) -> str:
    """Genera las condiciones de la decisión de forma determinista."""
    cond = s.get("condiciones", {})
    if decision == "APROBADO":
        monto  = fin.get("monto_aprobado", cond.get("monto", 0))
        plazo  = cond.get("plazo", "")
        tasa   = cond.get("tasa", "")
        freq   = cond.get("frecuencia", "")
        cuota  = fin.get("cuota_final", cond.get("cuota", 0))
        reducido = fin.get("monto_reducido")
        base = f"Monto ${monto:,.0f} | Plazo {plazo} pagos | Cuota ${cuota:,.2f} | Tasa {tasa}% anual | {freq}"
        if reducido:
            base += f" ⚠ Monto ajustado de ${fin.get('monto_solicitado',0):,.0f} por capacidad de pago"
        return base

    if decision == "VALIDACION":
        puntos = []
        score_bc = buro.get("score") or 0
        icc_val  = _resolver_icc(buro.get("icc"), buro.get("score"))
        if icc_val is not None and icc_val <= 3:
            puntos.append(f"ICC {icc_val}/9 bajo — revisar comportamiento de pago reciente con el cliente")
        if ab.get("nivel_endeudamiento") == "alto":
            puntos.append("Endeudamiento alto — verificar capacidad real de pago con comprobantes")
        if ab.get("total_cuentas", 0) <= 4:
            puntos.append("Historial delgado — solicitar referencias o avales")
        if buro.get("alertas_hawk"):
            puntos.append(f"Alertas HAWK: {', '.join(buro.get('alertas_hawk', []))} — validar con expediente físico")
        if not puntos:
            puntos.append("Revisar expediente completo antes de autorizar")
        return " | ".join(puntos)

    if decision == "ESCALAR_EJECUTIVO":
        puntos = []
        if buro.get("tiene_juicios"):
            puntos.append("Juicios activos — ejecutivo evalúa si procede excepción")
        if not fin.get("aprobado"):
            puntos.append("Finanzas no aprobadas por motor — ejecutivo valida ingresos directamente")
        if ab.get("peor_mop_historico", 0) >= 4:
            puntos.append(f"Peor MOP {ab['peor_mop_historico']} — ejecutivo evalúa contexto del atraso")
        if not puntos:
            puntos.append("Caso requiere revisión de ejecutivo por señales de riesgo combinadas")
        return " | ".join(puntos)

    if decision == "RECHAZADO":
        puntos = []
        if buro.get("creditos_vencidos", 0) > 0:
            puntos.append("Liquidar créditos vencidos en buró")
        if buro.get("saldo_vencido", 0) > 0:
            puntos.append(f"Regularizar saldo vencido ${buro['saldo_vencido']:,.0f}")
        if buro.get("tiene_juicios"):
            puntos.append("Resolver juicios activos")
        if not puntos:
            puntos.append("Mejorar historial crediticio y solicitar en 6-12 meses")
        return "Para reconsideración futura: " + " | ".join(puntos)

    return ""


def _generar_razon(decision: str, buro: dict, ab: dict, fin: dict, score: int,
                   ml: dict, s: dict) -> str:
    """Genera la razón principal de la decisión de forma determinista."""
    decil    = (ml or {}).get("valor_decil")
    score_bc = buro.get("score") or 0
    icc_val  = _resolver_icc(buro.get("icc"), buro.get("score"))
    ingreso  = fin.get("ingreso_neto", 0) or 0
    ratio    = fin.get("ratio_cuota_ingreso", 0) or 0

    if decision == "RECHAZADO":
        return _razon_rechazo(buro, ab, score, decil)

    if decision == "ESCALAR_EJECUTIVO":
        if buro.get("tiene_juicios"):
            return f"Juicios activos en buró — requiere evaluación de ejecutivo para determinar si procede excepción"
        if not fin.get("aprobado"):
            return f"Finanzas rechazadas por el motor (compromisos {ratio:.1f}% del ingreso neto ${ingreso:,.0f}) — ejecutivo valida"
        return f"Señales de riesgo combinadas (score {score}/10) que requieren revisión ejecutiva"

    if decision == "VALIDACION":
        razones = []
        if icc_val is not None and icc_val <= 3:
            razones.append(f"ICC {icc_val}/9 bajo")
        if ab.get("nivel_endeudamiento") == "alto":
            razones.append("endeudamiento alto")
        if ab.get("total_cuentas", 0) <= 4:
            razones.append("historial delgado")
        if score <= 5:
            razones.append(f"score {score}/10 insuficiente para aprobación directa")
        causa = " + ".join(razones) if razones else f"score {score}/10 requiere validación"
        return f"Perfil en zona de revisión por: {causa} — mesa de crédito valida antes de autorizar"

    # APROBADO
    partes = [f"BC Score {score_bc}", f"ICC {icc_val}/9" if icc_val else "", f"score {score}/10",
              f"ratio cuota/ingreso {ratio:.1f}%"]
    partes = [p for p in partes if p]
    if decil:
        partes.append(f"decil ML {decil}/10")
    return f"Perfil aprobado: {' | '.join(partes)}"


def _generar_recomendacion(decision: str, buro: dict, ab: dict, fin: dict,
                           score: int, ml: dict, s: dict) -> str:
    """Genera la recomendación para el ejecutivo de forma determinista."""
    decil   = (ml or {}).get("valor_decil")
    icc_val = _resolver_icc(buro.get("icc"), buro.get("score"))
    ratio   = fin.get("ratio_cuota_ingreso", 0) or 0

    if decision == "APROBADO":
        tips = []
        if icc_val is not None and icc_val <= 4:
            tips.append(f"ICC {icc_val}/9 moderado — agendar seguimiento a 30 días del primer pago")
        if ratio > 25:
            tips.append(f"Ratio {ratio:.1f}% — asegurar que cliente comprende el compromiso de pago")
        if decil and decil <= 5:
            tips.append(f"Modelo ML decil {decil}/10 — monitorear comportamiento en los primeros 3 pagos")
        if fin.get("monto_reducido"):
            tips.append("Monto fue reducido por capacidad de pago — explicar al cliente el ajuste")
        return " | ".join(tips) if tips else "Expediente limpio. Proceder con formalización normal."

    if decision == "VALIDACION":
        return ("Mesa de crédito: validar comprobantes de ingreso y revisar contexto de las alertas antes de resolver. "
                f"Score {score}/10 — caso borderline que puede aprobarse con verificación adicional.")

    if decision == "ESCALAR_EJECUTIVO":
        return ("Escalar al ejecutivo de crédito con el expediente completo. "
                "El ejecutivo tiene facultad de aprobar excepciones documentadas.")

    if decision == "RECHAZADO":
        return ("Informar al cliente los motivos específicos del rechazo y las acciones que puede tomar "
                "para regularizar su situación crediticia antes de una nueva solicitud.")

    return "Revisar expediente con ejecutivo."

def _razon_rechazo(buro: dict, ab: dict, score: int, decil) -> str:
    """Genera una razón de rechazo coherente con la regla determinista que lo causó."""
    if buro.get("tiene_juicios") and buro.get("saldo_vencido", 0) > 0:
        return "Juicios activos en buró con saldo vencido — rechazo inapelable"
    if buro.get("creditos_vencidos", 0) > 0:
        return f"{buro['creditos_vencidos']} crédito(s) vencido(s) en buró — rechazo inapelable"
    if buro.get("reporte_incompleto"):
        return "Reporte de buró incompleto — score no confiable, se requiere consulta manual"
    if score <= 3:
        return f"Score de riesgo muy bajo ({score}/10) — perfil no viable para crédito"
    if ab.get("peor_mop_historico", 0) >= 5:
        return f"Peor MOP histórico {ab['peor_mop_historico']} (≥90 días de atraso) — rechazo por historial grave"
    if decil is not None and decil <= 1:
        return f"Modelo ML: decil {decil}/10 — riesgo extremadamente alto, rechazo directo"
    if decil is not None and decil == 2:
        score_bc = buro.get("score") or 0
        if score_bc < 650:
            return f"Modelo ML decil 2 + BC Score {score_bc} < 650 — perfil de riesgo alto sin compensación suficiente"
        return "Modelo ML decil 2 — capacidad de pago o ICC insuficientes para aprobar excepción"
    return "Rechazado por reglas deterministas del sistema de crédito"


def _resolver_icc(icc_raw, score_raw=None) -> int | None:
    """
    Extrae el valor numérico del ICC desde:
      - icc_raw: campo explícito (int, str como '4' o '0004')
      - score_raw: campo score que puede ser "588/0004" (combinado)
    Retorna int 0-9 o None si no disponible.
    """
    fuente = icc_raw
    if fuente is None and score_raw is not None and "/" in str(score_raw):
        try:
            fuente = str(score_raw).split("/", 1)[1].strip()
        except Exception:
            pass
    if fuente is None:
        return None
    try:
        v = int(str(fuente).strip())
        return v if 0 <= v <= 9 else None
    except (ValueError, TypeError):
        return None


def _icc_etiqueta(icc_raw, score_raw=None) -> str:
    """Etiqueta corta del ICC para el expediente."""
    v = _resolver_icc(icc_raw, score_raw)
    if v is None:
        return "N/A"
    nivel = "BAJO" if v <= 3 else ("MEDIO" if v <= 6 else "ALTO")
    return f"{v}/9 ({nivel})"


def _icc_resumen(icc_raw, score_raw=None) -> str:
    """Genera descripción del ICC para incluir en el resumen del LLM."""
    v = _resolver_icc(icc_raw, score_raw)
    if v is None:
        return "- No disponible"
    if v <= 3:
        nivel = "BAJO — alta sensibilidad a nueva deuda, mayor riesgo de deterioro"
        recomendacion = "Aprobar monto reducido (50-70% del solicitado) o solicitar garantía adicional"
    elif v <= 6:
        nivel = "MEDIO — sensibilidad moderada a nueva deuda"
        recomendacion = "Aprobar monto solicitado con seguimiento"
    else:
        nivel = "ALTO — baja sensibilidad a nueva deuda, perfil estable"
        recomendacion = "Puede considerarse monto completo o mayor al solicitado"
    return f"- ICC: {v}/9 — Capacidad crediticia {nivel}\n  Recomendación de monto: {recomendacion}"


def agente_modelo_ml(estado: EstadoSolicitud) -> EstadoSolicitud:
    """
    Agente ML — XGBoost créditos nuevos.
    Asigna un decil de riesgo relativo (1=mayor riesgo, 10=menor riesgo)
    basado en la distribución de entrenamiento OOF.
    Se usa el decil (no la probabilidad) para evitar el efecto del shift de mora.
    Si el modelo no está disponible o falla, el estado pasa sin modelo_ml.
    """
    if not _ML_DISPONIBLE:
        return estado

    s = estado["solicitud"]

    # Solo aplica a tipo NUEVO — el modelo fue entrenado únicamente en primeras solicitudes.
    # RENOVACION queda fuera de alcance del modelo hasta que se entrene con esa población.
    tipo = (s.get("condiciones") or {}).get("tipo", "")
    if "NUEVO" not in str(tipo).upper():
        return estado

    try:
        resultado = _predecir_desde_solicitud(s)
        decil        = resultado["decil"]
        nivel_riesgo = resultado["nivel_riesgo"]
        prob_raw     = resultado["probabilidad_raw"]
        prob_cal     = resultado["probabilidad"]

        # Inyectar en modelo_ml dentro de la solicitud
        s_actualizada = dict(s)
        s_actualizada["modelo_ml"] = {
            "valor_decil":      decil,
            "nivel_riesgo":     nivel_riesgo,
            "decision_ml":      _decision_desde_decil(decil),
            "probabilidad":     prob_cal,
            "probabilidad_raw": prob_raw,
        }
        return {**estado, "solicitud": s_actualizada}

    except Exception as e:
        print(f"[AVISO] agente_modelo_ml falló: {e}")
        return estado


def _predecir_desde_solicitud(s: dict) -> dict:
    """Extrae los campos de la solicitud y llama al predictor."""
    buro    = s.get('buro') or {}
    fin     = s.get('finanzas') or {}
    cliente = s.get('cliente') or {}
    cond    = s.get('condiciones') or {}

    import numpy as np
    def _icc(v):
        if v is None: return np.nan
        try: return float(str(v).strip())
        except: return np.nan

    caso = {
        'bc_score':          buro.get('score'),
        'icc':               _icc(buro.get('icc')),
        'total_ingresos':    fin.get('total_ingresos'),
        'total_egresos':     fin.get('total_egresos'),
        'creditos_vencidos': buro.get('creditos_vencidos'),
        'creditos_cerrados': buro.get('creditos_cerrados'),
        'creditos_activos':  buro.get('creditos_activos'),
        'saldo_actual':      buro.get('saldo_actual'),
        'monto_solicitado':  cond.get('monto'),
        'tipo_vivienda':     cliente.get('tipo_vivienda'),
        'estado_civil':      cliente.get('estado_civil'),
    }
    return _predecir_ml(s)


def _decision_desde_decil(decil: int) -> str:
    """
    Mapea el decil a decisión según tasa de mora real observada:
      Decil 1-2: mora 41-63% → Rechazar
      Decil 1-2: mora 47-63% → Rechazar
      Decil 3:   mora 32-47% → Escalar a ejecutivo
      Decil 4-6: mora 14-35% → Aceptada con seguimiento
      Decil 7:   mora  1-26% → Aceptada con seguimiento
      Decil 8-10: mora <2%   → Aceptada
    """
    if decil <= 2:
        return "Rechazar"
    elif decil == 3:
        return "Escalar a ejecutivo"
    elif decil <= 7:
        return "Aceptada con seguimiento"
    else:
        return "Aceptada"


def agente_decision(estado: EstadoSolicitud) -> EstadoSolicitud:
    s = estado["solicitud"]
    kyc = estado["resultado_kyc"]
    fin = estado["resultado_financiero"]
    buro = estado["resultado_buro"]
    ab = estado["analisis_buro"]

    # ── Score cuantitativo (determinista) ──
    ml = s.get("modelo_ml") or {}
    score_cuant, desglose = _calcular_score_cuantitativo(kyc, fin, buro, ab, ml)

    dec_minima = _decision_minima(kyc, fin, buro, ab, score_cuant, ml)

    # ── Ajuste cualitativo determinista (el LLM no es confiable para valores numéricos) ──
    _icc_raw_exp2 = buro.get("icc")
    _, _icc_del_score2 = _parsear_score_buro(buro.get("score"))
    _icc_raw2 = _icc_raw_exp2 if _icc_raw_exp2 is not None else _icc_del_score2
    _icc_val2 = None
    if _icc_raw2 is not None:
        try:
            v = int(str(_icc_raw2).strip())
            _icc_val2 = v if 0 <= v <= 9 else None
        except (ValueError, TypeError):
            pass
    ajuste, justificacion_ajuste = _calcular_ajuste_cualitativo(buro, ab, _icc_val2, fin)

    score_final = max(1, min(10, score_cuant + ajuste))

    # ── Todo determinista — sin LLM ──
    decision_efectiva   = dec_minima
    razon_principal     = _generar_razon(dec_minima, buro, ab, fin, score_cuant, ml, s)
    condiciones         = _generar_condiciones(dec_minima, s, fin, buro, ab, ml)
    recomendacion       = _generar_recomendacion(dec_minima, buro, ab, fin, score_cuant, ml, s)

    estado["decision_final"] = {
        "decision":               decision_efectiva,
        "razon_principal":        razon_principal,
        "condiciones":            condiciones,
        "recomendacion_ejecutivo": recomendacion,
        "score_riesgo":           score_final,
        "score_cuantitativo":     score_cuant,
        "ajuste_cualitativo":     ajuste,
        "justificacion_ajuste":   justificacion_ajuste,
        "desglose_score":         desglose,
    }
    return estado

# ─────────────────────────────────────────
# AGENTE 5: Generador de Expediente
# ─────────────────────────────────────────
def agente_expediente(estado: EstadoSolicitud) -> EstadoSolicitud:
    from datetime import datetime
    s = estado["solicitud"]
    decision = estado["decision_final"]
    kyc = estado["resultado_kyc"]
    fin = estado["resultado_financiero"]
    buro = estado["resultado_buro"]
    ab = estado["analisis_buro"]

    expediente = f"""
╔══════════════════════════════════════════════════════════════╗
║         EXPEDIENTE DE DECISIÓN CREDITICIA                    ║
║         Generado: {datetime.now().strftime('%d/%m/%Y %H:%M')}                        ║
╚══════════════════════════════════════════════════════════════╝

SOLICITUD: {s['condiciones']['numero_solicitud']}
CLIENTE:   {s['cliente']['nombre']}
RFC:       {s['cliente']['rfc']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONDICIONES SOLICITADAS
  Producto:    {s['condiciones']['producto']}
  Monto:       ${s['condiciones']['monto']:,.2f}
  Plazo:       {s['condiciones']['plazo']} pagos | {s['condiciones']['frecuencia']}
  Cuota:       ${s['condiciones']['cuota']:,.2f}
  Tasa:        {s['condiciones']['tasa']}%

{'━'*62}

CONDICIONES FINALES AUTORIZADAS
{'  ⚠ MONTO REDUCIDO POR CAPACIDAD DE PAGO' if fin.get('monto_reducido') else '  Sin ajustes — se aprueba el monto solicitado'}
  Producto:    {s['condiciones']['producto']}
  Monto:       ${fin.get('monto_aprobado', s['condiciones']['monto']):,.2f}
  Cuota:       ${fin.get('cuota_final', s['condiciones']['cuota']):,.2f}
  Plazo:       {s['condiciones']['plazo']} pagos | {s['condiciones']['frecuencia']}
  Tasa:        {s['condiciones']['tasa']}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RESULTADOS POR AGENTE

  KYC / IDENTIFICACIÓN
    Estado:           {'✅ APROBADO' if kyc['aprobado'] else '❌ RECHAZADO'}
    Score biométrico: {kyc['score_biometrico']}/100
    Alertas:          {', '.join(kyc['alertas']) if kyc['alertas'] else 'Ninguna'}

  ANÁLISIS FINANCIERO
    Estado:           {'✅ APROBADO' if fin['aprobado'] else '❌ RECHAZADO'}
    Ingreso neto:     ${fin['ingreso_neto']:,.2f}
    Pagos buró:       ${fin.get('pago_buro', 0):,.2f}
    Total compromisos:${fin.get('total_compromisos', fin.get('ingreso_neto',0) - fin.get('capacidad_pago',0)):,.2f}  (cuota nueva + pagos buró)
    Compromisos/Ing:  {fin['ratio_cuota_ingreso']}%
    Cap. de pago:     ${fin['capacidad_pago']:,.2f}
    Alertas:          {', '.join(fin['alertas']) if fin['alertas'] else 'Ninguna'}

  BURÓ DE CRÉDITO
    Estado:           {'✅ APROBADO' if buro['aprobado'] else '❌ RECHAZADO'}
    Score:            {buro.get('tipo_score','N/A')} {buro['score'] or 'Sin score'} | ICC: {_icc_etiqueta(buro.get('icc'))}
    Cuentas:          {buro.get('total_creditos',0)} total ({buro.get('creditos_activos',0)} abiertas / {buro.get('creditos_cerrados',0)} cerradas)
    Reporte:          {'⚠️ INCOMPLETO — score no confiable' if buro.get('reporte_incompleto') else '✅ Completo'}
    Causas score:     {', '.join(buro.get('causas_score',[])) or 'No especificadas'}
    Alertas:          {', '.join(buro['alertas']) if buro['alertas'] else 'Ninguna'}

  MODELO PREDICTIVO ML"""

    ml_data = s.get("modelo_ml") or {}
    if ml_data and ml_data.get("valor_decil") is not None:
        decil = ml_data["valor_decil"]
        nivel = ml_data.get("nivel_riesgo") or _nivel_riesgo_decil(decil)
        semaforo = "🟢" if nivel == "BAJO" else ("🟡" if nivel == "MEDIO" else "🔴")
        expediente += f"""
    Decil:            {semaforo} {decil}/10 — Nivel de riesgo: {nivel}
    Decisión ML:      {ml_data.get('decision_ml','N/D')}
    Capacidad pago:   ${ml_data.get('capacidad_pago_ml') or 0:,.2f}
    Fico Score:       {ml_data.get('fico_score') if ml_data.get('fico_score') != -10 else 'Sin hit (-10)'}
    Score No Hit:     {ml_data.get('score_no_hit') if ml_data.get('score_no_hit') != -10 else 'Sin hit (-10)'}
    VaNoHi:           {ml_data.get('va_no_hi') if ml_data.get('va_no_hi') != -10 else 'Sin hit (-10)'}"""
    else:
        expediente += "\n    No disponible"

    expediente += f"""

  MÉTRICAS DE BURÓ
    Cuentas reportadas:   {ab.get('total_cuentas_reportadas', ab.get('total_cuentas',0))} ({ab.get('cuentas_abiertas',0)} abiertas / {ab.get('cuentas_cerradas',0)} cerradas)
    Cuentas con atraso:   {ab.get('cuentas_con_atraso',0)}
    Saldo actual:         ${ab.get('saldo_actual_abiertas',0):,.2f} de ${ab.get('credito_maximo_abiertas',0):,.2f} disponibles
    Utilización:          {ab.get('utilizacion_pct',0)}%
    Pago a realizar:      ${s['buro'].get('pago_a_realizar',0):,.2f}  (pago mensual comprometido en buró)
    Puntualidad:          {'N/D — sin detalle de pagos por cuenta' if not ab.get('tiene_datos_pagos') else f"{ab.get('tasa_puntualidad_pct',0)}% ({ab.get('pagos_puntuales',0)} puntuales / {ab.get('pagos_atrasados',0)} con atraso)"}
    Peor MOP histórico:   {ab.get('peor_mop_historico',0)} (1=puntual · 2=<30d · 3=30-59d · 4=60-89d · 5=90-119d · 7=cartera vencida)
    Saldo vencido:        ${ab.get('saldo_vencido',0):,.2f}

  DESGLOSE POR OTORGANTE
    {'Otorgante':<20} {'Abiertas':>9} {'Cerradas':>9} {'Saldo actual':>14} {'Crédito máx':>13}
    {'-'*68}"""

    for ot, d in ab.get('desglose_otorgante', {}).items():
        expediente += f"\n    {ot:<20} {d['abiertas']:>9} {d['cerradas']:>9} ${d['saldo']:>13,.0f} ${d['maximo']:>12,.0f}"

    if ab.get('detalle_parcial'):
        expediente += f"\n    ⚠ Detalle parcial: se muestran {ab.get('total_cuentas',0)} de {ab.get('total_cuentas_reportadas',0)} cuentas reportadas"

    expediente += f"""

  ANÁLISIS PROFUNDO DE BURÓ
    Comportamiento:       {ab.get('comportamiento_pago','')}
    Endeudamiento:        {ab.get('nivel_endeudamiento','')}
    Experiencia:          {ab.get('experiencia_crediticia','')}
    Tendencia reciente:   {ab.get('tendencia_reciente','')}
    Fortalezas:
{chr(10).join('      • ' + f for f in ab.get('fortalezas',[])) or '      Ninguna'}
    Áreas de oportunidad:
{chr(10).join('      • ' + a for a in ab.get('areas_oportunidad',[])) or '      Ninguna'}
    Monto solicitado:     {ab.get('recomendacion_monto','')}
    Resumen ejecutivo:    {ab.get('resumen_ejecutivo','')}

  DETALLE DE CUENTAS
    {'#':<4} {'Tipo':<22} {'Otorgante':<17} {'Est':<7} {'Apertura':<13} {'Cierre':<13} {'Máximo':>12} {'Saldo':>12} {'MOP':>4} {'PeorMOP':>8}
    {'-'*120}"""

    cuentas = estado["solicitud"]["buro"].get("cuentas", [])
    for c in cuentas:
        cierre = c.get('fecha_cierre') or 'N/A'
        expediente += f"\n    {c.get('numero',0):<4} {c.get('tipo_credito','')[:21]:<22} {c.get('otorgante','')[:16]:<17} {c.get('estado','')[:6]:<7} {(c.get('fecha_apertura') or '')[:12]:<13} {cierre[:12]:<13} {c.get('credito_maximo',0):>12,.0f} {c.get('saldo_actual',0):>12,.0f} {c.get('mop_actual',0):>4} {c.get('peor_mop',0):>8}"

    expediente += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DECISIÓN DEL SISTEMA: {decision['decision']}
Score de riesgo:      {decision['score_riesgo']}/10

  DESGLOSE DEL SCORE
  {'Factor':<22} {'Puntaje':>8} {'Peso':>6}   Interpretación
  {'-'*65}"""

    ds = decision.get('desglose_score', {})
    labels = {
        'score_buro':     ('Score de buró',      '35%'),
        'cuota_ingreso':  ('Cuota / Ingreso',     '25%'),
        'kyc_biometrico': ('KYC biométrico',      '20%'),
        'senales_riesgo': ('Señales de riesgo',   '20%'),
    }
    for key, (nombre, peso) in labels.items():
        val = ds.get(key, '-')
        nivel = 'Bajo riesgo' if isinstance(val, int) and val >= 7 else ('Riesgo medio' if isinstance(val, int) and val >= 4 else 'Alto riesgo')
        expediente += f"\n  {nombre:<22} {str(val)+'/10':>8} {peso:>6}   {nivel}"

    expediente += f"""
  {'-'*65}
  Base cuantitativa:    {decision.get('score_cuantitativo','-')}/10
  Ajuste cualitativo:   {'+' if (decision.get('ajuste_cualitativo',0) or 0) > 0 else ''}{decision.get('ajuste_cualitativo', 0)}  ({decision.get('justificacion_ajuste','Sin ajuste')})
  Score final:          {decision['score_riesgo']}/10

Razón:                {decision['razon_principal']}
Condiciones:          {decision['condiciones'] or 'Sin condiciones adicionales'}

{('ACCIÓN PARA EJECUTIVO: ' + (decision.get('recomendacion_ejecutivo') or '')) if decision['decision'] in ('ESCALAR_EJECUTIVO', 'VALIDACION') and decision.get('recomendacion_ejecutivo') else ''}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  DECISIÓN SUJETA A VALIDACIÓN Y FIRMA DE EJECUTIVO
    Este análisis es una herramienta de apoyo — no reemplaza
    la decisión final del área de crédito.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    estado["expediente"] = expediente

    # Persistir en base de datos
    if _DB_DISPONIBLE:
        try:
            folio = s["condiciones"].get("numero_solicitud", "SIN_FOLIO")
            datos_para_db = {
                "buro": s.get("buro", {}),
                "analisis_buro": estado.get("analisis_buro", {}),
                "resultado_financiero": estado.get("resultado_financiero", {}),
                "condiciones": s.get("condiciones", {}),
                "modelo_ml": s.get("modelo_ml") or {},
            }
            guardar_caso(
                folio=folio,
                decision_sistema=estado["decision_final"].get("decision", ""),
                score_sistema=estado["decision_final"].get("score_riesgo", 0),
                datos=datos_para_db,
                resumen_caso=expediente[:2000],  # primeros 2000 chars del expediente
                deliberacion_ia=estado.get("deliberacion_ia", ""),
            )
        except Exception as e:
            print(f"[AVISO] No se pudo guardar caso en BD: {e}")

    return estado

# ─────────────────────────────────────────
# GRAFO DE AGENTES (LangGraph)
# ─────────────────────────────────────────
def construir_grafo():
    grafo = StateGraph(EstadoSolicitud)

    grafo.add_node("kyc",             agente_kyc)
    grafo.add_node("financiero",      agente_financiero)
    grafo.add_node("buro",            agente_buro)
    grafo.add_node("lector_buro",     agente_lector_buro)      # narrativa completa qwen2.5:14b
    grafo.add_node("analisis_buro",   agente_analisis_buro)    # métricas estructuradas
    grafo.add_node("modelo_ml",       agente_modelo_ml)
    grafo.add_node("riesgo_buro",     agente_riesgo_buro)      # especialista buró
    grafo.add_node("perfil_negocio",  agente_perfil_negocio)   # especialista negocio
    grafo.add_node("alertas",         agente_alertas)          # especialista alertas
    grafo.add_node("patrones",        agente_patrones)         # casos históricos similares
    grafo.add_node("deliberador",     agente_deliberador)      # síntesis analista senior
    grafo.add_node("decision",        agente_decision)
    grafo.add_node("expediente",      agente_expediente)

    grafo.set_entry_point("kyc")
    grafo.add_edge("kyc",            "financiero")
    grafo.add_edge("financiero",     "buro")
    grafo.add_edge("buro",           "lector_buro")
    grafo.add_edge("lector_buro",    "analisis_buro")
    grafo.add_edge("analisis_buro",  "modelo_ml")
    grafo.add_edge("modelo_ml",      "riesgo_buro")
    grafo.add_edge("riesgo_buro",    "perfil_negocio")
    grafo.add_edge("perfil_negocio", "alertas")
    grafo.add_edge("alertas",        "patrones")
    grafo.add_edge("patrones",       "deliberador")
    grafo.add_edge("deliberador",    "decision")
    grafo.add_edge("decision",       "expediente")
    grafo.add_edge("expediente",     END)

    return grafo.compile()


def analizar(solicitud: SolicitudCompleta) -> dict:
    grafo = construir_grafo()
    estado_inicial = EstadoSolicitud(
        solicitud=solicitud.model_dump(),
        resultado_kyc={},
        resultado_financiero={},
        resultado_buro={},
        analisis_buro={},
        analisis_buro_completo="",
        deliberacion_ia="",
        decision_final={},
        expediente="",
        analisis_riesgo_buro="",
        analisis_perfil_negocio="",
        analisis_alertas="",
        patrones_historicos="",
    )
    resultado = grafo.invoke(estado_inicial)
    return resultado
