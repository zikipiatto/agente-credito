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

# Modelo local — sin internet
llm = OllamaLLM(model="mistral", temperature=0.1)

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
    decision_final: dict
    expediente: str

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

    # Validación gobierno y selfie: solo alerta informativa, no bloquea
    if ident["validacion_gobierno"] == 0 and ident["video_selfie"] == 0:
        alertas.append("Sin validación gobierno ni video selfie — revisar manualmente")

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

    ingreso_neto = ingresos - egresos
    ratio_cuota_ingreso = (cuota / ingresos * 100) if ingresos > 0 else 100
    capacidad_pago = ingreso_neto - cuota
    ratio_monto_ingreso = (monto / ingresos) if ingresos > 0 else 999

    alertas = []
    aprobado = True

    # Cuota/ingreso: alerta en >40%, rechaza solo si >60%
    if ratio_cuota_ingreso > 60:
        alertas.append(f"Cuota representa {ratio_cuota_ingreso:.1f}% del ingreso — capacidad de pago insuficiente")
        aprobado = False
    elif ratio_cuota_ingreso > 40:
        alertas.append(f"Cuota representa {ratio_cuota_ingreso:.1f}% del ingreso — revisar con ejecutivo")

    # Capacidad de pago negativa: rechaza
    if capacidad_pago < 0:
        alertas.append("Capacidad de pago negativa después de cuota")
        aprobado = False

    if ingresos <= 0:
        alertas.append("Sin ingresos declarados")
        aprobado = False

    estado["resultado_financiero"] = {
        "aprobado": aprobado,
        "ingreso_neto": ingreso_neto,
        "ratio_cuota_ingreso": round(ratio_cuota_ingreso, 1),
        "capacidad_pago": round(capacidad_pago, 2),
        "ratio_monto_ingreso": round(ratio_monto_ingreso, 2),
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
    score = buro.get("score")
    icc = buro.get("icc")
    creditos_activos = buro.get("creditos_activos", 0)
    creditos_cerrados = buro.get("creditos_cerrados", 0)
    total_creditos = creditos_activos + creditos_cerrados

    # ─── Códigos de exclusión: -8 y -9 ───
    # Significan que el cliente no tiene historial reportado en buró
    # No es negativo — simplemente no tiene historial. Se muestra el score pero se informa.
    CODIGOS_EXCLUSION = [-9, -8]
    es_codigo_exclusion = score is not None and int(score) in CODIGOS_EXCLUSION
    if es_codigo_exclusion:
        desc = {-9: "Sin historial crediticio registrado", -8: "Excluido de score por falta de historial"}
        alertas.append(f"Código de exclusión {int(score)}: {desc.get(int(score))} — primer crédito o historial insuficiente")
        # No rechaza, solo informa — es una situación normal en SOFIPO

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
        "alertas": alertas
    }
    return estado


# ─────────────────────────────────────────
# AGENTE 3B: Análisis Profundo de Cuentas de Buró
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
    Calcula el score de riesgo cuantitativo (1-10) con 4 ó 5 factores ponderados.
    Si hay resultado del modelo ML (decil), se incorpora como 5° factor.
    Retorna (score, desglose).
    """
    score_buro = buro.get("score")
    es_exclusion = score_buro is not None and int(score_buro) in [-8, -9]

    # Escala: 10 = mejor perfil (menor riesgo) · 1 = peor perfil (mayor riesgo)

    # ── Factor 1: Score de buró (35%) ──
    if buro.get("tiene_juicios") or buro.get("creditos_vencidos", 0) > 0:
        f_buro = 1
    elif buro.get("reporte_incompleto"):
        f_buro = 4
    elif es_exclusion or score_buro is None:
        f_buro = 5  # sin historial, neutral
    elif score_buro < 400:  f_buro = 1
    elif score_buro < 500:  f_buro = 3
    elif score_buro < 580:  f_buro = 5
    elif score_buro < 650:  f_buro = 7
    elif score_buro < 720:  f_buro = 9
    else:                   f_buro = 10

    # ── Factor 2: Cuota / Ingreso (25%) ──
    ratio = fin.get("ratio_cuota_ingreso", 0)
    if   ratio > 60: f_ci = 1
    elif ratio > 50: f_ci = 3
    elif ratio > 40: f_ci = 5
    elif ratio > 30: f_ci = 7
    elif ratio > 20: f_ci = 9
    else:            f_ci = 10

    # ── Factor 3: KYC biométrico (20%) ──
    bio = kyc.get("score_biometrico", 0)
    if   bio < 50: f_kyc = 1
    elif bio < 60: f_kyc = 4
    elif bio < 70: f_kyc = 6
    elif bio < 85: f_kyc = 8
    else:          f_kyc = 10

    # ── Factor 4: Señales de riesgo (20%) ──
    alertas_hawk = [a for a in buro.get("alertas", []) if "HAWK" in a.upper()]
    peor_mop = ab.get("peor_mop_historico", 0)
    saldo_vencido = buro.get("saldo_vencido", 0) or ab.get("saldo_vencido", 0)

    if buro.get("tiene_juicios"):                     f_riesgo = 1
    elif buro.get("creditos_vencidos", 0) > 0:        f_riesgo = 2
    elif saldo_vencido > 0:                           f_riesgo = 3
    elif peor_mop >= 5:                               f_riesgo = 4
    elif peor_mop >= 3:                               f_riesgo = 5
    elif peor_mop >= 2:                               f_riesgo = 7
    elif alertas_hawk:                                f_riesgo = 8
    else:                                             f_riesgo = 10

    # ── Factor 5: Decil del modelo ML (cuando está disponible) ──
    decil = ml.get("valor_decil") if ml else None
    tiene_ml = decil is not None and isinstance(decil, int) and 1 <= decil <= 10

    if tiene_ml:
        f_ml = decil  # ya está en escala 1-10 donde 10=mejor
        # Con ML: ajustar pesos para dar 20% al modelo
        score = (f_buro * 0.28 + f_ci * 0.22 + f_kyc * 0.15 +
                 f_riesgo * 0.15 + f_ml * 0.20)
        pesos = {"score_buro": "28%", "cuota_ingreso": "22%",
                 "kyc_biometrico": "15%", "senales_riesgo": "15%", "modelo_ml": "20%"}
    else:
        f_ml = None
        score = f_buro * 0.35 + f_ci * 0.25 + f_kyc * 0.20 + f_riesgo * 0.20
        pesos = {"score_buro": "35%", "cuota_ingreso": "25%",
                 "kyc_biometrico": "20%", "senales_riesgo": "20%"}

    score_redondeado = max(1, min(10, round(score)))

    desglose = {
        "score_buro":     f_buro,
        "cuota_ingreso":  f_ci,
        "kyc_biometrico": f_kyc,
        "senales_riesgo": f_riesgo,
        "modelo_ml":      f_ml,
        "tiene_ml":       tiene_ml,
        "pesos":          pesos,
    }
    return score_redondeado, desglose


PROMPT_DECISION = PromptTemplate(
    input_variables=["resumen"],
    template="""Eres un motor de decisión crediticia para una SOFIPO en México.
Analiza la solicitud y responde SOLO un objeto JSON con estos campos:
- decision: una de estas tres opciones exactas: APROBADO, RECHAZADO, ESCALAR_EJECUTIVO
- ajuste_cualitativo: +1 (perfil mejor de lo esperado, sube score), 0 (sin ajuste), o -1 (señales de riesgo adicionales, baja score)
- justificacion_ajuste: texto muy breve explicando el ajuste cualitativo
- razon_principal: texto breve explicando la decision
- condiciones: texto con condiciones si aplica, o null
- recomendacion_ejecutivo: texto si es ESCALAR_EJECUTIVO, o null

Reglas para decision:
- RECHAZADO si hay juicios, creditos vencidos, score bajo o reporte incompleto
- ESCALAR_EJECUTIVO si hay alertas menores, casos borderline o datos faltantes
- APROBADO solo si KYC, finanzas y buro estan todos aprobados sin alertas criticas

Reglas para ajuste_cualitativo:
- Usa -1 si el historial cualitativo es notablemente positivo (larga trayectoria, comportamiento ejemplar)
- Usa +1 si hay señales cualitativas de riesgo adicional (tendencia negativa reciente, endeudamiento creciente)
- Usa 0 en cualquier otro caso

Reglas para condiciones (IMPORTANTE, no dejar null si aplica):
- Si APROBADO: especifica monto maximo recomendado, plazo sugerido o garantias si aplica
- Si ESCALAR_EJECUTIVO: lista los puntos especificos que debe revisar el ejecutivo
- Si RECHAZADO: explica que condiciones tendria que cumplir para reconsiderar

Solicitud a analizar:
{resumen}

Responde solo el JSON:"""
)

def agente_decision(estado: EstadoSolicitud) -> EstadoSolicitud:
    s = estado["solicitud"]
    kyc = estado["resultado_kyc"]
    fin = estado["resultado_financiero"]
    buro = estado["resultado_buro"]
    ab = estado["analisis_buro"]

    # ── Score cuantitativo (determinista) ──
    ml = s.get("modelo_ml") or {}
    score_cuant, desglose = _calcular_score_cuantitativo(kyc, fin, buro, ab, ml)

    resumen = f"""
CONDICIONES DEL CRÉDITO:
- Producto: {s['condiciones']['producto']} | Tipo: {s['condiciones']['tipo']}
- Monto: ${s['condiciones']['monto']:,.2f} | Plazo: {s['condiciones']['plazo']} pagos
- Cuota: ${s['condiciones']['cuota']:,.2f} | Tasa: {s['condiciones']['tasa']}%
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
- Cuota/Ingreso: {fin['ratio_cuota_ingreso']}%
- Capacidad de pago después de cuota: ${fin['capacidad_pago']:,.2f}
- Alertas: {fin['alertas'] or 'Ninguna'}

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

MODELO PREDICTIVO ML:
{f"- Decil: {ml.get('valor_decil')}/10 (mayor = mejor cliente) | Nivel de riesgo: {ml.get('nivel_riesgo','N/D')} | Decisión ML: {ml.get('decision_ml','N/D')} | Cap. pago ML: ${ml.get('capacidad_pago_ml') or 0:,.2f}" if ml and ml.get('valor_decil') else "- No disponible (se usaron 4 factores sin ML)"}

SCORE CUANTITATIVO CALCULADO: {score_cuant}/10
({' + '.join(f"{k}={v}×{desglose['pesos'][k]}" for k,v in desglose.items() if k not in ('pesos','tiene_ml') and v is not None)})
"""

    chain = PROMPT_DECISION | llm
    respuesta = chain.invoke({"resumen": resumen})

    llm_result = _parsear_json(respuesta, fallback={
        "decision": "ESCALAR_EJECUTIVO",
        "ajuste_cualitativo": 0,
        "justificacion_ajuste": "No se pudo generar análisis cualitativo",
        "razon_principal": "No se pudo generar decisión automática — revisión manual requerida",
        "condiciones": None,
        "recomendacion_ejecutivo": "Revisar expediente manualmente con el ejecutivo de crédito."
    })

    # ── Ajuste cualitativo del LLM (acotado a -1, 0, +1) ──
    ajuste = llm_result.get("ajuste_cualitativo", 0)
    try:
        ajuste = max(-1, min(1, int(ajuste)))
    except (TypeError, ValueError):
        ajuste = 0

    score_final = max(1, min(10, score_cuant + ajuste))

    estado["decision_final"] = {
        **llm_result,
        "score_riesgo":         score_final,
        "score_cuantitativo":   score_cuant,
        "ajuste_cualitativo":   ajuste,
        "justificacion_ajuste": llm_result.get("justificacion_ajuste", ""),
        "desglose_score":       desglose,
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

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RESULTADOS POR AGENTE

  KYC / IDENTIFICACIÓN
    Estado:           {'✅ APROBADO' if kyc['aprobado'] else '❌ RECHAZADO'}
    Score biométrico: {kyc['score_biometrico']}/100
    Alertas:          {', '.join(kyc['alertas']) if kyc['alertas'] else 'Ninguna'}

  ANÁLISIS FINANCIERO
    Estado:           {'✅ APROBADO' if fin['aprobado'] else '❌ RECHAZADO'}
    Ingreso neto:     ${fin['ingreso_neto']:,.2f}
    Cuota/Ingreso:    {fin['ratio_cuota_ingreso']}%
    Cap. de pago:     ${fin['capacidad_pago']:,.2f}
    Alertas:          {', '.join(fin['alertas']) if fin['alertas'] else 'Ninguna'}

  BURÓ DE CRÉDITO
    Estado:           {'✅ APROBADO' if buro['aprobado'] else '❌ RECHAZADO'}
    Score:            {buro.get('tipo_score','N/A')} {buro['score'] or 'Sin score'} | ICC: {buro.get('icc','N/A')}
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

{'ACCIÓN PARA EJECUTIVO: ' + decision.get('recomendacion_ejecutivo', '') if decision['decision'] == 'ESCALAR_EJECUTIVO' else ''}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  DECISIÓN SUJETA A VALIDACIÓN Y FIRMA DE EJECUTIVO
    Este análisis es una herramienta de apoyo — no reemplaza
    la decisión final del área de crédito.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    estado["expediente"] = expediente
    return estado

# ─────────────────────────────────────────
# GRAFO DE AGENTES (LangGraph)
# ─────────────────────────────────────────
def construir_grafo():
    grafo = StateGraph(EstadoSolicitud)

    grafo.add_node("kyc", agente_kyc)
    grafo.add_node("financiero", agente_financiero)
    grafo.add_node("buro", agente_buro)
    grafo.add_node("analisis_buro", agente_analisis_buro)
    grafo.add_node("decision", agente_decision)
    grafo.add_node("expediente", agente_expediente)

    grafo.set_entry_point("kyc")
    grafo.add_edge("kyc", "financiero")
    grafo.add_edge("financiero", "buro")
    grafo.add_edge("buro", "analisis_buro")
    grafo.add_edge("analisis_buro", "decision")
    grafo.add_edge("decision", "expediente")
    grafo.add_edge("expediente", END)

    return grafo.compile()


def analizar(solicitud: SolicitudCompleta) -> dict:
    grafo = construir_grafo()
    estado_inicial = EstadoSolicitud(
        solicitud=solicitud.model_dump(),
        resultado_kyc={},
        resultado_financiero={},
        resultado_buro={},
        analisis_buro={},
        decision_final={},
        expediente=""
    )
    resultado = grafo.invoke(estado_inicial)
    return resultado
