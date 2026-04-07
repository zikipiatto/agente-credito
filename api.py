from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from modelos import SolicitudCompleta
from agentes import analizar
from parsear_buro import parsear_pdf_buro
from datetime import datetime
from pydantic import BaseModel as PydanticBaseModel

app = FastAPI(
    title="Motor de Decisión de Crédito — SOFIPO",
    description="Sistema multi-agente de análisis y decisión crediticia. Todo local, sin internet.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return {"status": "ok", "version": "2.0.0", "mensaje": "Motor de decisión activo"}

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.post("/parsear-buro")
async def parsear_buro_endpoint(archivo: UploadFile = File(...)):
    """
    Recibe un PDF de Buró de Crédito y devuelve los datos extraídos
    listos para pre-llenar el formulario de la solicitud.
    """
    if not archivo.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos PDF")
    try:
        contenido = await archivo.read()
        datos = parsear_pdf_buro(contenido)
        # Quitar el texto raw antes de devolver (puede ser muy grande)
        datos.pop("_texto_raw", None)
        return {"status": "ok", "datos": datos}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al parsear PDF: {e}")


@app.post("/analizar")
def analizar_solicitud(solicitud: SolicitudCompleta):
    try:
        resultado = analizar(solicitud)

        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),

            # ── Datos de la solicitud ──
            "solicitud": {
                "numero": solicitud.condiciones.numero_solicitud,
                "fecha": solicitud.condiciones.fecha,
                "cliente": solicitud.cliente.nombre,
                "rfc": solicitud.cliente.rfc,
                "producto": solicitud.condiciones.producto,
                "tipo": solicitud.condiciones.tipo,
                "monto": solicitud.condiciones.monto,
                "plazo": solicitud.condiciones.plazo,
                "cuota": solicitud.condiciones.cuota,
                "tasa": solicitud.condiciones.tasa,
                "frecuencia": solicitud.condiciones.frecuencia,
            },

            # ── Resultado Modelo ML ──
            # Leemos desde resultado["solicitud"]["modelo_ml"] porque el agente ML
            # inyecta el decil ahí durante el pipeline de LangGraph.
            # solicitud.modelo_ml (el objeto Pydantic original) siempre es None
            # porque el frontend ya no envía esos campos.
            "modelo_ml": (lambda ml: {
                "disponible":        ml is not None,
                "valor_decil":       ml.get("valor_decil")       if ml else None,
                "nivel_riesgo":      ml.get("nivel_riesgo")      if ml else None,
                "decision_ml":       ml.get("decision_ml")       if ml else None,
                "capacidad_pago_ml": ml.get("capacidad_pago_ml") if ml else None,
                "fico_score":        ml.get("fico_score")        if ml else None,
                "score_no_hit":      ml.get("score_no_hit")      if ml else None,
                "va_no_hi":          ml.get("va_no_hi")          if ml else None,
            })(resultado["solicitud"].get("modelo_ml")),

            # ── Resultado KYC ──
            "kyc": {
                "aprobado": resultado["resultado_kyc"]["aprobado"],
                "score_biometrico": resultado["resultado_kyc"]["score_biometrico"],
                "alertas": resultado["resultado_kyc"]["alertas"],
            },

            # ── Resultado Financiero ──
            "financiero": {
                "aprobado": resultado["resultado_financiero"]["aprobado"],
                "ingreso_mensual": solicitud.finanzas.total_ingresos,
                "egreso_mensual": solicitud.finanzas.total_egresos,
                "ingreso_neto": resultado["resultado_financiero"]["ingreso_neto"],
                "pago_buro": resultado["resultado_financiero"]["pago_buro"],
                "total_compromisos": resultado["resultado_financiero"]["total_compromisos"],
                "cuota_sobre_ingreso_pct": resultado["resultado_financiero"]["ratio_cuota_ingreso"],
                "capacidad_pago": resultado["resultado_financiero"]["capacidad_pago"],
                "monto_solicitado": resultado["resultado_financiero"]["monto_solicitado"],
                "monto_aprobado": resultado["resultado_financiero"]["monto_aprobado"],
                "monto_reducido": resultado["resultado_financiero"]["monto_reducido"],
                "alertas": resultado["resultado_financiero"]["alertas"],
            },

            # ── Resultado Buró ──
            "buro": {
                "aprobado": resultado["resultado_buro"]["aprobado"],
                "score": resultado["resultado_buro"]["score"],
                "icc": resultado["resultado_buro"]["icc"],
                "tipo_score": resultado["resultado_buro"]["tipo_score"],
                "causas_score": resultado["resultado_buro"]["causas_score"],
                "reporte_incompleto": resultado["resultado_buro"]["reporte_incompleto"],
                "total_cuentas": resultado["resultado_buro"]["total_creditos"],
                "cuentas_abiertas": resultado["resultado_buro"]["creditos_activos"],
                "cuentas_cerradas": resultado["resultado_buro"]["creditos_cerrados"],
                "alertas": resultado["resultado_buro"]["alertas"],
            },

            # ── Análisis profundo de buró ──
            "analisis_buro": {
                "metricas": {
                    "utilizacion_credito_pct":  resultado["analisis_buro"].get("utilizacion_pct"),
                    "tasa_puntualidad_pct":      resultado["analisis_buro"].get("tasa_puntualidad_pct"),
                    "pagos_puntuales":           resultado["analisis_buro"].get("pagos_puntuales"),
                    "pagos_atrasados":           resultado["analisis_buro"].get("pagos_atrasados"),
                    "peor_mop_historico":        resultado["analisis_buro"].get("peor_mop_historico"),
                    "cuentas_con_atraso":        resultado["analisis_buro"].get("cuentas_con_atraso"),
                    "saldo_actual_abiertas":     resultado["analisis_buro"].get("saldo_actual_abiertas"),
                    "credito_maximo_abiertas":   resultado["analisis_buro"].get("credito_maximo_abiertas"),
                    "credito_maximo_historico":  resultado["analisis_buro"].get("credito_maximo_historico"),
                    "saldo_vencido":             resultado["analisis_buro"].get("saldo_vencido"),
                    "otorgantes":                resultado["analisis_buro"].get("otorgantes"),
                },
                "interpretacion": {
                    "comportamiento_pago":    resultado["analisis_buro"].get("comportamiento_pago"),
                    "nivel_endeudamiento":    resultado["analisis_buro"].get("nivel_endeudamiento"),
                    "experiencia_crediticia": resultado["analisis_buro"].get("experiencia_crediticia"),
                    "tendencia_reciente":     resultado["analisis_buro"].get("tendencia_reciente"),
                    "fortalezas":             resultado["analisis_buro"].get("fortalezas", []),
                    "areas_oportunidad":      resultado["analisis_buro"].get("areas_oportunidad", []),
                    "recomendacion_monto":    resultado["analisis_buro"].get("recomendacion_monto"),
                    "resumen_ejecutivo":      resultado["analisis_buro"].get("resumen_ejecutivo"),
                },
                "detalle_cuentas": solicitud.buro.model_dump().get("cuentas", []),
            },

            # ── Condiciones finales del crédito (para el ejecutivo) ──
            "condiciones_finales": {
                "monto":      resultado["resultado_financiero"]["monto_aprobado"],
                "cuota":      resultado["resultado_financiero"]["cuota_final"],
                "plazo":      solicitud.condiciones.plazo,
                "tasa":       solicitud.condiciones.tasa,
                "frecuencia": solicitud.condiciones.frecuencia,
                "producto":   solicitud.condiciones.producto,
                "monto_reducido":   resultado["resultado_financiero"]["monto_reducido"],
                "monto_solicitado": resultado["resultado_financiero"]["monto_solicitado"],
            },

            # ── Decisión final ──
            "decision": {
                "resultado":              resultado["decision_final"]["decision"],
                "score_riesgo":           resultado["decision_final"]["score_riesgo"],
                "score_cuantitativo":     resultado["decision_final"]["score_cuantitativo"],
                "ajuste_cualitativo":     resultado["decision_final"]["ajuste_cualitativo"],
                "justificacion_ajuste":   resultado["decision_final"]["justificacion_ajuste"],
                "desglose_score":         resultado["decision_final"]["desglose_score"],
                "razon_principal":        resultado["decision_final"]["razon_principal"],
                "condiciones":            resultado["decision_final"].get("condiciones"),
                "recomendacion_ejecutivo": resultado["decision_final"].get("recomendacion_ejecutivo"),
            },

            # ── Análisis narrativo completo de buró (lector_buro qwen2.5:14b) ──
            "analisis_buro_completo": resultado.get("analisis_buro_completo", ""),

            # ── Deliberación IA (síntesis analista senior deepseek-r1:8b) ──
            "deliberacion_ia": resultado.get("deliberacion_ia", ""),

            # ── Análisis del equipo especializado ──
            "analisis_equipo": {
                "riesgo_buro":       resultado.get("analisis_riesgo_buro", ""),
                "perfil_negocio":    resultado.get("analisis_perfil_negocio", ""),
                "alertas":           resultado.get("analisis_alertas", ""),
                "patrones":          resultado.get("patrones_historicos", ""),
            },

            # ── Expediente texto (para imprimir o mostrar al ejecutivo) ──
            "expediente_texto": resultado["expediente"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class FeedbackAnalista(PydanticBaseModel):
    folio: str
    decision_analista: str   # APROBADO, VALIDACION, ESCALAR_EJECUTIVO, RECHAZADO
    comentario: str

@app.post("/feedback")
def registrar_feedback(feedback: FeedbackAnalista):
    """Registra la decisión y comentario del analista de mesa de crédito."""
    try:
        from base_datos import registrar_feedback as _reg
        _reg(feedback.folio, feedback.decision_analista, feedback.comentario)
        return {"status": "ok", "mensaje": "Feedback registrado correctamente"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/casos")
def listar_casos():
    """Lista los casos analizados con su feedback."""
    try:
        from base_datos import listar_casos as _listar
        return {"status": "ok", "casos": _listar()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
