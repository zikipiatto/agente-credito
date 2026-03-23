from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from modelos import SolicitudCompleta
from agentes import analizar
from datetime import datetime

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
                "cuota_sobre_ingreso_pct": resultado["resultado_financiero"]["ratio_cuota_ingreso"],
                "capacidad_pago": resultado["resultado_financiero"]["capacidad_pago"],
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

            # ── Expediente texto (para imprimir o mostrar al ejecutivo) ──
            "expediente_texto": resultado["expediente"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
