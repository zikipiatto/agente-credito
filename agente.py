from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel

# Modelo local via Ollama
llm = OllamaLLM(model="llama3.2", temperature=0.1)

# Prompt especializado en decisiones de crédito
PROMPT = PromptTemplate(
    input_variables=["solicitud"],
    template="""Eres un motor de decisión de crédito. Analiza la siguiente solicitud y responde ÚNICAMENTE en este formato JSON:

{{
  "decision": "APROBADO" | "RECHAZADO" | "ESCALAR_EJECUTIVO",
  "score_riesgo": 1-10,
  "razon": "explicación breve",
  "condiciones": "condiciones si aplica o null",
  "datos_faltantes": ["lista de datos que faltan"] o []
}}

Criterios:
- APROBADO: buenos ingresos, historial limpio, deuda/ingreso < 40%
- RECHAZADO: mal historial, deuda/ingreso > 60%, ingresos insuficientes
- ESCALAR_EJECUTIVO: caso borderline, datos incompletos o monto alto

Solicitud:
{solicitud}

Responde solo el JSON, sin texto adicional."""
)

class SolicitudCredito(BaseModel):
    nombre: str
    ingreso_mensual: float
    monto_solicitado: float
    plazo_meses: int
    historial_crediticio: str  # "limpio", "manchado", "sin historial"
    deudas_actuales: float = 0.0
    tipo_credito: str = "personal"
    observaciones: str = ""

def analizar_solicitud(solicitud: SolicitudCredito) -> dict:
    texto = f"""
    Nombre: {solicitud.nombre}
    Ingreso mensual: ${solicitud.ingreso_mensual:,.2f}
    Monto solicitado: ${solicitud.monto_solicitado:,.2f}
    Plazo: {solicitud.plazo_meses} meses
    Cuota estimada: ${solicitud.monto_solicitado / solicitud.plazo_meses:,.2f}/mes
    Historial crediticio: {solicitud.historial_crediticio}
    Deudas actuales: ${solicitud.deudas_actuales:,.2f}
    Tipo de crédito: {solicitud.tipo_credito}
    Relación deuda/ingreso: {((solicitud.deudas_actuales + solicitud.monto_solicitado / solicitud.plazo_meses) / solicitud.ingreso_mensual * 100):.1f}%
    Observaciones: {solicitud.observaciones or 'Ninguna'}
    """

    chain = PROMPT | llm
    respuesta = chain.invoke({"solicitud": texto})

    import json
    # Limpiar respuesta y parsear JSON
    respuesta_limpia = respuesta.strip()
    if "```" in respuesta_limpia:
        respuesta_limpia = respuesta_limpia.split("```")[1]
        if respuesta_limpia.startswith("json"):
            respuesta_limpia = respuesta_limpia[4:]

    return json.loads(respuesta_limpia)


if __name__ == "__main__":
    # Ejemplo de prueba
    solicitud = SolicitudCredito(
        nombre="Carlos Pérez",
        ingreso_mensual=25000,
        monto_solicitado=80000,
        plazo_meses=24,
        historial_crediticio="limpio",
        deudas_actuales=5000,
        tipo_credito="personal",
        observaciones="Trabaja en empresa estable hace 3 años"
    )

    print("Analizando solicitud...")
    resultado = analizar_solicitud(solicitud)

    import json
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
