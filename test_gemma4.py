"""
Comparación de modelos en el agente_deliberador:
  deepseek-r1:8b  (modelo anterior)
  gemma4:e4b      (modelo nuevo)

Usa el estado pre-construido de cada caso demo para evitar correr el
pipeline completo (3-5 min por caso). Solo evalúa el deliberador.
"""
import time
import re
from langchain_ollama import OllamaLLM

# ── Reutilizamos el prompt y la función de limpieza de agentes.py ──
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from agentes import PROMPT_DELIBERADOR, _limpiar_deliberacion


MODELOS = {
    "deepseek-r1:8b": OllamaLLM(model="deepseek-r1:8b", temperature=0.2),
    "gemma4:e4b":     OllamaLLM(model="gemma4:e4b",     temperature=1.0, top_p=0.95),
}

# ── Casos de prueba (contexto pre-construido para el deliberador) ──
CASOS = [
    {
        "nombre": "Cristel — APROBADO (perfil limpio)",
        "decision": "APROBADO",
        "score": 7,
        "analisis_riesgo": "ICC 5, 35 cuentas, historial limpio. Sin vencidos. BC Score 662. Comportamiento de pago puntual en todos los créditos activos.",
        "analisis_negocio": "Colegio PAULO FREIRE con 20 años de antigüedad. Ingresos $45,000, egresos $18,000. Cuota/ingreso 4.8%. Margen holgado.",
        "analisis_alertas": "Solo alerta telefónica de zona postal (HAWK nivel bajo). Sin juicios, sin señales de fraude, sin inconsistencias en KYC.",
    },
    {
        "nombre": "Jorge — RECHAZADO (mal buró + juicios)",
        "decision": "RECHAZADO",
        "score": 2,
        "analisis_riesgo": "3 juicios activos con saldo vencido $45,000. ICC 2 (muy bajo). MOP máximo 7 en 2 cuentas. Patrón de deterioro progresivo en los últimos 18 meses.",
        "analisis_negocio": "Ferretería 8 años. Ingresos $25,000, egresos $22,000. Margen neto $3,000. Cuota propuesta $4,200 — imposible cubrir.",
        "analisis_alertas": "HAWK con 3 juicios activos (civil y familiar). Endeudamiento creciente. Inconsistencia entre ingresos declarados y saldos de créditos activos.",
    },
    {
        "nombre": "María — ESCALAR_EJECUTIVO (sin capacidad de pago)",
        "decision": "ESCALAR_EJECUTIVO",
        "score": 5,
        "analisis_riesgo": "BC Score 640, ICC 4. Sin vencidos pero cuota/ingreso actual 48%. Historial de pagos puntual pero endeudamiento en el límite superior.",
        "analisis_negocio": "Papelería 5 años. Ingresos $12,000, egresos $9,000. Disponible $3,000. Cuota propuesta $2,400 — deja $600 de margen mensual.",
        "analisis_alertas": "Sin juicios. Alerta por endeudamiento alto. Domicilio y negocio distantes 15 km. Validación de gobierno pendiente.",
    },
    {
        "nombre": "Alejandra — VALIDACION (thin file)",
        "decision": "VALIDACION",
        "score": 6,
        "analisis_riesgo": "ICC 9 excelente, BC Score 702. Pero solo 3 cuentas abiertas, todas con menos de 1 año. Thin file: historial insuficiente para validar comportamiento a largo plazo.",
        "analisis_negocio": "Estética 3 años. Ingresos $18,000, egresos $7,000. Cuota/ingreso 11%. Buena capacidad de pago pero negocio joven.",
        "analisis_alertas": "Sin juicios ni HAWK graves. Negocio informal sin comprobantes fiscales. Requiere validación de ingresos y referencias comerciales.",
    },
]


def deliberar(modelo_llm, caso: dict) -> tuple[str, float]:
    """Ejecuta el deliberador y retorna (texto_limpio, segundos)."""
    t0 = time.time()
    chain = PROMPT_DELIBERADOR | modelo_llm
    respuesta = chain.invoke({
        "decision_sistema":      caso["decision"],
        "score":                 caso["score"],
        "analisis_riesgo":       caso["analisis_riesgo"],
        "analisis_negocio":      caso["analisis_negocio"],
        "analisis_alertas":      caso["analisis_alertas"],
        "patrones":              "Sin casos similares previos en la base de datos.",
        "analisis_buro_narrativo": "",
    })
    texto = _limpiar_deliberacion(respuesta)
    return texto, round(time.time() - t0, 1)


# ── Runner ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  COMPARACIÓN DELIBERADOR: deepseek-r1:8b  vs  gemma4:e4b")
print("=" * 70)

tiempos = {m: [] for m in MODELOS}

for caso in CASOS:
    print(f"\n{'─'*70}")
    print(f"📋 {caso['nombre']}  |  Decisión: {caso['decision']}  |  Score: {caso['score']}/10")
    print(f"{'─'*70}")

    for nombre_modelo, llm in MODELOS.items():
        print(f"\n🤖 {nombre_modelo}")
        try:
            texto, secs = deliberar(llm, caso)
            tiempos[nombre_modelo].append(secs)
            print(f"   ⏱  {secs}s")
            # Imprimir con indentación
            for linea in texto.strip().splitlines():
                print(f"   {linea}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

# ── Resumen de tiempos ────────────────────────────────────────────
print(f"\n{'='*70}")
print("  TIEMPOS PROMEDIO")
print(f"{'='*70}")
for nombre, ts in tiempos.items():
    if ts:
        print(f"  {nombre:<20} → {sum(ts)/len(ts):.1f}s promedio  (casos: {ts})")
print()
