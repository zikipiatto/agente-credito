"""
Evals del Motor de Decisión Crediticia
Evalúa: decisión correcta, razón coherente y monto razonable
Juez: Mistral local (sin rate limits, sin internet)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.models.base_model import DeepEvalBaseLLM
import ollama

from modelos import (
    SolicitudCompleta, Condiciones, Identificacion,
    DatosCliente, Negocio, IngresosEgresos, Buro, CuentaBuro
)
from agentes import analizar


# ── Juez Mistral local ──────────────────────────────────────────
class MistralJuez(DeepEvalBaseLLM):
    def __init__(self):
        self.model_name = "mistral:latest"

    def load_model(self):
        return self.model_name

    def generate(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model_name


# ── Dataset de casos de prueba ──────────────────────────────────
CASOS = [
    {
        "nombre": "Lourdes — juicios activos",
        "decision_esperada": "ESCALAR_EJECUTIVO",
        "razon_esperada": "Tiene juicios activos pero con deuda manejable, se escala a ejecutivo para revisión.",
        "solicitud": SolicitudCompleta(
            condiciones=Condiciones(
                numero_solicitud="47491944",
                fecha="20/03/2026",
                monto=10350.00,
                tipo="PRODUCTIVO",
                producto="Capital de Trabajo",
                tasa=131.0,
                cuota=2070.33,
                frecuencia="MENSUAL DE 28 DÍAS",
                plazo=8
            ),
            identificacion=Identificacion(
                verificacion_id=100,
                reconocimiento_facial=88,
                deteccion_vida=100,
                validacion_gobierno=0,
                video_selfie=0,
                distancia_km=0.0727
            ),
            cliente=DatosCliente(
                nombre="LOURDES PRISCILA PEREZ ORTIZ",
                rfc="PEOL861211333",
                curp="PEOL861211MDFRRR03",
                fecha_nacimiento="11/12/1986",
                genero="FEMENINO",
                estado_civil="SOLTERO(A)",
                tipo_vivienda="FAMILIAR",
                dependientes_economicos=1,
                antiguedad_domicilio_anios=30
            ),
            negocio=Negocio(nombre="VENTA DE ROPA", giro="COMERCIO", antiguedad_fecha="20/03/2020"),
            finanzas=IngresosEgresos(total_ingresos=30000, total_egresos=12000),
            buro=Buro(
                score=None,
                etiqueta_riesgo=None,
                alertas_hawk=["JUICIO AMPARO 2023", "JUICIO AMPARO 2024"],
                tiene_juicios=True,
                creditos_activos=1,
                creditos_vencidos=0,
                peor_atraso_dias=0
            )
        ),
    },
    {
        "nombre": "Cristel — perfil limpio",
        "decision_esperada": "APROBADO",
        "razon_esperada": "Perfil crediticio sólido, sin vencidos, buena capacidad de pago.",
        "solicitud": SolicitudCompleta(
            condiciones=Condiciones(
                numero_solicitud="47465130",
                fecha="20/03/2026",
                monto=15000.00,
                tipo="PRODUCTIVO",
                producto="Capital de Trabajo",
                tasa=131.0,
                cuota=2142.86,
                frecuencia="MENSUAL DE 28 DÍAS",
                plazo=7
            ),
            identificacion=Identificacion(
                verificacion_id=100,
                reconocimiento_facial=95,
                deteccion_vida=100,
                validacion_gobierno=100,
                video_selfie=90,
                distancia_km=0.05
            ),
            cliente=DatosCliente(
                nombre="CRISTEL MARTINEZ MARQUEZ",
                rfc="MAMC8203097X2",
                curp="MRMRCR82030921M000",
                fecha_nacimiento="09/03/1982",
                genero="FEMENINO",
                estado_civil="SOLTERO(A)",
                tipo_vivienda="FAMILIAR",
                dependientes_economicos=0,
                antiguedad_domicilio_anios=17
            ),
            negocio=Negocio(nombre="COLEGIO PAULO FREIRE", giro="SERVICIOS", antiguedad_fecha="30/04/2005"),
            finanzas=IngresosEgresos(total_ingresos=45000, total_egresos=18000),
            buro=Buro(
                score=662,
                icc="0005",
                tipo_score="BC SCORE",
                alertas_hawk=["TELEFONO NO CORRESPONDE A ZONA POSTAL"],
                tiene_juicios=False,
                creditos_activos=2,
                creditos_cerrados=33,
                creditos_vencidos=0,
                peor_atraso_dias=0,
                saldo_actual=173978.00,
                saldo_vencido=0.00,
                cuentas=[
                    CuentaBuro(numero=1, tipo_credito="PRÉSTAMO PERSONAL", otorgante="MICROFINANCIERA",
                               estado="ABIERTA", fecha_apertura="03-Sep-2025",
                               credito_maximo=35050.00, saldo_actual=10057.00, saldo_vencido=0.00,
                               mop_actual=1, peor_mop=1, pagos_puntuales=4, pagos_atrasados=0),
                    CuentaBuro(numero=2, tipo_credito="PRÉSTAMO PERSONAL", otorgante="BANCO",
                               estado="ABIERTA", fecha_apertura="26-Mar-2025",
                               credito_maximo=104832.00, saldo_actual=93408.00, saldo_vencido=0.00,
                               mop_actual=1, peor_mop=1, pagos_puntuales=8, pagos_atrasados=0),
                ]
            )
        ),
    },
    {
        "nombre": "Roberto — reporte de buró incompleto",
        "decision_esperada": "RECHAZADO",
        "razon_esperada": "El reporte de buró está incompleto: tiene score BC pero sin cuentas registradas.",
        "solicitud": SolicitudCompleta(
            condiciones=Condiciones(
                numero_solicitud="47019195",
                fecha="20/03/2026",
                monto=8000.00,
                tipo="PRODUCTIVO",
                producto="Capital de Trabajo",
                tasa=131.0,
                cuota=1333.33,
                frecuencia="MENSUAL DE 28 DÍAS",
                plazo=6
            ),
            identificacion=Identificacion(
                verificacion_id=100,
                reconocimiento_facial=91,
                deteccion_vida=100,
                validacion_gobierno=0,
                video_selfie=0,
                distancia_km=0.10
            ),
            cliente=DatosCliente(
                nombre="ROBERTO GARCIA LUNA",
                rfc="GALR900512AB3",
                curp="GALR900512HDFRCB04",
                fecha_nacimiento="12/05/1990",
                genero="MASCULINO",
                estado_civil="CASADO",
                tipo_vivienda="FAMILIAR",
                dependientes_economicos=1,
                antiguedad_domicilio_anios=5
            ),
            negocio=Negocio(nombre="TALLER MECANICO", giro="SERVICIOS", antiguedad_fecha="01/06/2019"),
            finanzas=IngresosEgresos(total_ingresos=22000, total_egresos=9000),
            buro=Buro(
                score=680,
                tipo_score="BC SCORE",
                etiqueta_riesgo=None,
                alertas_hawk=[],
                tiene_juicios=False,
                creditos_activos=0,
                creditos_cerrados=0,
                creditos_vencidos=0,
                peor_atraso_dias=0,
                saldo_actual=0.0,
                saldo_vencido=0.0
            )
        ),
    },
]


# ── Métricas ────────────────────────────────────────────────────
juez = MistralJuez()

coherencia_razon = GEval(
    name="Coherencia de razón",
    criteria=(
        "La razón principal explica de forma coherente y suficiente "
        "por qué se tomó la decisión crediticia indicada. "
        "Ignora el estilo o redacción; evalúa solo si la lógica es correcta."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=0.6,
    model=juez,
    async_mode=False,
)


# ── Ejecutar evals ──────────────────────────────────────────────
print("\n" + "=" * 65)
print("  EVALS — MOTOR DE DECISIÓN CREDITICIA")
print("=" * 65)

resultados = []

for caso in CASOS:
    print(f"\n📋 {caso['nombre']}")
    print("-" * 50)

    resultado = analizar(caso["solicitud"])
    decision_real = resultado["decision_final"].get("decision", "DESCONOCIDA")
    razon_real    = resultado["decision_final"].get("razon_principal", "")

    print(f"  Decisión esperada: {caso['decision_esperada']}")
    print(f"  Decisión real:     {decision_real}")

    # Métrica 1: decisión correcta (determinista, sin LLM)
    decision_ok = decision_real == caso["decision_esperada"]
    print(f"  Decisión:  {'✅ PASS' if decision_ok else '❌ FAIL'}")

    # Métrica 2: razón coherente (GEval con Mistral)
    tc = LLMTestCase(
        input=f"Decisión: {decision_real}",
        actual_output=razon_real,
        expected_output=caso["razon_esperada"],
    )
    coherencia_razon.measure(tc)
    razon_ok = coherencia_razon.is_successful()
    print(f"  Razón:     {'✅ PASS' if razon_ok else '❌ FAIL'} (score {coherencia_razon.score:.2f})")
    print(f"             {coherencia_razon.reason}")

    resultados.append({
        "nombre": caso["nombre"],
        "decision_ok": decision_ok,
        "razon_ok": razon_ok,
    })

# ── Resumen ─────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  RESUMEN")
print("=" * 65)
total = len(resultados)
dec_ok  = sum(1 for r in resultados if r["decision_ok"])
razon_ok = sum(1 for r in resultados if r["razon_ok"])
print(f"  Decisión correcta:  {dec_ok}/{total}")
print(f"  Razón coherente:    {razon_ok}/{total}")
print()
for r in resultados:
    d = "✅" if r["decision_ok"] else "❌"
    rz = "✅" if r["razon_ok"] else "❌"
    print(f"  {d} dec  {rz} razón  →  {r['nombre']}")
print()
