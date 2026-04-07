"""
Evaluación rápida del ajuste_cualitativo para los 10 casos demo.
Corre sin API ni LLM — solo la lógica determinista.
Uso: python test_ajuste.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from agentes import _calcular_ajuste_cualitativo, _calcular_score_cuantitativo

# ── Casos demo (datos del HTML) ───────────────────────────────────────────────
# buro: score, icc, creditos_activos, creditos_cerrados, saldo_vencido, tiene_juicios, creditos_vencidos, alertas_hawk
# ab:   peor_mop_historico, nivel_endeudamiento, tendencia_reciente, total_cuentas, saldo_vencido
# fin:  total_ingresos, total_egresos, ratio_cuota_ingreso, aprobado
# ml:   valor_decil

CASOS = [
    {
        "nombre": "✅ Cristel Martinez",
        "buro": {"score": 662, "icc": "0005", "creditos_activos": 2, "creditos_cerrados": 33,
                 "saldo_vencido": 0, "tiene_juicios": False, "creditos_vencidos": 0,
                 "alertas": ["Alertas HAWK: TELEFONO NO CORRESPONDE A ZONA POSTAL"], "reporte_incompleto": False},
        "ab":   {"peor_mop_historico": 0, "nivel_endeudamiento": "alto", "tendencia_reciente": "estable",
                 "total_cuentas": 35, "saldo_vencido": 0, "saldo_actual_abiertas": 103465},
        "fin":  {"total_ingresos": 45000, "total_egresos": 18000, "ratio_cuota_ingreso": 33.4,
                 "aprobado": True, "ingreso_neto": 27000, "monto_aprobado": 15000, "monto_reducido": False},
        "ml":   {"valor_decil": 7},
        "kyc":  {"aprobado": True, "score_biometrico": 98.3, "alertas": []},
    },
    {
        "nombre": "❌ Jorge Ramirez (mal buró)",
        "buro": {"score": 350, "icc": "0003", "creditos_activos": 2, "creditos_cerrados": 5,
                 "saldo_vencido": 8500, "tiene_juicios": True, "creditos_vencidos": 2,
                 "alertas": [], "reporte_incompleto": False},
        "ab":   {"peor_mop_historico": 5, "nivel_endeudamiento": "alto", "tendencia_reciente": "negativa",
                 "total_cuentas": 7, "saldo_vencido": 8500, "saldo_actual_abiertas": 45000},
        "fin":  {"total_ingresos": 28000, "total_egresos": 14000, "ratio_cuota_ingreso": 20.0,
                 "aprobado": True, "ingreso_neto": 14000, "monto_aprobado": 12000, "monto_reducido": False},
        "ml":   {"valor_decil": 2},
        "kyc":  {"aprobado": True, "score_biometrico": 93.3, "alertas": []},
    },
    {
        "nombre": "❌ María López (sin capacidad)",
        "buro": {"score": 640, "icc": "0004", "creditos_activos": 1, "creditos_cerrados": 4,
                 "saldo_vencido": 0, "tiene_juicios": False, "creditos_vencidos": 0,
                 "alertas": [], "reporte_incompleto": False},
        "ab":   {"peor_mop_historico": 0, "nivel_endeudamiento": "alto", "tendencia_reciente": "estable",
                 "total_cuentas": 5, "saldo_vencido": 0, "saldo_actual_abiertas": 12000},
        "fin":  {"total_ingresos": 7500, "total_egresos": 6800, "ratio_cuota_ingreso": 92.0,
                 "aprobado": False, "ingreso_neto": 700, "monto_aprobado": 10000, "monto_reducido": False},
        "ml":   {"valor_decil": 4},
        "kyc":  {"aprobado": True, "score_biometrico": 95.0, "alertas": []},
    },
    {
        "nombre": "⚠️  Rafael Mendoza (decil bajo ML)",
        "buro": {"score": 620, "icc": "0007", "creditos_activos": 3, "creditos_cerrados": 8,
                 "saldo_vencido": 0, "tiene_juicios": False, "creditos_vencidos": 0,
                 "alertas": ["Alertas HAWK: TELEFONO NO CORRESPONDE A ZONA POSTAL"], "reporte_incompleto": False},
        "ab":   {"peor_mop_historico": 1, "nivel_endeudamiento": "alto", "tendencia_reciente": "creciente",
                 "total_cuentas": 11, "saldo_vencido": 0, "saldo_actual_abiertas": 68000},
        "fin":  {"total_ingresos": 32000, "total_egresos": 15000, "ratio_cuota_ingreso": 25.0,
                 "aprobado": True, "ingreso_neto": 17000, "monto_aprobado": 12000, "monto_reducido": False},
        "ml":   {"valor_decil": 2},
        "kyc":  {"aprobado": True, "score_biometrico": 92.0, "alertas": []},
    },
    {
        "nombre": "⚠️  Ana Flores (buró vacío)",
        "buro": {"score": 695, "icc": "0002", "creditos_activos": 0, "creditos_cerrados": 0,
                 "saldo_vencido": 0, "tiene_juicios": False, "creditos_vencidos": 0,
                 "alertas": [], "reporte_incompleto": False},
        "ab":   {"peor_mop_historico": 0, "nivel_endeudamiento": "indeterminado", "tendencia_reciente": "indeterminada",
                 "total_cuentas": 0, "saldo_vencido": 0, "saldo_actual_abiertas": 0},
        "fin":  {"total_ingresos": 20000, "total_egresos": 8000, "ratio_cuota_ingreso": 10.0,
                 "aprobado": True, "ingreso_neto": 12000, "monto_aprobado": 8000, "monto_reducido": False},
        "ml":   None,
        "kyc":  {"aprobado": False, "score_biometrico": 96.0, "alertas": []},
    },
    {
        "nombre": "🔔 Alejandra Sanchez (thin file)",
        "buro": {"score": 702, "icc": "0009", "creditos_activos": 1, "creditos_cerrados": 2,
                 "saldo_vencido": 0, "tiene_juicios": False, "creditos_vencidos": 0,
                 "alertas": [], "reporte_incompleto": False},
        "ab":   {"peor_mop_historico": 0, "nivel_endeudamiento": "medio", "tendencia_reciente": "positiva",
                 "total_cuentas": 3, "saldo_vencido": 0, "saldo_actual_abiertas": 5200},
        "fin":  {"total_ingresos": 22000, "total_egresos": 9500, "ratio_cuota_ingreso": 15.0,
                 "aprobado": True, "ingreso_neto": 12500, "monto_aprobado": 8000, "monto_reducido": False},
        "ml":   {"valor_decil": 7},
        "kyc":  {"aprobado": True, "score_biometrico": 94.3, "alertas": []},
    },
    {
        "nombre": "✅ Patricia Herrera (umbral 40%)",
        "buro": {"score": 650, "icc": "0006", "creditos_activos": 3, "creditos_cerrados": 12,
                 "saldo_vencido": 0, "tiene_juicios": False, "creditos_vencidos": 0,
                 "alertas": [], "reporte_incompleto": False},
        "ab":   {"peor_mop_historico": 0, "nivel_endeudamiento": "alto", "tendencia_reciente": "estable",
                 "total_cuentas": 15, "saldo_vencido": 0, "saldo_actual_abiertas": 48000},
        "fin":  {"total_ingresos": 38000, "total_egresos": 16000, "ratio_cuota_ingreso": 36.0,
                 "aprobado": True, "ingreso_neto": 22000, "monto_aprobado": 20000, "monto_reducido": False},
        "ml":   {"valor_decil": 8},
        "kyc":  {"aprobado": True, "score_biometrico": 96.0, "alertas": []},
    },
]

# ── Evaluación ─────────────────────────────────────────────────────────────────
def parse_icc(icc_raw):
    if icc_raw is None: return None
    try: return int(str(icc_raw).strip())
    except: return None

print(f"\n{'─'*95}")
print(f"{'CASO':<35} {'ICC':>4} {'Cuentas':>8} {'MOP':>5} {'Sv>0':>6} {'Tendencia':<18} {'Ajuste':>7}  Razón")
print(f"{'─'*95}")

for c in CASOS:
    buro = c["buro"]
    ab   = c["ab"]
    fin  = c["fin"]
    ml   = c["ml"] or {}
    kyc  = c["kyc"]

    icc_val = parse_icc(buro.get("icc"))
    ajuste, razon = _calcular_ajuste_cualitativo(buro, ab, icc_val, fin)
    score_cuant, desglose = _calcular_score_cuantitativo(kyc, fin, buro, ab, ml)
    score_final = max(1, min(10, score_cuant + ajuste))

    sv = buro.get("saldo_vencido", 0) or ab.get("saldo_vencido", 0)
    tend = ab.get("tendencia_reciente", "")[:15]

    ajuste_str = f"+1 ▲" if ajuste == 1 else ("-1 ▼" if ajuste == -1 else " 0 —")
    score_str  = f"[{score_cuant}→{score_final}]"
    print(f"{c['nombre']:<35} {icc_val if icc_val is not None else 'N/A':>4} "
          f"{ab['total_cuentas']:>8} {ab['peor_mop_historico']:>5} {sv:>6,.0f} "
          f"{tend:<18} {ajuste_str:>7}  {score_str}  {razon[:50]}")

print(f"{'─'*95}\n")
print("DIAGNÓSTICO:")
print("  +1 requiere: ICC≥7 AND cuentas≥10 AND MOP≤1 AND endeudamiento no alto")
print("  -1 requiere: tendencia_neg OR ICC≤2 OR saldo_vencido>0")
print()
print("OBSERVACIONES:")
print("  - Cristel (ICC=5, 35 ctas): nunca sube porque ICC < 7")
print("  - Rafael (ICC=7, 11 ctas): nunca sube porque tendencia 'creciente' → dispara -1")
print("  - Alejandra (ICC=9, 3 ctas): nunca sube porque < 10 cuentas")
print("  - Patricia (ICC=6, 15 ctas): nunca sube porque ICC < 7")
print()
print("  → El +1 casi nunca aplica en SOFIPO microfinanciera (clientes típicos ICC 4-6, cuentas 3-10)")
print("  → Posible recalibración: bajar umbral a ICC≥6 y cuentas≥5 para +1")
