"""
Prueba con dos escenarios:
- Solicitud 47491944: caso con alertas (Lourdes Priscila)
- Solicitud 47019193: happy path con buró limpio (Carlos Mendoza)
"""
import json
from datetime import datetime
from modelos import (
    SolicitudCompleta, Condiciones, Identificacion,
    DatosCliente, Negocio, IngresosEgresos, Buro, CuentaBuro
)
from agentes import analizar


def guardar_auditoria(solicitud, resultado):
    log = {
        "timestamp": datetime.now().isoformat(),
        "numero_solicitud": solicitud.condiciones.numero_solicitud,
        "cliente": solicitud.cliente.nombre,
        "kyc": resultado["resultado_kyc"],
        "financiero": resultado["resultado_financiero"],
        "buro": resultado["resultado_buro"],
        "decision": resultado["decision_final"]
    }
    with open("auditoria.jsonl", "a") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────
# CASO 1 — Con alertas (Lourdes Priscila)
# ─────────────────────────────────────────
solicitud_1 = SolicitudCompleta(
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
    negocio=Negocio(
        nombre="VENTA DE ROPA",
        giro="COMERCIO",
        antiguedad_fecha="20/03/2020"
    ),
    finanzas=IngresosEgresos(
        total_ingresos=30000,
        total_egresos=12000
    ),
    buro=Buro(
        score=None,
        etiqueta_riesgo=None,
        alertas_hawk=["JUICIO AMPARO 2023", "JUICIO AMPARO 2024"],
        tiene_juicios=True,
        creditos_activos=1,
        creditos_vencidos=0,
        peor_atraso_dias=0
    )
)

# ─────────────────────────────────────────
# CASO 2 — Happy path (Cristel Martinez, folio 47465130)
# Datos reales de buró: 35 créditos, todos MOP=1, score BC 662
# ─────────────────────────────────────────
solicitud_2 = SolicitudCompleta(
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
    negocio=Negocio(
        nombre="COLEGIO PAULO FREIRE",
        giro="SERVICIOS",
        antiguedad_fecha="30/04/2005"
    ),
    finanzas=IngresosEgresos(
        total_ingresos=45000,
        total_egresos=18000
    ),
    buro=Buro(
        score=662,
        icc="0005",
        tipo_score="BC SCORE",
        causas_score=[
            "Última cuenta nueva aperturada recientemente",
            "Consulta muy reciente",
            "Consulta muy reciente"
        ],
        etiqueta_riesgo=None,
        alertas_hawk=[
            "TELEFONO NO CORRESPONDE A ZONA POSTAL",
            "TELEFONO CORRESPONDE A NUMERO FIJO"
        ],
        tiene_juicios=False,
        creditos_activos=2,
        creditos_cerrados=33,
        creditos_vencidos=0,
        peor_atraso_dias=0,
        saldo_actual=173978.00,
        saldo_vencido=0.00,
        cuentas=[
            # Cuentas abiertas (datos reales del reporte)
            CuentaBuro(numero=1, tipo_credito="PRÉSTAMO PERSONAL", otorgante="MICROFINANCIERA",
                       estado="ABIERTA", fecha_apertura="03-Sep-2025",
                       credito_maximo=35050.00, saldo_actual=10057.00, saldo_vencido=0.00,
                       mop_actual=1, peor_mop=1, pagos_puntuales=4, pagos_atrasados=0),
            CuentaBuro(numero=2, tipo_credito="PRÉSTAMO PERSONAL", otorgante="BANCO",
                       estado="ABIERTA", fecha_apertura="26-Mar-2025",
                       credito_maximo=104832.00, saldo_actual=93408.00, saldo_vencido=0.00,
                       mop_actual=1, peor_mop=1, pagos_puntuales=8, pagos_atrasados=0),
            # Muestra representativa de cuentas cerradas (todas con MOP=1)
            CuentaBuro(numero=3, tipo_credito="PRÉSTAMO PERSONAL", otorgante="BANCO",
                       estado="CERRADA", fecha_apertura="26-Ago-2025", fecha_cierre="16-Dic-2025",
                       credito_maximo=75840.00, saldo_actual=0.00, saldo_vencido=0.00,
                       mop_actual=0, peor_mop=1, pagos_puntuales=4, pagos_atrasados=0),
            CuentaBuro(numero=4, tipo_credito="PRÉSTAMO PERSONAL", otorgante="BANCO",
                       estado="CERRADA", fecha_apertura="05-Jul-2022", fecha_cierre="05-Jul-2023",
                       credito_maximo=26500.00, saldo_actual=0.00, saldo_vencido=0.00,
                       mop_actual=0, peor_mop=1, pagos_puntuales=12, pagos_atrasados=0),
            CuentaBuro(numero=10, tipo_credito="PRÉSTAMO PERSONAL", otorgante="BANCO",
                       estado="CERRADA", fecha_apertura="13-Feb-2024", fecha_cierre="04-Jun-2024",
                       credito_maximo=78252.00, saldo_actual=0.00, saldo_vencido=0.00,
                       mop_actual=0, peor_mop=2, pagos_puntuales=3, pagos_atrasados=1),
            CuentaBuro(numero=19, tipo_credito="PRÉSTAMO PERSONAL", otorgante="BANCO",
                       estado="CERRADA", fecha_apertura="08-Jul-2021", fecha_cierre="10-Ene-2023",
                       credito_maximo=80500.00, saldo_actual=0.00, saldo_vencido=0.00,
                       mop_actual=0, peor_mop=1, pagos_puntuales=18, pagos_atrasados=0),
            CuentaBuro(numero=31, tipo_credito="PRÉSTAMO PERSONAL", otorgante="BANCO",
                       estado="CERRADA", fecha_apertura="17-Oct-2019", fecha_cierre="20-Abr-2021",
                       credito_maximo=49279.00, saldo_actual=0.00, saldo_vencido=0.00,
                       mop_actual=0, peor_mop=1, pagos_puntuales=18, pagos_atrasados=0),
        ]
    )
)

# ─────────────────────────────────────────
# CASO 3 — BC SCORE sin cuentas (reporte incompleto)
# ─────────────────────────────────────────
solicitud_3 = SolicitudCompleta(
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
    negocio=Negocio(
        nombre="TALLER MECANICO",
        giro="SERVICIOS",
        antiguedad_fecha="01/06/2019"
    ),
    finanzas=IngresosEgresos(
        total_ingresos=22000,
        total_egresos=9000
    ),
    buro=Buro(
        score=680,
        tipo_score="BC SCORE",   # ← tiene score pero sin cuentas = reporte incompleto
        etiqueta_riesgo=None,
        alertas_hawk=[],
        tiene_juicios=False,
        creditos_activos=0,      # ← aquí está el error del reporte
        creditos_cerrados=0,
        creditos_vencidos=0,
        peor_atraso_dias=0,
        saldo_actual=0.0,
        saldo_vencido=0.0
    )
)

# ─────────────────────────────────────────
# EJECUTAR LOS TRES CASOS
# ─────────────────────────────────────────
for solicitud in [solicitud_1, solicitud_2, solicitud_3]:
    print(f"\n{'='*65}")
    print(f"  ANALIZANDO SOLICITUD {solicitud.condiciones.numero_solicitud}")
    print(f"  Cliente: {solicitud.cliente.nombre}")
    print(f"{'='*65}\n")

    resultado = analizar(solicitud)
    print(resultado["expediente"])
    guardar_auditoria(solicitud, resultado)
    print("✅ Registro guardado en auditoria.jsonl\n")
