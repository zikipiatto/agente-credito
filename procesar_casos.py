"""
Procesa los 14 casos del Excel + PDFs de buró contra el API local.
Genera un resumen de decisiones en tabla.
"""
import re, json, math, requests, pandas as pd, pdfplumber
from pathlib import Path

API_URL  = "http://localhost:8000/analizar"
PDF_DIR  = Path("Ejemplos Solicitud y Buro/Reporte de Buro")
EXCEL    = PDF_DIR / "Reporte buro.xlsx"

# ── Helpers de parseo ────────────────────────────────────────────

def extraer_texto(pdf_path):
    texto = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            texto += t + "\n"
    return texto

def parsear_score_icc(texto):
    m = re.search(r'BC SCORE\s*/\s*ICC\s+([\d]+)/([\d]+)', texto)
    if m:
        return int(m.group(1)), m.group(2)   # score int, ICC string "0006"
    return None, None

def parsear_hawk(texto):
    alertas = []
    bloque = re.search(r'MENSAJES DE ALERTA HAWK(.*?)SCORE', texto, re.DOTALL)
    bloque_text = bloque.group(1) if bloque else ''
    if bloque:
        for linea in bloque_text.splitlines():
            linea = linea.strip()
            m = re.search(r'\d{2}-\w+-\d{4}\s+\d{3}\s+[\w\s]+\s{2,}(.+)', linea)
            if m:
                msg = m.group(1).strip()
                if msg and msg not in alertas:
                    alertas.append(msg)
    # Juicios: solo en la sección HAWK o en tipo de cuenta — evitar falsos positivos del resto del texto
    tiene_juicios = any(kw in bloque_text.upper() for kw in ['JUICIO', 'AMPARO', 'EMBARGO', 'DEMANDA'])
    if not tiene_juicios:
        # También buscar en detalle de cuentas (entre DETALLE y RESUMEN)
        det = re.search(r'DETALLE DE LOS CREDITOS(.*?)RESUMEN DE CREDITOS', texto, re.DOTALL)
        if det:
            tiene_juicios = any(kw in det.group(1).upper() for kw in ['JUICIO', 'AMPARO'])
    return alertas, tiene_juicios

def parsear_resumen(texto):
    """Extrae la línea Tot: del resumen de créditos."""
    m = re.search(
        r'Tot:\s*([\d,]+)\s+([\d,.]+)\s+([\d,.]+)\s+([\d,.]+)\s+([\d,.]+)\s+([\d,.]+)\s+'
        r'([\d,]+)\s+([\d,.]+)\s+([\d,.]+)\s+([\d,.]+)\s+([\d,.]+)',
        texto
    )
    if m:
        def f(s): return float(s.replace(',', ''))
        return {
            'cuentas_abiertas':  int(m.group(1)),
            'limite_abiertas':   f(m.group(2)),
            'maximo_abiertas':   f(m.group(3)),
            'saldo_abiertas':    f(m.group(4)),
            'vencido_abiertas':  f(m.group(5)),
            'pago_a_realizar':   f(m.group(6)),
            'cuentas_cerradas':  int(m.group(7)),
        }
    return {}

def parsear_cuentas(texto):
    """
    Extrae cuentas individuales del reporte.
    Retorna lista de dicts con los campos de CuentaBuro.
    """
    cuentas = []
    # Detectar bloques de cuenta por patrón: número de cuenta, tipo, otorgante
    # Buscamos líneas con MOP tipo "01=CUENTA CON PAGO PUNTUAL Y ADECUADO" o "97=..."
    bloques = re.split(r'\n(?=\s*\d+\s+(?:PAGOS FIJOS|PRÉSTAMO PERSONAL|TARJETA DE CRÉDITO|CRÉDITO HIPOTECARIO|CUENTA DE CHEQUES))', texto)

    tipo_mop = {
        '01': 1, '02': 2, '03': 3, '04': 4, '05': 5, '06': 6, '07': 7,
        '97': 7, '96': 7, '9A': 7,
    }

    for bloque in bloques[1:]:
        lines = bloque.strip().splitlines()
        if not lines:
            continue

        primera = lines[0].strip()

        # Tipo de crédito y otorgante
        tipos_credito = ['PAGOS FIJOS', 'PRÉSTAMO PERSONAL', 'TARJETA DE CRÉDITO',
                         'CRÉDITO HIPOTECARIO', 'CUENTA DE CHEQUES', 'ARRENDAMIENTO']
        tipo = next((t for t in tipos_credito if t in primera.upper()), 'OTRO')

        otorgantes = ['BANCO', 'MICROFINANCIERA', 'FINANCIERA', 'SOFIPO', 'ISBAN',
                      'INFONAVIT', 'FONACOT', 'TIENDA DEPARTAMENTAL']
        otorgante = next((o for o in otorgantes if o in primera.upper()), 'OTRO')

        # Importes: buscar números grandes (crédito máximo, saldo)
        numeros = re.findall(r'([\d,]+\.\d{2})', bloque)
        numeros_f = [float(n.replace(',', '')) for n in numeros if float(n.replace(',', '')) > 0]

        credito_maximo = numeros_f[0] if len(numeros_f) > 0 else 0
        saldo_actual   = numeros_f[1] if len(numeros_f) > 1 else 0
        saldo_vencido  = numeros_f[2] if len(numeros_f) > 2 else 0

        # Pago a realizar de la cuenta
        monto_pagar = numeros_f[4] if len(numeros_f) > 4 else 0

        # Estado: ABIERTA si no hay fecha de cierre, CERRADA si la tiene
        tiene_cierre = bool(re.search(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|'
                                      r'Ene|Abr|Ago|[A-Z][a-z]{2})-20\d{2}\b.*\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|'
                                      r'Ene|Abr|Ago|[A-Z][a-z]{2})-20\d{2}\b', bloque))
        estado = 'CERRADA' if tiene_cierre else 'ABIERTA'

        # MOP actual y peor MOP del historial
        mop_actual = 1
        m_forma = re.search(r'(0[1-7]|97|96|9A)=CUENTA', bloque)
        if m_forma:
            mop_actual = tipo_mop.get(m_forma.group(1), 1)

        # Historial de pagos: extraer solo códigos formales MOP del historial
        # Buscamos la línea de historial de pagos (secuencia densa tipo "1 1 1 2 1 1")
        # Solo tomamos MOP de la sección de historial, no de texto general
        mop_desde_codigos = [tipo_mop.get(m, int(m[1]) if m[0]=='0' else 7)
                             for m in re.findall(r'(0[1-7]|97|96|9A)=CUENTA', bloque)]

        # Buscar línea de historial: secuencia de ≥4 valores separados por espacios (solo dígitos 1-7 y X)
        hist_linea = re.search(r'(?:^|\n)((?:[1-7X]\s+){4,}[1-7X])', bloque)
        if hist_linea:
            hist_vals = hist_linea.group(1).split()
            pagos_puntuales = hist_vals.count('1')
            pagos_atrasados = sum(1 for v in hist_vals if v in ('2','3','4','5'))
            mop_max_hist = max((int(v) for v in hist_vals if v.isdigit()), default=mop_actual)
        else:
            pagos_puntuales = 0
            pagos_atrasados = 0
            mop_max_hist = mop_actual

        # Peor MOP: el mayor entre el historial y los códigos MOP formales encontrados
        peor_mop = max(mop_max_hist, max(mop_desde_codigos, default=mop_actual))
        if re.search(r'\b(97|96|9A)=CUENTA', bloque):
            peor_mop = max(peor_mop, 7)

        # Fecha de apertura
        fechas = re.findall(r'(\d{2}-\w+-\d{4})', bloque)
        fecha_apertura = fechas[0] if fechas else None

        cuentas.append({
            'numero':          len(cuentas) + 1,
            'tipo_credito':    tipo,
            'otorgante':       otorgante,
            'estado':          estado,
            'fecha_apertura':  fecha_apertura,
            'fecha_cierre':    None,
            'credito_maximo':  credito_maximo,
            'saldo_actual':    saldo_actual,
            'saldo_vencido':   saldo_vencido,
            'mop_actual':      mop_actual,
            'peor_mop':        peor_mop,
            'pagos_puntuales': pagos_puntuales,
            'pagos_atrasados': pagos_atrasados,
        })

    return cuentas

def calc_cuota(monto, tasa, plazo, frecuencia):
    dias = {'QUINCENAL': 14, 'MENSUAL DE 28 DÍAS': 28, 'MENSUAL': 30, 'SEMANAL': 7}.get(frecuencia, 28)
    r = (tasa / 100) * (dias / 365)
    if r == 0 or plazo == 0:
        return round(monto / plazo, 2) if plazo else 0
    return round(monto * r / (1 - math.pow(1 + r, -plazo)), 2)

def frec_str(frec_num):
    mapping = {15: 'QUINCENAL', 28: 'MENSUAL DE 28 DÍAS', 30: 'MENSUAL', 7: 'SEMANAL'}
    return mapping.get(int(frec_num), 'MENSUAL DE 28 DÍAS')

# ── Procesamiento principal ──────────────────────────────────────

df = pd.read_excel(EXCEL)
resultados = []

print(f"\n{'─'*80}")
print(f"Procesando {len(df)} casos...")
print(f"{'─'*80}\n")

for _, row in df.iterrows():
    num = int(row['numero_solicitud'])
    pdf_path = PDF_DIR / f"REPORTE DE BURO DE CREDITO_{num}.pdf"

    if not pdf_path.exists():
        print(f"[{num}] ⚠ PDF no encontrado, saltando")
        continue

    print(f"[{num}] Procesando...", end=" ", flush=True)

    # ── Parsear PDF ──
    texto = extraer_texto(pdf_path)
    score_val, icc_val   = parsear_score_icc(texto)
    hawks, tiene_juicios = parsear_hawk(texto)
    resumen              = parsear_resumen(texto)
    cuentas              = parsear_cuentas(texto)

    # ── Detectar créditos vencidos (cuentas con MOP >= 5 o saldo vencido) ──
    creditos_vencidos = sum(1 for c in cuentas if c['peor_mop'] >= 5 or c['saldo_vencido'] > 0)
    peor_mop_global   = max((c['peor_mop'] for c in cuentas), default=0)

    frec = frec_str(row['frecuencia'])
    monto = float(row['monto_solicitado'])
    tasa  = float(row['tasa'])
    plazo = int(row['plazo'])
    cuota = calc_cuota(monto, tasa, plazo, frec)

    # ── Construir payload ──
    payload = {
        "condiciones": {
            "numero_solicitud": str(num),
            "fecha":            "2026-03-27",
            "monto":            monto,
            "tipo":             "PRODUCTIVO",
            "producto":         "Capital de Trabajo",
            "tasa":             tasa,
            "cuota":            cuota,
            "frecuencia":       frec,
            "plazo":            plazo,
        },
        "identificacion": {
            "verificacion_id":      100,
            "reconocimiento_facial": 90,
            "deteccion_vida":        100,
            "validacion_gobierno":   100,
            "video_selfie":          0,
            "distancia_km":          0.5,
        },
        "cliente": {
            "nombre":                    texto.split('\n')[6].strip()[:50] if len(texto.split('\n')) > 6 else "CLIENTE",
            "rfc":                       re.search(r'\b[A-Z]{4}\d{6}[A-Z0-9]{3}\b', texto).group(0) if re.search(r'\b[A-Z]{4}\d{6}[A-Z0-9]{3}\b', texto) else "XXXX000000X00",
            "curp":                      "",
            "fecha_nacimiento":          "01/01/1980",
            "genero":                    str(row.get('genero', 'FEMENINO')),
            "estado_civil":              "SOLTERO(A)",
            "tipo_vivienda":             "FAMILIAR",
            "dependientes_economicos":   int(row.get('dependientes_economicos', 0)),
            "antiguedad_domicilio_anios": int(row.get('antiguedad_domicilio', 5)),
        },
        "negocio": {
            "nombre":          "NEGOCIO",
            "giro":            str(row.get('giro_negocio', 'SERVICIOS'))[:50],
            "antiguedad_fecha": "2020-01-01",
        },
        "finanzas": {
            "total_ingresos": float(row['total_ingresos']),
            "total_egresos":  float(row['total_egresos']),
        },
        "buro": {
            "score":              score_val,
            "icc":                icc_val,
            "tipo_score":         "BC SCORE" if score_val else None,
            "causas_score":       [],
            "etiqueta_riesgo":    None,
            "alertas_hawk":       hawks,
            "tiene_juicios":      tiene_juicios,
            "creditos_activos":   resumen.get('cuentas_abiertas', 0),
            "creditos_cerrados":  resumen.get('cuentas_cerradas', 0),
            "creditos_vencidos":  creditos_vencidos,
            "peor_atraso_dias":   0,
            "saldo_actual":       resumen.get('saldo_abiertas', 0),
            "saldo_vencido":      resumen.get('vencido_abiertas', 0),
            "pago_a_realizar":    resumen.get('pago_a_realizar', 0),
            "cuentas":            cuentas,
        },
        "modelo_ml": {
            "fico_score":        -10,
            "score_no_hit":      -10,
            "va_no_hi":          -10,
            "decision_ml":       "Aceptada" if int(row.get('decil_prod_bc', 5)) >= 6 else "Validar con mesa de crédito",
            "capacidad_pago_ml": None,
            "valor_decil":       int(row.get('decil_prod_bc', 5)),
            "nivel_riesgo":      "BAJO" if int(row.get('decil_prod_bc', 5)) >= 7 else ("MEDIO" if int(row.get('decil_prod_bc', 5)) >= 4 else "ALTO"),
        }
    }

    # ── Llamar al API ──
    try:
        resp = requests.post(API_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        dec  = data['decision']
        fin  = data['financiero']
        cf   = data['condiciones_finales']

        resultado = {
            'solicitud':       num,
            'score_bc':        score_val,
            'icc':             icc_val,
            'decil_ml':        int(row.get('decil_prod_bc', 5)),
            'ingresos':        float(row['total_ingresos']),
            'egresos':         float(row['total_egresos']),
            'ingreso_neto':    fin['ingreso_neto'],
            'pago_buro':       fin['pago_buro'],
            'monto_solicitado': monto,
            'monto_aprobado':  cf['monto'],
            'cuota_final':     cf['cuota'],
            'monto_reducido':  cf['monto_reducido'],
            'decision':        dec['resultado'],
            'score_riesgo':    dec['score_riesgo'],
            'razon':           dec['razon_principal'][:80] if dec.get('razon_principal') else '',
        }
        resultados.append(resultado)
        simbolo = {'APROBADO':'✅','RECHAZADO':'❌','ESCALAR_EJECUTIVO':'⚠️','VALIDACION':'🔔'}.get(dec['resultado'],'?')
        print(f"{simbolo} {dec['resultado']}  score={dec['score_riesgo']}/10  monto_aprobado=${cf['monto']:,.0f}")

    except Exception as e:
        print(f"❌ Error API: {e}")
        resultados.append({'solicitud': num, 'decision': 'ERROR', 'razon': str(e)[:80]})

# ── Resumen final ────────────────────────────────────────────────
print(f"\n{'═'*80}")
print("RESUMEN DE RESULTADOS")
print(f"{'═'*80}")

df_res = pd.DataFrame(resultados)
print(df_res[['solicitud','score_bc','decil_ml','monto_solicitado','monto_aprobado',
              'monto_reducido','decision','score_riesgo']].to_string(index=False))

print(f"\n{'─'*40}")
print("Distribución de decisiones:")
if 'decision' in df_res.columns:
    print(df_res['decision'].value_counts().to_string())

# Guardar a JSON para referencia
out_path = "resultados_casos.json"
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2, default=str)
print(f"\nResultados guardados en {out_path}")
