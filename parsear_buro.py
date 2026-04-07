"""
Extracción de datos desde Reportes de Buró de Crédito (Buró de Crédito México).
Soporta PDFs de Buró de Crédito con BC SCORE / ICC y SCORE NO HIT.

Uso:
    from parsear_buro import parsear_pdf_buro
    datos = parsear_pdf_buro("reporte.pdf")
    # datos["score"], datos["icc"], datos["cuentas_abiertas"], ...
"""
import re
import io
import pdfplumber


# ─────────────────────────────────────────────────────────────────────────────
# Texto base
# ─────────────────────────────────────────────────────────────────────────────

def extraer_texto(fuente) -> str:
    """
    Extrae texto de un PDF.
    fuente puede ser: ruta str/Path, bytes, o file-like object.
    """
    if isinstance(fuente, (str,)) or hasattr(fuente, '__fspath__'):
        ctx = pdfplumber.open(fuente)
    else:
        if isinstance(fuente, bytes):
            fuente = io.BytesIO(fuente)
        ctx = pdfplumber.open(fuente)

    texto = ""
    with ctx as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            texto += t + "\n"
    return texto


# ─────────────────────────────────────────────────────────────────────────────
# Score e ICC
# ─────────────────────────────────────────────────────────────────────────────

def _parsear_score_icc(texto: str) -> dict:
    """
    Extrae score, ICC, tipo de score y causas del texto del PDF.

    Formatos soportados (sección SCORE del reporte Buró de Crédito):
      BC SCORE / ICC   0607/0005       → tipo=BC SCORE / ICC, score=607, icc=5
      BC SCORE / ICC   -009/0007       → tipo=BC SCORE / ICC, score=-9, icc=7 (código exclusión)
      BC SCORE         662             → tipo=BC SCORE, score=662, icc=None
      SCORE NO HIT     450             → tipo=SCORE NO HIT, score=450, icc=None

    Las causas del score se extraen del bloque entre el encabezado y DETALLE DE LOS CREDITOS.
    """
    resultado = {
        "score":        None,
        "icc":          None,
        "tipo_score":   "SIN SCORE",
        "causas_score": [],
    }

    # ── Extraer causas del bloque SCORE ──
    bloque_score = re.search(
        r'NOMBRE DEL SCORE[^\n]*\n(.*?)DETALLE DE LOS CREDITOS',
        texto, re.DOTALL | re.IGNORECASE
    )
    causas = []
    if bloque_score:
        for linea in bloque_score.group(1).splitlines():
            linea = linea.strip()
            # Excluir: líneas con "BC SCORE", etiquetas de riesgo, muy cortas
            if (len(linea) > 20
                    and 'BC SCORE' not in linea.upper()
                    and 'ACEPTADO PARA' not in linea.upper()
                    and not re.match(r'^[-\d\s]+$', linea)):
                causas.append(linea)
    resultado["causas_score"] = causas[:4]

    # ── Formato combinado: BC SCORE / ICC  0607/0005 ──
    # El score puede tener ceros a la izquierda (0697 = 697) o ser negativo (-009 = -9)
    m = re.search(r'BC\s+SCORE\s*/\s*ICC\s+(-?0*[\d]+)\s*/\s*(0*[\d]+)', texto, re.IGNORECASE)
    if m:
        resultado["score"]      = int(m.group(1))   # 0697 → 697, -009 → -9
        resultado["icc"]        = int(m.group(2))   # 0006 → 6
        resultado["tipo_score"] = "BC SCORE / ICC"
        return resultado

    # ── BC SCORE sin ICC (valor positivo o código) ──
    m = re.search(r'BC\s+SCORE\s+(-?0*[\d]+)', texto, re.IGNORECASE)
    if m:
        resultado["score"]      = int(m.group(1))
        resultado["tipo_score"] = "BC SCORE"
        # ICC en línea separada
        m_icc = re.search(r'\bICC\b\s+(0*[\d]+)', texto, re.IGNORECASE)
        if m_icc:
            resultado["icc"] = int(m_icc.group(1))
        return resultado

    # ── SCORE NO HIT ──
    m = re.search(r'SCORE\s+NO\s+HIT\s+(-?0*[\d]+)', texto, re.IGNORECASE)
    if m:
        resultado["score"]      = int(m.group(1))
        resultado["tipo_score"] = "SCORE NO HIT"
        return resultado

    return resultado


# ─────────────────────────────────────────────────────────────────────────────
# HAWK y Juicios
# ─────────────────────────────────────────────────────────────────────────────

def _parsear_hawk(texto: str) -> dict:
    """
    Extrae alertas HAWK y detecta juicios en el reporte.

    Returns:
        {alertas_hawk: list, tiene_juicios: bool}
    """
    alertas = []

    # Bloque entre "MENSAJES DE ALERTA HAWK" y "SCORE"
    bloque_match = re.search(r'MENSAJES DE ALERTA HAWK(.*?)(?:SCORE|BC SCORE)', texto, re.DOTALL | re.IGNORECASE)
    bloque_text  = bloque_match.group(1) if bloque_match else ''

    if bloque_text:
        for linea in bloque_text.splitlines():
            linea = linea.strip()
            # Formato: NÚM  FECHA  CLAVE  OTORGANTE+MENSAJE+ORIGEN (un solo espacio entre campos)
            m = re.search(r'^\d+\s+\d{2}-\w{3}-\d{4}\s+\d{3}\s+(.+)', linea)
            if m:
                msg = m.group(1).strip()
                if msg and msg not in alertas:
                    alertas.append(msg)

    # Juicios: en bloque HAWK o en detalle de cuentas
    PALABRAS_JUICIO = ['JUICIO', 'AMPARO', 'EMBARGO', 'DEMANDA', 'LITIGIO']
    tiene_juicios = any(kw in bloque_text.upper() for kw in PALABRAS_JUICIO)
    if not tiene_juicios:
        det = re.search(r'DETALLE DE LOS CREDITOS(.*?)(?:RESUMEN DE CREDITOS|$)', texto, re.DOTALL | re.IGNORECASE)
        if det:
            tiene_juicios = any(kw in det.group(1).upper() for kw in PALABRAS_JUICIO)

    return {"alertas_hawk": alertas, "tiene_juicios": tiene_juicios}


# ─────────────────────────────────────────────────────────────────────────────
# Resumen de créditos (totales)
# ─────────────────────────────────────────────────────────────────────────────

def _parsear_resumen(texto: str) -> dict:
    """
    Extrae la línea Tot: del resumen de créditos.
    Columnas: cuentas_abiertas, limite, maximo, saldo, vencido, pago, cuentas_cerradas...
    """
    m = re.search(
        r'Tot:\s*([\d,]+)\s+([\d,.]+)\s+([\d,.]+)\s+([\d,.]+)\s+([\d,.]+)\s+([\d,.]+)\s+'
        r'([\d,]+)',
        texto
    )
    if m:
        def f(s): return float(s.replace(',', ''))
        return {
            'cuentas_abiertas': int(m.group(1)),
            'limite_abiertas':  f(m.group(2)),
            'maximo_abiertas':  f(m.group(3)),
            'saldo_abiertas':   f(m.group(4)),
            'vencido_abiertas': f(m.group(5)),
            'pago_a_realizar':  f(m.group(6)),
            'cuentas_cerradas': int(m.group(7)),
        }
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Detalle de cuentas
# ─────────────────────────────────────────────────────────────────────────────

_TIPO_MOP = {
    '01': 1, '02': 2, '03': 3, '04': 4, '05': 5, '06': 6, '07': 7,
    '97': 7, '96': 7, '9A': 7,
}

_TIPOS_CREDITO = [
    'PAGOS FIJOS', 'PRÉSTAMO PERSONAL', 'TARJETA DE CRÉDITO',
    'CRÉDITO HIPOTECARIO', 'CUENTA DE CHEQUES', 'ARRENDAMIENTO',
    'AUTOMOTRIZ', 'MICROCRÉDITO',
]

_OTORGANTES = [
    'BANCO', 'MICROFINANCIERA', 'FINANCIERA', 'SOFIPO', 'SOFOM',
    'ISBAN', 'INFONAVIT', 'FONACOT', 'TIENDA DEPARTAMENTAL', 'CAJA',
]


def _parsear_cuentas(texto: str) -> list:
    """
    Extrae cuentas individuales usando XX=CUENTA como ancla.

    El layout multi-columna del PDF parte las palabras en líneas distintas
    (e.g. PRÉSTAMO / PERSONAL en líneas separadas), así que no es posible
    dividir por tipo de crédito. El marcador XX=CUENTA siempre aparece en
    una sola línea y es el ancla fiable para delimitar cada cuenta.

    Columnas de importes en el PDF (orden consistente):
      [0] LÍMITE CRÉDITO MÁXIMO   [1] SALDO MÁXIMO ALCANZADO
      [2] SALDO ACTUAL             [3] SALDO VENCIDO
      [4] MONTO A PAGAR
    """
    cuentas = []
    patron_mop = re.compile(r'(0[1-7]|97|96|9A)=CUENTA')
    matches = list(patron_mop.finditer(texto))

    if not matches:
        return cuentas

    for idx, m in enumerate(matches):
        mop_codigo = m.group(1)
        mop_actual = _TIPO_MOP.get(mop_codigo, 1)

        # Contexto antes del marcador (tipo, otorgante) y bloque posterior (datos)
        inicio_ctx = max(0, m.start() - 300)
        fin_ctx    = matches[idx + 1].start() if idx + 1 < len(matches) else len(texto)
        bloque_ctx = texto[inicio_ctx:m.start()]   # para tipo/otorgante/fechas
        bloque_post = texto[m.end():fin_ctx]        # para importes e historial

        # ── Línea de importes: la primera línea con "MX" y cantidades ──
        # Formato: "N [PAGOS] [OTORGANTE] MX [fechas] importe importe ..."
        m_data = re.search(
            r'^\s*\d+\s+(?:[\w\s]{0,30}?)MX\s+.+?((?:[\d,]+\.\d{2}[\s-]*){1,6})',
            bloque_post, re.MULTILINE
        )
        numeros_f = []
        if m_data:
            numeros_f = [float(n.replace(',', ''))
                         for n in re.findall(r'[\d,]+\.\d{2}', m_data.group(0))]

        limite        = numeros_f[0] if len(numeros_f) > 0 else 0
        saldo_maximo  = numeros_f[1] if len(numeros_f) > 1 else 0
        saldo_actual  = numeros_f[2] if len(numeros_f) > 2 else 0
        saldo_vencido = numeros_f[3] if len(numeros_f) > 3 else 0
        monto_pagar   = numeros_f[4] if len(numeros_f) > 4 else 0
        # Si el límite es 0 (cuenta liquidada/castigada) usar saldo máximo como referencia
        credito_maximo = limite if limite > 0 else saldo_maximo

        # ── Estado ABIERTA / CERRADA ──
        # Cerrada: "TOTAL SIN REC" o saldo=0 con indicador SALDO=0
        esta_cerrada = bool(re.search(
            r'TOTAL SIN REC|CUENTA CERRADA|SALDO=0', bloque_post, re.IGNORECASE
        ))
        estado = 'CERRADA' if esta_cerrada else 'ABIERTA'

        # ── Historial de pagos ──
        # Línea de meses: "N O S A J J M A M F E D ..."  seguida de dígitos/X
        hist_match = re.search(
            r'(?:[A-Z]\s+){4,}[A-Z]\s*\n\s*([0-9X](?:\s+[0-9X])+)',
            bloque_post
        )
        pagos_puntuales = pagos_atrasados = 0
        mop_max_hist = mop_actual
        if hist_match:
            hist_vals = hist_match.group(1).split()
            pagos_puntuales = hist_vals.count('1')
            pagos_atrasados = sum(1 for v in hist_vals if v.isdigit() and int(v) >= 2)
            mop_max_hist    = max((int(v) for v in hist_vals if v.isdigit()), default=mop_actual)

        peor_mop = max(mop_actual, mop_max_hist)

        # ── Tipo, otorgante y fechas (buscar en contexto previo + posterior) ──
        bloque_completo = bloque_ctx + mop_codigo + '=CUENTA' + bloque_post
        # El layout multi-columna parte los nombres en líneas distintas (ej. PRÉSTAMO / PERSONAL),
        # así que colapsamos whitespace y buscamos con tolerancia entre palabras.
        bloque_upper = re.sub(r'\s+', ' ', bloque_completo).upper()
        tipo = 'OTRO'
        for t in _TIPOS_CREDITO:
            palabras = t.upper().split()
            patron_t = r'.{0,80}'.join(re.escape(p) for p in palabras)
            if re.search(patron_t, bloque_upper):
                tipo = t
                break
        otorg = next((o for o in _OTORGANTES if o in bloque_upper), 'OTRO')
        fechas = re.findall(r'\d{2}-\w{3}-\d{4}', bloque_completo)
        fecha_apertura = fechas[0] if fechas else None

        cuentas.append({
            'numero':          len(cuentas) + 1,
            'tipo_credito':    tipo,
            'otorgante':       otorg,
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
            'monto_pagar':     monto_pagar,
        })

    return cuentas


# ─────────────────────────────────────────────────────────────────────────────
# Datos personales básicos
# ─────────────────────────────────────────────────────────────────────────────

def _parsear_persona(texto: str) -> dict:
    """Extrae RFC y nombre del encabezado del reporte.

    La línea después de 'NOMBRE(S) APELLIDOS' tiene el formato:
      NOMBRE RFC FECHA_NAC IFE CURP ...       (RFC presente)
      NOMBRE FECHA_NAC IFE CURP ...           (RFC ausente en algunos reportes)

    El boundary del nombre puede ser el RFC (4 letras+6 dígitos) o,
    si no hay RFC, la fecha de nacimiento (DD-Mon-YYYY).
    """
    rfc = None
    nombre = None

    m_header = re.search(
        r'NOMBRE\(?S?\)?\s+APELLIDOS[^\n]*\n([^\n]+)',
        texto, re.IGNORECASE
    )
    if m_header:
        linea = m_header.group(1).strip()

        # Intentar RFC como boundary (4 letras mayúsculas + 6 dígitos + 0-3 alfanumerico)
        m_rfc_pos = re.search(r'[A-Z]{4}\d{6}[A-Z0-9]{0,3}', linea)
        if m_rfc_pos:
            rfc    = m_rfc_pos.group(0)
            nombre = linea[:m_rfc_pos.start()].strip()
        else:
            # Sin RFC: usar fecha de nacimiento como boundary (DD-Mon-YYYY)
            m_fecha = re.search(r'\d{2}-[A-Za-z]{3}-\d{4}', linea)
            if m_fecha:
                nombre = linea[:m_fecha.start()].strip()
            else:
                nombre = linea.strip()

    # Fallback RFC: buscar en las primeras 3 páginas del texto completo
    if not rfc:
        m_rfc = re.search(r'\b([A-Z]{4}\d{6}[A-Z0-9]{0,3})\b', texto[:3000])
        if m_rfc:
            rfc = m_rfc.group(1)

    return {"rfc": rfc, "nombre": nombre}


# ─────────────────────────────────────────────────────────────────────────────
# Folio y fecha de consulta
# ─────────────────────────────────────────────────────────────────────────────

def _parsear_folio_fecha(texto: str) -> dict:
    """Extrae el folio y la fecha de consulta del encabezado del reporte."""
    folio = None
    fecha_consulta = None

    m_folio = re.search(r'FOLIO[:\s]+(\d+)', texto, re.IGNORECASE)
    if m_folio:
        folio = m_folio.group(1)

    m_fecha = re.search(r'FECHA DE CONSULTA[:\s]+(\d{2}/\w+/\d{4})', texto, re.IGNORECASE)
    if m_fecha:
        fecha_consulta = m_fecha.group(1)

    return {"folio": folio, "fecha_consulta": fecha_consulta}


# ─────────────────────────────────────────────────────────────────────────────
# Función principal
# ─────────────────────────────────────────────────────────────────────────────

def parsear_pdf_buro(fuente) -> dict:
    """
    Parsea un reporte de Buró de Crédito (Buró de Crédito México) y
    devuelve un dict listo para construir la sección `buro` de la solicitud.

    Args:
        fuente: ruta al PDF (str/Path), bytes o file-like object

    Returns:
        {
          score, icc, tipo_score, causas_score,
          alertas_hawk, tiene_juicios,
          cuentas_abiertas, cuentas_cerradas, creditos_vencidos,
          saldo_actual, saldo_vencido, pago_a_realizar,
          peor_atraso_dias,
          cuentas: [...],
          rfc, nombre,
          _texto_raw: str  (para debugging)
        }
    """
    texto = extraer_texto(fuente)

    score_data  = _parsear_score_icc(texto)
    hawk        = _parsear_hawk(texto)
    resumen     = _parsear_resumen(texto)
    cuentas     = _parsear_cuentas(texto)
    persona     = _parsear_persona(texto)
    encabezado  = _parsear_folio_fecha(texto)

    # Créditos vencidos: desde cuentas parseadas o inferido del saldo vencido
    creditos_vencidos_cuentas = sum(
        1 for c in cuentas if c['peor_mop'] >= 5 or c['saldo_vencido'] > 0
    )
    saldo_vencido = resumen.get("vencido_abiertas", 0)
    # Si el resumen muestra saldo vencido pero no extrajimos cuentas, asumir ≥1 vencido
    if creditos_vencidos_cuentas == 0 and saldo_vencido > 0:
        creditos_vencidos = 1
    else:
        creditos_vencidos = creditos_vencidos_cuentas

    # Peor MOP: desde cuentas parseadas o escaneando el texto completo
    if cuentas:
        peor_mop_global = max((c['peor_mop'] for c in cuentas), default=0)
    else:
        # Fallback: buscar el MOP más alto en todo el texto (ej. "05=CUENTA", "97=CUENTA")
        mops_en_texto = [
            _TIPO_MOP.get(m, 0)
            for m in re.findall(r'(0[1-7]|97|96|9A)=CUENTA', texto)
        ]
        peor_mop_global = max(mops_en_texto, default=0)

    return {
        # Score
        "score":        score_data["score"],
        "icc":          score_data["icc"],
        "tipo_score":   score_data["tipo_score"],
        "causas_score": score_data["causas_score"],

        # HAWK
        "alertas_hawk":  hawk["alertas_hawk"],
        "tiene_juicios": hawk["tiene_juicios"],

        # Resumen numérico (preferir valores del Tot: del resumen)
        "cuentas_abiertas":  resumen.get("cuentas_abiertas", len([c for c in cuentas if c["estado"] == "ABIERTA"])),
        "cuentas_cerradas":  resumen.get("cuentas_cerradas", len([c for c in cuentas if c["estado"] == "CERRADA"])),
        "creditos_vencidos": creditos_vencidos,
        "saldo_actual":      resumen.get("saldo_abiertas", 0),
        "saldo_vencido":     saldo_vencido,
        "pago_a_realizar":   resumen.get("pago_a_realizar", 0),
        "peor_atraso_dias":  peor_mop_global * 30,  # estimación: MOP × 30 días

        # Detalle de cuentas (cuando disponible)
        "cuentas": cuentas,

        # Persona
        "rfc":    persona["rfc"],
        "nombre": persona["nombre"],

        # Encabezado
        "folio":          encabezado["folio"],
        "fecha_consulta": encabezado["fecha_consulta"],

        # Debug
        "_texto_raw": texto,
    }
