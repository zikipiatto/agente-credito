from pydantic import BaseModel
from typing import Optional

class Condiciones(BaseModel):
    numero_solicitud: str
    fecha: str
    monto: float
    tipo: str                    # PRODUCTIVO / CONSUMO
    producto: str                # Capital de Trabajo, etc.
    tasa: float                  # % anual
    cuota: float
    frecuencia: str              # MENSUAL DE 28 DÍAS
    plazo: int                   # número de pagos

class Identificacion(BaseModel):
    verificacion_id: float       # score 0-100
    reconocimiento_facial: float
    deteccion_vida: float
    validacion_gobierno: float
    video_selfie: float
    distancia_km: float

class DatosCliente(BaseModel):
    nombre: str
    rfc: str
    curp: str
    fecha_nacimiento: str
    genero: str
    estado_civil: str
    tipo_vivienda: str
    dependientes_economicos: int
    antiguedad_domicilio_anios: int

class Negocio(BaseModel):
    nombre: str
    giro: str
    antiguedad_fecha: str        # fecha desde que opera el negocio

class IngresosEgresos(BaseModel):
    total_ingresos: float
    total_egresos: float

class CuentaBuro(BaseModel):
    numero: int
    tipo_credito: str                  # PRÉSTAMO PERSONAL, TARJETA, etc.
    otorgante: str                     # BANCO, MICROFINANCIERA, etc.
    estado: str                        # "ABIERTA" | "CERRADA"
    fecha_apertura: Optional[str] = None
    fecha_cierre: Optional[str] = None
    credito_maximo: float = 0.0
    saldo_actual: float = 0.0
    saldo_vencido: float = 0.0
    mop_actual: int = 0               # 0=sin info, 1=puntual, 2-9=días atraso
    peor_mop: int = 0                 # peor MOP histórico registrado
    pagos_puntuales: int = 0          # conteo de MOP=1 en historial
    pagos_atrasados: int = 0          # conteo de MOP>1 en historial

class Buro(BaseModel):
    score: Optional[float] = None
    icc: Optional[str] = None         # ICC del reporte (ej: "0005")
    tipo_score: Optional[str] = None  # "BC SCORE", "FICO", etc.
    causas_score: list[str] = []      # razones del valor del score
    etiqueta_riesgo: Optional[str] = None
    alertas_hawk: list[str] = []
    tiene_juicios: bool = False
    creditos_activos: int = 0
    creditos_cerrados: int = 0
    creditos_vencidos: int = 0
    peor_atraso_dias: int = 0
    saldo_actual: float = 0.0
    saldo_vencido: float = 0.0
    cuentas: list[CuentaBuro] = []    # detalle de todas las cuentas

class ResultadoML(BaseModel):
    fico_score: Optional[float] = None      # -10 = sin hit
    score_no_hit: Optional[float] = None    # -10 = sin hit
    va_no_hi: Optional[float] = None        # -10 = sin hit
    decision_ml: Optional[str] = None       # Aceptada / Validar con mesa de crédito
    capacidad_pago_ml: Optional[float] = None
    valor_decil: Optional[int] = None       # 1-10, donde 10 = mejor cliente
    nivel_riesgo: Optional[str] = None      # ALTO / MEDIO / BAJO (derivado del decil)

class SolicitudCompleta(BaseModel):
    condiciones: Condiciones
    identificacion: Identificacion
    cliente: DatosCliente
    negocio: Negocio
    finanzas: IngresosEgresos
    buro: Buro
    modelo_ml: Optional[ResultadoML] = None  # resultado del modelo predictivo (opcional)
