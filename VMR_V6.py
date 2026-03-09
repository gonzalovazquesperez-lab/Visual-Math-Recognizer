import cv2
import mediapipe as mp
import pyttsx3
import threading
import queue
import time
import math
import numpy as np
from collections import deque

# ---------- Parámetros y Umbrales ----------
PROCESS_EVERY_N = 2
MIN_HAND_AREA_RATIO = 0.02
MODEL_COMPLEXITY = 0

# Parámetros de Confirmación
CONFIRM_FRAMES = 50
INCONSISTENCY_TOLERANCE = 5
HAND_LOSS_TOLERANCE = 10
BUTTON_CONFIRM_FRAMES = 30

# Mapa de operaciones (5 = %)
OP_MAP = {1: "+", 2: "-", 3: "*", 4: "/", 5: "^"}

# Puntos clave
FINGER_TIPS = [8, 12, 16, 20]
MCP_JOINTS = [5, 9, 13, 17]
THUMB_TIP = 4
KNUCKLE_BASE = 2

# ---------- CONTROL DE MODOS ----------
MODE_CALCULATOR = 'CALCULATOR'
MODE_POINTER = 'POINTER'
MODE_FRACTION = 'FRACTION'
mode = MODE_CALCULATOR

# =========================
# ===== BLOQUE TTS ========
# =========================

# Cola de mensajes TTS y control de cierre
tts_q = queue.Queue(maxsize=8)
_shutdown_flag = threading.Event()
tts_thread = None

# Control anti-spam y debounce
_TTS_MIN_SAME_MSG_INTERVAL = 1.2   # segundos mínimos entre repetir el mismo mensaje
_TTS_MIN_ANY_MSG_INTERVAL = 0.12   # segundos mínimos entre cualquier mensaje
_TTS_TRIM_TO = 4                   # si la cola está llena, mantener estos últimos

_last_tts_time = 0.0
_last_tts_msg = None
_last_tts_msg_time = 0.0

def _clean_tts_queue():
    """Vacía la cola parcialmente dejando como máximo _TTS_TRIM_TO elementos."""
    try:
        tmp = []
        while True:
            try:
                tmp.append(tts_q.get_nowait())
            except Exception:
                break
        keep = tmp[-_TTS_TRIM_TO:] if len(tmp) > _TTS_TRIM_TO else tmp
        for item in keep:
            try:
                tts_q.put_nowait(item)
            except Exception:
                break
    except Exception:
        pass

def hablar(texto):
    """Encola texto para TTS con protecciones contra spam/repeticiones."""
    global _last_tts_time, _last_tts_msg, _last_tts_msg_time
    try:
        txt = str(texto)
    except Exception:
        txt = repr(texto)

    now = time.time()

    # Evitar repetir exactamente el mismo texto con demasiada frecuencia
    if _last_tts_msg is not None and txt == _last_tts_msg:
        if now - _last_tts_msg_time < _TTS_MIN_SAME_MSG_INTERVAL:
            return

    # Proteccion global: no enviar mensajes muy seguidos
    if now - _last_tts_time < _TTS_MIN_ANY_MSG_INTERVAL:
        try:
            if not tts_q.full():
                tts_q.put_nowait(txt)
                _last_tts_time = now
                _last_tts_msg = txt
                _last_tts_msg_time = now
            else:
                _clean_tts_queue()
                if not tts_q.full():
                    tts_q.put_nowait(txt)
                    _last_tts_time = now
                    _last_tts_msg = txt
                    _last_tts_msg_time = now
        except Exception:
            try:
                print("[TTS fallback] " + txt)
            except:
                pass
        return

    # Si la cola está llena, limpiar parcialmente
    if tts_q.full():
        _clean_tts_queue()

    # Intentar encolar
    try:
        tts_q.put_nowait(txt)
        _last_tts_time = now
        _last_tts_msg = txt
        _last_tts_msg_time = now
    except queue.Full:
        # Si sigue llena, descartamos el nuevo para evitar bloqueo
        pass

def _choose_spanish_voice(engine):
    try:
        voices = engine.getProperty('voices') or []
    except Exception:
        return None
    for v in voices:
        try:
            name = (getattr(v, 'name', '') or "").lower()
            vid = (getattr(v, 'id', '') or "").lower()
            langs = str(getattr(v, 'languages', '')).lower()
            if "spanish" in name or "es-" in vid or "es_" in vid or "es-" in langs or "es_" in langs or "sabina" in name or "mex" in name:
                return getattr(v, 'id', None) or getattr(v, 'name', None)
        except Exception:
            continue
    return None

def _init_engine_and_speak_once_check():
    """Inicializa el motor pyttsx3 y retorna engine o None (para debug)."""
    try:
        engine = pyttsx3.init()
        return engine
    except Exception as e:
        print("[TTS] ERROR iniciando pyttsx3:", e)
        return None

def tts_worker():
    """Worker que inicializa el engine y consume la cola hasta que se le pida cerrar."""
    engine = _init_engine_and_speak_once_check()
    if engine is None:
        # Fallback: imprimir mensajes en consola si no hay motor
        print("[TTS] Motor no disponible, usando fallback de consola.")
        while not _shutdown_flag.is_set():
            try:
                msg = tts_q.get(timeout=0.2)
            except Exception:
                continue
            if msg is None:
                try:
                    tts_q.task_done()
                except Exception:
                    pass
                continue
            try:
                print("[TTS fallback] " + str(msg))
            except Exception:
                pass
            finally:
                try:
                    tts_q.task_done()
                except Exception:
                    pass
        return

    # Mostrar y seleccionar voz en español si existe
    try:
        voices = engine.getProperty('voices') or []
        # debug print voces disponibles
        print("[TTS] Voces detectadas:", len(voices))
        for i, v in enumerate(voices[:20]):
            try:
                print(f"  voz[{i}] name='{getattr(v,'name',None)}' id='{getattr(v,'id',None)}' langs='{getattr(v,'languages',None)}'")
            except Exception:
                pass
        sel = _choose_spanish_voice(engine)
        if sel:
            try:
                engine.setProperty('voice', sel)
                print("[TTS] Voz española seleccionada:", sel)
            except Exception as e:
                print("[TTS] No se pudo setear voz española:", e)
        else:
            print("[TTS] No se detectó voz española; se usará la por defecto.")
        try:
            engine.setProperty('rate', 150)
        except Exception:
            pass
    except Exception as e:
        print("[TTS] Error configurando voces:", e)

    # Bucle principal de consumo
    while not _shutdown_flag.is_set():
        try:
            msg = tts_q.get(timeout=0.25)
        except Exception:
            continue
        if msg is None:
            try:
                tts_q.task_done()
            except Exception:
                pass
            continue
        try:
            engine.say(str(msg))
            engine.runAndWait()
        except Exception as e:
            print("[TTS] Error durante speak:", e)
            try:
                print("[TTS fallback] " + str(msg))
            except Exception:
                pass
        finally:
            try:
                tts_q.task_done()
            except Exception:
                pass

    # intento de liberar engine
    try:
        engine.stop()
    except Exception:
        pass

# Arrancar hilo TTS (no daemon para poder controlarlo al cerrar)
tts_thread = threading.Thread(target=tts_worker)
tts_thread.start()
print("[TTS] Hilo TTS iniciado.")

# =========================
# ===== FIN BLOQUE TTS ====
# =========================

# ---------- MediaPipe Setup ----------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Colores personalizados para el esqueleto
color_hand_conn = (0, 255, 255)
color_hand_landmark = (255, 0, 255)
draw_spec_conn = mp_draw.DrawingSpec(color=color_hand_conn, thickness=2)
draw_spec_land = mp_draw.DrawingSpec(color=color_hand_landmark, thickness=2, circle_radius=4)

hands_detector = mp_hands.Hands(
    max_num_hands=2,
    model_complexity=MODEL_COMPLEXITY,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------- Utilidades ----------
def hand_bbox_area(landmarks, img_w, img_h):
    xs = [lm.x * img_w for lm in landmarks]
    ys = [lm.y * img_h for lm in landmarks]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))

def is_thumb_up_vector(landmarks, hand_label='Right'):
    try:
        if hand_label == 'Right':
            return landmarks[THUMB_TIP].x < landmarks[THUMB_TIP - 1].x
        else:
            return landmarks[THUMB_TIP].x > landmarks[THUMB_TIP - 1].x
    except Exception:
        return False

def count_fingers(landmarks, hand_label='Right', smoothing_deque=None):
    count = 0
    for tip, mcp in zip(FINGER_TIPS, MCP_JOINTS):
        try:
            if landmarks[tip].y < landmarks[mcp].y:
                count += 1
        except:
            pass
    if is_thumb_up_vector(landmarks, hand_label):
        count += 1

    if smoothing_deque is not None:
        smoothing_deque.append(count)
        avg = int(round(sum(smoothing_deque) / len(smoothing_deque)))
        return avg

    return count

def draw_hud_panel(img, x, y, w, h, color=(0,0,0), alpha=0.6):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x, y), (x + w, y + h), (min(color[0]+50,255), min(color[1]+50,255), min(color[2]+50,255)), 1)

def calcular_mcd(a, b):
    while b:
        a, b = b, a % b
    return a

def simplificar_fraccion(num, den):
    if den == 0:
        return "Error", 0
    if num == 0:
        return 0, 1
    if den < 0:
        num = -num
        den = -den
    
    comun_divisor = calcular_mcd(abs(num), den)
    return num // comun_divisor, den // comun_divisor


# ---------- Estado Global ----------
valor1 = None
valor2 = None
operacion = None
resultado = None
fase = "pedir_v1" # La fase inicial es V1 para Enteros

# --- NUEVAS VARIABLES PARA FRACCIONES ---
num_v1 = None
den_v1 = None
num_v2 = None
den_v2 = None
num_res = None
den_res = None
# ----------------------------------------

temp_val = None
confirm_counter = 0
inconsistent_counter = 0
hand_lost_counter = 0
button_confirm_counter = 0
active_button_key = None
is_negative_current_input = False
is_fraction_mode = False

last_hover_button = None

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_idx = 0
hablar("Sistema iniciado. Modo calculadora activo.")

last_results = None
right_finger_smooth = deque(maxlen=8)
left_finger_smooth = deque(maxlen=8)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        img = frame.copy()
        h, w, _ = img.shape
        img_area = w * h

        # Inicialización limpia por frame
        val_izq = None
        val_der = None
        right_index_tip = None
        left_index_tip = None
        total_fingers = None

        if frame_idx % PROCESS_EVERY_N == 0:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            last_results = hands_detector.process(rgb)

        results = last_results

        # Variables temporales para el cálculo, para no modificar val_izq/der antes del cálculo final
        val_izq_detectado = None
        val_der_detectado = None

        if results and getattr(results, 'multi_hand_landmarks', None):
            for lm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                area = hand_bbox_area(lm.landmark, w, h)
                if area < img_area * MIN_HAND_AREA_RATIO:
                    continue

                wrist_x = lm.landmark[0].x
                pos_label = 'Left' if wrist_x < 0.5 else 'Right'
                label_final = pos_label

                if label_final == 'Right':
                    fingers = count_fingers(lm.landmark, hand_label='Right', smoothing_deque=right_finger_smooth)
                    val_der_detectado = fingers
                    right_index_tip = (lm.landmark[8].x, lm.landmark[8].y)
                    mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS, draw_spec_land, draw_spec_conn)
                else:
                    fingers = count_fingers(lm.landmark, hand_label='Left', smoothing_deque=left_finger_smooth)
                    val_izq_detectado = fingers
                    left_index_tip = (lm.landmark[8].x, lm.landmark[8].y)
                    mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS, draw_spec_land, draw_spec_conn)

        # --- LÓGICA DE CERO EXCLUSIVO (Puño Cerrado) ---
        if val_izq_detectado is not None or val_der_detectado is not None:
            v_izq = val_izq_detectado if val_izq_detectado is not None else 0
            v_der = val_der_detectado if val_der_detectado is not None else 0
            total_fingers = v_izq + v_der

        val_izq = val_izq_detectado
        val_der = val_der_detectado

        disp = cv2.flip(frame, 1)
        dh, dw, _ = disp.shape

        # Determinar el color y texto del modo
        if mode == MODE_CALCULATOR:
            mode_txt = "MODO: CALCULADORA"
            mode_color = (0, 255, 0)
        elif mode == MODE_POINTER:
            mode_txt = "MODO: PUNTERO"
            mode_color = (0, 100, 255)
        elif mode == MODE_FRACTION:
            mode_txt = "MODO: FRACCIONES"
            mode_color = (255, 165, 0)

        draw_hud_panel(disp, 0, 0, dw, 80, (20, 20, 20), 0.8)
        cv2.putText(disp, mode_txt, (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, mode_color, 2)
        cv2.putText(disp, "[Tecla 'P' para cambiar]", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        if is_negative_current_input:
            draw_hud_panel(disp, dw//2 - 100, 10, 200, 60, (0, 0, 150), 0.8)
            cv2.putText(disp, "NEGATIVO (-)", (dw//2 - 80, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)


        # === BLOQUE BOTONES ===
        BTN_W = 220
        BTN_H = 70
        TOP_PADDING = 80 + 12
        LEFT_MARGIN = 30
        RIGHT_MARGIN = 30
        CENTER_X = (dw - BTN_W) // 2

        box_neg = (LEFT_MARGIN, TOP_PADDING, LEFT_MARGIN + BTN_W, TOP_PADDING + BTN_H)
        box_res = (dw - RIGHT_MARGIN - BTN_W, TOP_PADDING, dw - RIGHT_MARGIN, TOP_PADDING + BTN_H)
        box_frac = (CENTER_X, TOP_PADDING, CENTER_X + BTN_W, TOP_PADDING + BTN_H)

        hover_button = None

        # Lógica de Puntero Unificado
        current_tip = None
        if right_index_tip:
            current_tip = right_index_tip
        elif left_index_tip:
            current_tip = left_index_tip

        if current_tip:
            rx, ry = current_tip
            ix = int(rx * dw)
            iy = int(ry * dh)
            px = dw - ix
            py = iy

            pointer_color = (0, 255, 255) if mode == MODE_POINTER else (0, 255, 0)
            cv2.line(disp, (px-10, py), (px+10, py), pointer_color, 2)
            cv2.line(disp, (px, py-10), (px, py+10), pointer_color, 2)
            cv2.circle(disp, (px, py), 8, pointer_color, 2)

            if mode == MODE_POINTER:
                if box_neg[0] < px < box_neg[2] and box_neg[1] < py < box_neg[3]:
                    hover_button = 'N'
                elif box_res[0] < px < box_res[2] and box_res[1] < py < box_res[3]:
                    hover_button = 'R'
                elif box_frac[0] < px < box_frac[2] and box_frac[1] < py < box_frac[3]:
                    hover_button = 'F'

        # voz al entrar en hover
        if hover_button and hover_button != last_hover_button:
            if hover_button == 'N':
                hablar("Sobre botón Negativo")
            elif hover_button == 'R':
                hablar("Sobre botón Reiniciar")
            elif hover_button == 'F':
                hablar("Sobre botón Fracción")
        last_hover_button = hover_button

        # lógica dwell
        if hover_button:
            if active_button_key == hover_button:
                button_confirm_counter += 1
            else:
                active_button_key = hover_button
                button_confirm_counter = 1

            if button_confirm_counter >= BUTTON_CONFIRM_FRAMES:
                if hover_button == 'N':
                    is_negative_current_input = not is_negative_current_input
                    estado = "activado" if is_negative_current_input else "desactivado"
                    hablar(f"Signo negativo {estado}")
                elif hover_button == 'R':
                    valor1 = None; valor2 = None; operacion = None; resultado = None
                    num_v1 = None; den_v1 = None; num_v2 = None; den_v2 = None; num_res = None; den_res = None
                    if is_fraction_mode:
                        fase = "pedir_num1"
                        hablar("Sistema de fracciones reiniciado.")
                    else:
                        fase = "pedir_v1"
                        hablar("Sistema reiniciado.")
                elif hover_button == 'F':
                    valor1 = None; valor2 = None; operacion = None; resultado = None
                    num_v1 = None; den_v1 = None; num_v2 = None; den_v2 = None; num_res = None; den_res = None
                    is_fraction_mode = not is_fraction_mode
                    if is_fraction_mode:
                        mode = MODE_FRACTION
                        fase = "pedir_num1"
                        hablar("Modo Fracciones activado.")
                    else:
                        mode = MODE_CALCULATOR
                        fase = "pedir_v1"
                        hablar("Modo Calculadora Simple activado.")
                button_confirm_counter = 0
                active_button_key = None
        else:
            button_confirm_counter = 0
            active_button_key = None

        # dibujar botones fijos
        def draw_button_fixed(img, box, text, key, is_hover, progress=0, active_state=False):
            x1, y1, x2, y2 = box
            if key == 'N':
                color = (0, 165, 255) if active_state else ((0, 100, 200) if is_hover else (60, 60, 60))
            elif key == 'F':
                color = (255, 100, 0) if active_state else ((200, 80, 0) if is_hover else (60, 60, 60))
            else:
                color = (0, 0, 200) if is_hover else (60, 60, 60)
            if mode != MODE_POINTER and key in ('N', 'F'):
                 color = (30, 30, 30)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            if progress > 0 and mode == MODE_POINTER:
                fill = int((x2-x1)*progress)
                cv2.rectangle(img, (x1, y2-8), (x1+fill, y2), (255,255,255), -1)
            thickness = 2 if is_hover else 1
            cv2.rectangle(img, (x1, y1), (x2, y2), (200,200,200), thickness)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            tx = x1 + (x2 - x1 - text_size[0]) // 2
            ty = y1 + (y2 - y1 + text_size[1]) // 2
            cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        prog_n = (button_confirm_counter / BUTTON_CONFIRM_FRAMES) if active_button_key == 'N' else 0
        prog_r = (button_confirm_counter / BUTTON_CONFIRM_FRAMES) if active_button_key == 'R' else 0
        prog_f = (button_confirm_counter / BUTTON_CONFIRM_FRAMES) if active_button_key == 'F' else 0

        txt_neg = "NEGATIVO [ON]" if is_negative_current_input else "NEGATIVO"
        draw_button_fixed(disp, box_neg, txt_neg, 'N', hover_button == 'N', prog_n, is_negative_current_input)
        draw_button_fixed(disp, box_res, "REINICIAR", 'R', hover_button == 'R', prog_r)
        txt_frac = "FRACCIÓN [ON]" if is_fraction_mode else "FRACCIÓN"
        draw_button_fixed(disp, box_frac, txt_frac, 'F', hover_button == 'F', prog_f, is_fraction_mode)

        # texto de estado (NEGRO)
        status_font = cv2.FONT_HERSHEY_SIMPLEX
        status_scale = 0.55
        status_th = 1

        neg_status = "ACTIVADO" if is_negative_current_input else "DESACTIVADO"
        sx_neg = box_neg[0]
        sy_neg = box_neg[3] + 22
        col_neg = (0, 0, 0)
        cv2.putText(disp, f"NEGATIVO: {neg_status}", (sx_neg, sy_neg), status_font, status_scale, col_neg, status_th)
        frac_status = "ACTIVADO" if is_fraction_mode else "DESACTIVADO"
        sx_frac = box_frac[0]
        sy_frac = box_frac[3] + 22
        col_frac = (0, 0, 0)
        cv2.putText(disp, f"FRACCIÓN: {frac_status}", (sx_frac, sy_frac), status_font, status_scale, col_frac, status_th)

        rx = box_res[0]
        ry = box_res[3] + 22
        col_res = (0, 0, 0)
        cv2.putText(disp, "REINICIAR: LISTO", (rx, ry), status_font, status_scale, col_res, status_th)

        # --- 2. LÓGICA DE CALCULADORA ---
        if mode == MODE_CALCULATOR or mode == MODE_FRACTION:
            current_input = None
            is_op_phase = "operacion" in fase
            is_val_phase = "v" in fase or "num" in fase or "den" in fase

            if is_val_phase:
                if total_fingers is not None and 0 <= total_fingers <= 10:
                    current_input = total_fingers
            elif is_op_phase:
                if val_izq is not None and val_izq in OP_MAP:
                    if val_der is None or val_der == 0:
                        current_input = val_izq

            if fase not in ("resultado_mostrado_final", "calcular_resultado_fraccion", "calcular_resultado"):
                if current_input is not None:
                    hand_lost_counter = 0
                    if temp_val is None:
                        temp_val = current_input
                        val_name = str(temp_val)
                        if is_op_phase:
                            val_name = OP_MAP.get(temp_val,'?')
                        elif "num" in fase:
                            val_name = f"numerador {temp_val}"
                        elif "den" in fase:
                            val_name = f"denominador {temp_val}"
                        hablar(f"Detectado {val_name}. Mantén fijo.")
                    if current_input == temp_val:
                        confirm_counter += 1
                        inconsistent_counter = 0
                    else:
                        inconsistent_counter += 1
                        if inconsistent_counter > INCONSISTENCY_TOLERANCE:
                            temp_val = current_input
                            confirm_counter = 0
                            inconsistent_counter = 0
                            val_name = str(temp_val)
                            if is_op_phase:
                                val_name = OP_MAP.get(temp_val,'?')
                            elif "num" in fase:
                                val_name = f"numerador {temp_val}"
                            elif "den" in fase:
                                val_name = f"denominador {temp_val}"
                            hablar(f"Cambio a {val_name}.")
                    if confirm_counter >= CONFIRM_FRAMES:
                        raw = temp_val
                        is_value = ("num" in fase or "v" in fase)
                        final_val = raw
                        if is_value and raw > 0 and is_negative_current_input:
                             final_val = -raw

                        if mode == MODE_FRACTION:
                            if fase == "pedir_num1":
                                num_v1 = final_val
                                fase = "pedir_den1"
                                hablar(f"Numerador uno guardado: {num_v1}. Dame el denominador.")
                            elif fase == "pedir_den1":
                                den_v1 = raw
                                if den_v1 == 0:
                                    hablar("Denominador no puede ser cero. Intenta de nuevo.")
                                    den_v1 = None
                                    temp_val = None; confirm_counter = 0
                                    continue
                                fase = "pedir_operacion"
                                hablar(f"Denominador uno guardado: {den_v1}. Elige operación.")
                            elif "operacion" in fase:
                                operacion = OP_MAP.get(raw, None)
                                if operacion == "%":
                                    hablar("El operador de porcentaje no es compatible con el modo fracciones. Elige otra operación.")
                                    temp_val = None; confirm_counter = 0
                                    continue
                                fase = "pedir_num2"
                                hablar(f"Operación {operacion} seleccionada. Dame el numerador dos.")
                            elif fase == "pedir_num2":
                                num_v2 = final_val
                                fase = "pedir_den2"
                                hablar(f"Numerador dos guardado: {num_v2}. Dame el denominador dos.")
                            elif fase == "pedir_den2":
                                den_v2 = raw
                                if den_v2 == 0:
                                    hablar("Denominador no puede ser cero. Intenta de nuevo.")
                                    den_v2 = None
                                    temp_val = None; confirm_counter = 0
                                    continue
                                fase = "calcular_resultado_fraccion"
                                hablar(f"Denominador dos guardado: {den_v2}.")
                        else:
                            if "v1" in fase:
                                valor1 = final_val
                                fase = "pedir_operacion"
                                hablar(f"Valor uno guardado: {valor1}. Elige operación.")
                            elif "operacion" in fase:
                                operacion = OP_MAP.get(raw, None)
                                fase = "pedir_v2"
                                hablar(f"Operación {operacion} seleccionada. Dame el segundo valor.")
                            elif "v2" in fase:
                                valor2 = final_val
                                fase = "calcular_resultado"
                                hablar(f"Valor dos guardado: {valor2}.")
                        temp_val = None; confirm_counter = 0
                        if mode == MODE_CALCULATOR:
                             is_negative_current_input = False
                else:
                    hand_lost_counter += 1
                    if hand_lost_counter > HAND_LOSS_TOLERANCE:
                        temp_val = None; confirm_counter = 0
                        hand_lost_counter = 0
                        is_op_phase = "operacion" in fase
                        is_val_phase = "v" in fase or "num" in fase or "den" in fase
                        if is_op_phase or is_val_phase:
                             hablar("Gesto perdido o entrada no válida.")
            # Lógica de cálculo final (Enteros)
            if fase == "calcular_resultado":
                try:
                    if operacion == "+":
                        resultado = valor1 + valor2
                    elif operacion == "-":
                        resultado = valor1 - valor2
                    elif operacion == "*":
                        resultado = valor1 * valor2
                    elif operacion == "/":
                        resultado = round(valor1 / valor2, 2) if valor2 != 0 else "Error"
                    elif operacion == "^":
                        resultado = valor1 ** valor2
                    if resultado == "Error":
                        hablar("Error de cálculo o división por cero.")
                    else:
                        hablar(f"El resultado es: {resultado}")
                except Exception:
                    resultado = "Error"
                    hablar("Error de cálculo.")
                fase = "resultado_mostrado_final"

            # Lógica de cálculo final (Fracciones)
            if fase == "calcular_resultado_fraccion":
                try:
                    if den_v1 == 0 or den_v2 == 0:
                        num_res = "Error"; den_res = 0
                    elif operacion == "+":
                        num = num_v1 * den_v2 + num_v2 * den_v1
                        den = den_v1 * den_v2
                        num_res, den_res = simplificar_fraccion(num, den)
                    elif operacion == "-":
                        num = num_v1 * den_v2 - num_v2 * den_v1
                        den = den_v1 * den_v2
                        num_res, den_res = simplificar_fraccion(num, den)
                    elif operacion == "*":
                        num = num_v1 * num_v2
                        den = den_v1 * den_v2
                        num_res, den_res = simplificar_fraccion(num, den)
                    elif operacion == "/":
                        num = num_v1 * den_v2
                        den = den_v1 * num_v2
                        if den == 0: num_res = "Error"; den_res = 0
                        else: num_res, den_res = simplificar_fraccion(num, den)
                    if den_res == 0:
                         hablar("Error: División por cero.")
                    else:
                        hablar(f"El resultado es: {num_res} sobre {den_res}")
                except Exception:
                    num_res = "Error"; den_res = 0
                    hablar("Error de cálculo en fracciones.")
                fase = "resultado_mostrado_final"

        else:
            temp_val = None; confirm_counter = 0

        # --- 3. DIBUJAR UI AVANZADA ---
        panel_w = 550
        panel_h = 220
        panel_x = (dw - panel_w) // 2
        panel_y = (dh - panel_h) // 2 - 20

        draw_hud_panel(disp, panel_x, panel_y, panel_w, panel_h, (40, 40, 40), 0.7)

        if mode == MODE_CALCULATOR:
            def get_current_val_txt(phase_name, stored_val, is_current):
                txt = f"Val: {stored_val}" if stored_val is not None else "Val: --"
                if phase_name in fase and is_current and temp_val is not None:
                    disp_val = temp_val
                    disp_sign = '-' if is_negative_current_input and disp_val > 0 else ''
                    txt = f"Val: {disp_sign}{disp_val}"
                return txt

            t_v1 = get_current_val_txt("v1", valor1, fase == "pedir_v1")
            t_op = f"Op: {operacion}" if operacion is not None else "Op:  --"
            if "operacion" in fase and temp_val is not None:
                t_op = f"Op: {OP_MAP.get(temp_val,'?')}"
            t_v2 = get_current_val_txt("v2", valor2, fase == "pedir_v2")
            t_res = str(resultado) if resultado is not None else ""
            cv2.putText(disp, t_v1, (panel_x+20, panel_y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)
            cv2.putText(disp, t_op, (panel_x+20, panel_y+100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)
            cv2.putText(disp, t_v2, (panel_x+20, panel_y+150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)
            if resultado is not None:
                 cv2.putText(disp, f"= {t_res}", (panel_x+20, panel_y+200), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 2)

        elif mode == MODE_FRACTION:
            def format_frac_val(val, sign_active, is_num, current_phase):
                if current_phase == fase and temp_val is not None:
                    val = temp_val
                    sign_active = is_negative_current_input
                if val is not None:
                    disp_sign = '-' if sign_active and is_num and val > 0 else ''
                    return f"{disp_sign}{val}"
                return "--"

            n1_txt = format_frac_val(num_v1, is_negative_current_input, True, "pedir_num1")
            d1_txt = format_frac_val(den_v1, False, False, "pedir_den1")
            op_txt = operacion if operacion is not None else "?"
            if "operacion" in fase and temp_val is not None:
                op_txt = OP_MAP.get(temp_val,'?')
            n2_txt = format_frac_val(num_v2, is_negative_current_input, True, "pedir_num2")
            d2_txt = format_frac_val(den_v2, False, False, "pedir_den2")
            res_txt = ""
            if num_res is not None and den_res is not None:
                if num_res == "Error": res_txt = "Error"
                else: res_txt = f"{num_res}/{den_res}"
            col_frac = (255, 165, 0)
            cv2.putText(disp, n1_txt, (panel_x+30, panel_y+40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col_frac, 2)
            cv2.line(disp, (panel_x+20, panel_y+50), (panel_x+100, panel_y+50), col_frac, 2)
            cv2.putText(disp, d1_txt, (panel_x+30, panel_y+80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col_frac, 2)
            cv2.putText(disp, op_txt, (panel_x+130, panel_y+60), cv2.FONT_HERSHEY_DUPLEX, 1.2, (200,200,200), 2)
            cv2.putText(disp, n2_txt, (panel_x+200, panel_y+40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col_frac, 2)
            cv2.line(disp, (panel_x+190, panel_y+50), (panel_x+270, panel_y+50), col_frac, 2)
            cv2.putText(disp, d2_txt, (panel_x+200, panel_y+80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col_frac, 2)
            cv2.putText(disp, "=", (panel_x+320, panel_y+60), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 2)
            if res_txt:
                cv2.putText(disp, res_txt, (panel_x+360, panel_y+60), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 2)

        if temp_val is not None and confirm_counter > 0:
            bar_w_main = 500
            fill = int((confirm_counter/CONFIRM_FRAMES)*bar_w_main)
            bx = panel_x + 20
            by = panel_y + panel_h + 10
            cv2.rectangle(disp, (bx, by), (bx+bar_w_main, by+10), (100,100,100), -1)
            cv2.rectangle(disp, (bx, by), (bx+fill, by+10), (0,255,0), -1)
            cv2.putText(disp, "Procesando...", (bx, by+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,255,150), 1)

        # Instrucción Inferior
        msg_inst = ""
        if mode == MODE_POINTER:
            msg_inst = "MODO PUNTERO: Usa el índice para interactuar con los botones."
        elif mode == MODE_CALCULATOR:
            if fase == "pedir_v1":
                msg_inst = "Muestra DEDOS (0-10) para el VALOR 1."
            elif fase == "pedir_operacion":
                msg_inst = "Mano IZQUIERDA para OPERACION (1-5)."
            elif fase == "pedir_v2":
                msg_inst = "Muestra DEDOS (0-10) para el VALOR 2."
            elif "resultado" in fase:
                msg_inst = "Calculo finalizado. Reinicia."
        elif mode == MODE_FRACTION:
            if fase == "pedir_num1":
                msg_inst = "Muestra DEDOS (0-10) para el NUMERADOR 1."
            elif fase == "pedir_den1":
                msg_inst = "Muestra DEDOS (1-10) para el DENOMINADOR 1."
            elif fase == "pedir_operacion":
                msg_inst = "Mano IZQUIERDA para OPERACION (1-4)."
            elif fase == "pedir_num2":
                msg_inst = "Muestra DEDOS (0-10) para el NUMERADOR 2."
            elif fase == "pedir_den2":
                msg_inst = "Muestra DEDOS (1-10) para el DENOMINADOR 2."
            elif "resultado" in fase:
                msg_inst = "Calculo de fracciones finalizado. Reinicia."

        cv2.putText(disp, msg_inst, (dw//2 - 200, dh - 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        op_info = "GESTOS OPERADORES (Mano Derecha): 1: + | 2: - | 3: * | 4: / | 5: ^"
        draw_hud_panel(disp, dw//2 - 380, dh - 60, 760, 40, (0, 0, 0), 0.7)
        cv2.putText(disp, op_info, (dw//2 - 370, dh - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("VMR Beta 0.8", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('p') or key == ord('P'):
            temp_val = None; confirm_counter = 0;
            if mode == MODE_CALCULATOR:
                mode = MODE_POINTER
                txt_mode = "Modo puntero activado"
            elif mode == MODE_POINTER:
                if is_fraction_mode:
                    mode = MODE_FRACTION
                    fase = "pedir_num1"
                    txt_mode = "Modo fracciones activado"
                else:
                    mode = MODE_CALCULATOR
                    fase = "pedir_v1"
                    txt_mode = "Modo calculadora activado"
            elif mode == MODE_FRACTION:
                mode = MODE_POINTER
                txt_mode = "Modo puntero activado"
            hablar(txt_mode)

        if key == 27 or key == ord('q'):
            break

except Exception as e:
    print("Error en el loop principal:", e)
    import traceback
    traceback.print_exc()
finally:
    try:
        # Cierre ordenado del hilo TTS
        _shutdown_flag.set()
        try: tts_q.put_nowait(None)
        except: pass
        try: tts_thread.join(timeout=2.0)
        except: pass
    except Exception:
        pass
    cap.release()
    cv2.destroyAllWindows()