import cv2
import numpy as np
import pyautogui
import time
import os
from itertools import combinations
from collections import defaultdict
import math

# --- Configurações Globais ---
global_tabuleiro_offset_x = -1
global_tabuleiro_offset_y = -1
global_cell_size = -1

# --- CORREÇÃO: Caminho dinâmico para o Sprite Sheet ---
# Isso torna o script portátil. Ele vai procurar a pasta 'tiles' no mesmo diretório do bot.py
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SPRITE_SHEET_PATH = os.path.join(SCRIPT_DIR, "tiles", "cloneskin.png")
SPRITE_SHEET = None

# Título da janela (Verifique se o título da sua janela é exatamente este)
MINESWEEPER_WINDOW_TITLE = "Minesweeper"

# --- CORREÇÃO: Coordenadas corrigidas para o seu sprite sheet 'cloneskin.png' ---
TILE_MAP_COORDS_ORIGINAL = {
    # Números de 1-8
    "1": (16, 0, 16, 16),
    "2": (32, 0, 16, 16),
    "3": (48, 0, 16, 16),
    "4": (64, 0, 16, 16),
    "5": (80, 0, 16, 16),
    "6": (96, 0, 16, 16),
    "7": (112, 0, 16, 16),
    "8": (128, 0, 16, 16), # Embora a imagem não tenha um '8' claro, mantemos a posição esperada
    
    # Estados das células
    "empty_zero_revealed": (0, 0, 16, 16),
    "closed": (0, 16, 16, 16),
    
    # --- AVISO CRÍTICO: O template para 'flag' NÃO EXISTE na sua imagem 'cloneskin.png' ---
    # A coordenada (32, 16) na sua imagem é um "1 com fundo vermelho", não uma bandeira.
    # O bot NÃO CONSEGUIRÁ marcar bandeiras. A lógica de bandeiras foi desativada.
    # Para funcionar, você precisa de um sprite sheet com a imagem da bandeira.
    # "flag": (32, 16, 16, 16), # Esta linha está desativada intencionalmente.
    
    "mine_unrevealed": (16, 16, 16, 16),
    "mine_red_exploded": (64, 16, 16, 16),
    "wrong_flag": (48, 16, 16, 16),
    "question_mark": (80, 16, 16, 16),
}

FACE_MAP_COORDS_ORIGINAL = {
    "face_happy":         (0, 55, 25, 25), # Já está em x=0, não pode ir mais para a esquerda
    "face_mouth_open":    (27, 55, 25, 25),
    "face_dead":          (54, 55, 25, 25),
    "face_win":           (81, 55, 25, 25),
    "face_happy_pressed": (108, 55, 25, 25)
}

# Diretório para templates calibrados
CALIBRATED_TEMPLATES_DIR = "calibrated_templates"
CALIBRATED_TEMPLATES = {}

# Offsets de detecção (começamos com 0 e ajustamos dinamicamente)
DETECTION_OFFSET_X = 0
DETECTION_OFFSET_Y = 0

def load_sprite_sheet():
    """Carrega o sprite sheet uma única vez."""
    global SPRITE_SHEET
    if SPRITE_SHEET is None:
        if not os.path.exists(SPRITE_SHEET_PATH):
             raise FileNotFoundError(f"ERRO: Sprite sheet não encontrado em '{SPRITE_SHEET_PATH}'. Verifique se o arquivo 'cloneskin.png' está na pasta 'tiles'.")
        SPRITE_SHEET = cv2.imread(SPRITE_SHEET_PATH)
        if SPRITE_SHEET is None:
            raise IOError(f"Não foi possível carregar '{SPRITE_SHEET_PATH}' com o OpenCV.")
        print(f"Sprite sheet carregado: {SPRITE_SHEET.shape}")

def extract_template_from_sprite(tile_name):
    """Extrai um template do sprite sheet."""
    if tile_name in TILE_MAP_COORDS_ORIGINAL:
        x, y, w, h = TILE_MAP_COORDS_ORIGINAL[tile_name]
    elif tile_name in FACE_MAP_COORDS_ORIGINAL:
        x, y, w, h = FACE_MAP_COORDS_ORIGINAL[tile_name]
    else:
        # Retorna None em vez de erro para lidar com o caso da bandeira ausente
        return None
    
    return SPRITE_SHEET[y:y+h, x:x+w]

def get_calibrated_template(tile_name):
    """Retorna um template calibrado."""
    if tile_name not in CALIBRATED_TEMPLATES:
        template_path = os.path.join(CALIBRATED_TEMPLATES_DIR, f"{tile_name}.png")
        
        if os.path.exists(template_path):
            template = cv2.imread(template_path)
        elif global_cell_size > 0:
            original = extract_template_from_sprite(tile_name)
            if original is None: # Se o template não existe no sprite map
                CALIBRATED_TEMPLATES[tile_name] = None
                return None
            
            if tile_name in FACE_MAP_COORDS_ORIGINAL:
                _, _, orig_w, orig_h = FACE_MAP_COORDS_ORIGINAL[tile_name]
                template = cv2.resize(original, (orig_w, orig_h), interpolation=cv2.INTER_AREA)
            else:
                template = cv2.resize(original, (global_cell_size, global_cell_size), interpolation=cv2.INTER_AREA)
            
            os.makedirs(CALIBRATED_TEMPLATES_DIR, exist_ok=True)
            cv2.imwrite(template_path, template)
        else:
            # Tenta carregar do sprite sheet original se a calibração ainda não ocorreu
            template = extract_template_from_sprite(tile_name)

        CALIBRATED_TEMPLATES[tile_name] = template
    
    return CALIBRATED_TEMPLATES[tile_name]

def screenshot_game_window():
    """Captura screenshot da janela do jogo."""
    try:
        windows = pyautogui.getWindowsWithTitle(MINESWEEPER_WINDOW_TITLE)
        if not windows:
            print(f"AVISO: Janela com o título '{MINESWEEPER_WINDOW_TITLE}' não encontrada.")
            return None, (0, 0, 0, 0)
        
        window = windows[0]
        # Ativar a janela pode não ser necessário e pode atrapalhar
        # if not window.isActive:
        #     window.activate()
        #     time.sleep(0.1)
        
        region = (window.left, window.top, window.width, window.height)
        screenshot = pyautogui.screenshot(region=region)
        
        return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR), region
    
    except Exception as e:
        print(f"Erro ao capturar screenshot: {e}")
        return None, (0, 0, 0, 0)

def find_template_matches(image, template, threshold=0.8, debug_name=None):
    """Encontra todas as correspondências de um template na imagem."""
    if template is None or template.size == 0:
        return []
    
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    h, w = template.shape[:2]
    
    locations = np.where(result >= threshold)
    matches = []
    
    for pt in zip(*locations[::-1]):
        matches.append([int(pt[0]), int(pt[1]), w, h])
    
    if not matches:
        return []
    
    matches = np.array(matches)
    scores = [result[y, x] for x, y, _, _ in matches]
    
    indices = cv2.dnn.NMSBoxes(matches.tolist(), scores, threshold, 0.4)
    
    final_matches = []
    if len(indices) > 0:
        for i in indices.flatten():
            final_matches.append(tuple(matches[i]))
            
    return final_matches

def click_center(x, y, w, h, window_x=0, window_y=0, button='left'):
    """Clica no centro de uma área."""
    center_x = window_x + x + w // 2
    center_y = window_y + y + h // 2
    
    pyautogui.moveTo(center_x, center_y, duration=0.1)
    pyautogui.click(button=button)
    time.sleep(0.05)

def is_game_over(image):
    """Verifica se o jogo terminou."""
    face_dead = get_calibrated_template("face_dead")
    face_win = get_calibrated_template("face_win")
    
    if face_dead is not None and len(find_template_matches(image, face_dead, 0.85)) > 0:
        return True
    if face_win is not None and len(find_template_matches(image, face_win, 0.85)) > 0:
        return True
        
    return False

def build_board(image):
    """Constrói o tabuleiro a partir da imagem."""
    global global_tabuleiro_offset_x, global_tabuleiro_offset_y, global_cell_size
    
    cell_types = {
        "1": {"threshold": 0.8}, "2": {"threshold": 0.8},
        "3": {"threshold": 0.8}, "4": {"threshold": 0.8},
        "5": {"threshold": 0.8}, "6": {"threshold": 0.8},
        "7": {"threshold": 0.8}, "8": {"threshold": 0.8},
        "closed": {"threshold": 0.85},
        "empty_zero_revealed": {"threshold": 0.8},
        # A detecção de 'flag' foi removida pois o template não existe
    }
    
    all_cells = []
    
    for cell_type, config in cell_types.items():
        template = get_calibrated_template(cell_type)
        if template is None:
            continue
        
        matches = find_template_matches(image, template, config["threshold"], cell_type)
        for (x, y, w, h) in matches:
            all_cells.append((x, y, w, h, cell_type))
    
    if not all_cells:
        return {}
    
    if global_cell_size == -1:
        closed_cells = [c for c in all_cells if c[4] == "closed"]
        if closed_cells:
            x, y, w, h, _ = closed_cells[0]
            # Uma pequena margem para garantir que peguemos a célula inteira
            global_tabuleiro_offset_x = x 
            global_tabuleiro_offset_y = y
            global_cell_size = w
            print(f"Calibração: Offset=({x},{y}), Tamanho célula={w}x{h}")
    
    board = {}
    if global_cell_size <= 0: return {}

    for x, y, w, h, cell_type in all_cells:
        grid_col = round((x - global_tabuleiro_offset_x) / global_cell_size)
        grid_row = round((y - global_tabuleiro_offset_y) / global_cell_size)
        
        priority = {
            "1": 9, "2": 9, "3": 9, "4": 9, "5": 9, "6": 9, "7": 9, "8": 9,
            "empty_zero_revealed": 5, "closed": 1
        }
        
        cell_data = {
            "type": "empty" if cell_type == "empty_zero_revealed" else cell_type,
            "x": x, "y": y, "w": w, "h": h
        }
        
        if grid_row not in board:
            board[grid_row] = {}
        
        if grid_col not in board[grid_row] or priority.get(cell_type, 0) > priority.get(board[grid_row][grid_col]["type"], 0):
            board[grid_row][grid_col] = cell_data
    
    return board

def get_neighbors(board, row, col):
    """Retorna os vizinhos de uma célula."""
    neighbors = []
    for r in range(row - 1, row + 2):
        for c in range(col - 1, col + 2):
            if r == row and c == col: continue
            if r in board and c in board[r]:
                neighbors.append((r, c, board[r][c]))
    return neighbors

def solve_deterministic(board, window_x, window_y):
    """Aplica lógica determinística do campo minado."""
    made_move = False
    
    numbered_cells = []
    for row in board:
        for col in board[row]:
            cell = board[row][col]
            if cell["type"].isdigit():
                numbered_cells.append((row, col, cell))
    
    # Regra 1: Marcar minas (DESATIVADA PELA FALTA DO SPRITE DE BANDEIRA)
    # Sem conseguir identificar bandeiras, esta regra não pode ser usada.
    
    # Regra 2: Revelar células seguras
    for row, col, cell in numbered_cells:
        number = int(cell["type"])
        neighbors = get_neighbors(board, row, col)
        
        closed_neighbors = [n for n in neighbors if n[2]["type"] == "closed"]
        flagged_neighbors = [n for n in neighbors if n[2]["type"] == "flag"] # Sempre será 0
        
        # Clica em células seguras se todas as minas ao redor já estiverem marcadas
        # Como não marcamos, essa regra só funcionará para a borda da área revelada
        if len(flagged_neighbors) == number and len(closed_neighbors) > 0:
            for nr, nc, ncell in closed_neighbors:
                click_center(ncell["x"], ncell["y"], ncell["w"], ncell["h"], 
                           window_x, window_y, "left")
                print(f"✅ Revelado ({nr}, {nc}) por lógica determinística")
                # Não atualizamos o estado do tabuleiro aqui, vamos ler da tela novamente
            made_move = True
            return board, True # Retorna após a primeira ação para reavaliar o tabuleiro
    
    return board, made_move

def make_first_move(board, window_x, window_y):
    """Faz a primeira jogada no centro do tabuleiro."""
    if not board: return False
    
    rows = sorted(board.keys())
    if not rows: return False
    cols = sorted(set(c for r in board for c in board[r].keys()))
    if not cols: return False
    
    center_row = rows[len(rows) // 2]
    center_col = cols[len(cols) // 2]
    
    if center_row in board and center_col in board[center_row]:
        cell = board[center_row][center_col]
        if cell["type"] == "closed":
            click_center(cell["x"], cell["y"], cell["w"], cell["h"], 
                       window_x, window_y, "left")
            print(f"🖱️ Primeira jogada em ({center_row}, {center_col})")
            return True
    
    return make_random_move(board, window_x, window_y, first_move=True)


def make_random_move(board, window_x, window_y, first_move=False):
    """Faz uma jogada aleatória em uma célula fechada."""
    import random
    
    closed_cells = [cell for row in board.values() for cell in row.values() if cell["type"] == "closed"]
    
    if closed_cells:
        cell_to_click = random.choice(closed_cells)
        click_center(cell_to_click["x"], cell_to_click["y"], cell_to_click["w"], cell_to_click["h"], 
                   window_x, window_y, "left")
        if not first_move:
             print("🎲 Sem jogada segura, fazendo uma jogada aleatória.")
        return True
    
    return False

def calibrate_templates():
    """Calibra os templates iniciais."""
    global global_cell_size
    
    load_sprite_sheet()
    
    # Limpa templates antigos para forçar a recalibração
    if os.path.exists(CALIBRATED_TEMPLATES_DIR):
        import shutil
        shutil.rmtree(CALIBRATED_TEMPLATES_DIR)
    os.makedirs(CALIBRATED_TEMPLATES_DIR, exist_ok=True)
    
    print("Calibrando templates... Certifique-se que a janela do jogo está visível.")
    
    image, _ = screenshot_game_window()
    if image is None:
        print("Não foi possível capturar a janela para calibração.")
        return False
    
    original_closed = extract_template_from_sprite("closed")
    if original_closed is None:
        print("Template 'closed' não encontrado no sprite sheet.")
        return False

    # Tenta encontrar com um threshold mais baixo para a calibração inicial
    matches = find_template_matches(image, original_closed, 0.7)
    
    if not matches:
        print("Não foi possível encontrar células fechadas para calibração. Verifique se o jogo está aberto e visível.")
        return False
    
    x, y, w, h = matches[0]
    global_cell_size = w
    
    calibrated_closed = image[y:y+h, x:x+w]
    cv2.imwrite(os.path.join(CALIBRATED_TEMPLATES_DIR, "closed.png"), calibrated_closed)
    
    print(f"Templates calibrados! Tamanho da célula detectado: {global_cell_size}x{global_cell_size}")
    # Pré-gera todos os outros templates com o novo tamanho
    all_templates = list(TILE_MAP_COORDS_ORIGINAL.keys()) + list(FACE_MAP_COORDS_ORIGINAL.keys())
    for name in all_templates:
        get_calibrated_template(name)

    return True

def restart_game(image, window_x, window_y):
    """Reinicia o jogo clicando na face feliz ou na face morta/vencedora."""
    face_to_click = None
    for face_name in ["face_happy", "face_dead", "face_win"]:
        face_template = get_calibrated_template(face_name)
        if face_template is not None:
            matches = find_template_matches(image, face_template, 0.8)
            if matches:
                face_to_click = matches[0]
                break
    
    if face_to_click:
        x, y, w, h = face_to_click
        click_center(x, y, w, h, window_x, window_y, "left")
        print("🔄 Jogo reiniciado")
        return True
    
    print("AVISO: Não foi possível encontrar a face para reiniciar o jogo.")
    return False

# --- LOOP PRINCIPAL ---
def main():
    global global_tabuleiro_offset_x, global_tabuleiro_offset_y, global_cell_size, CALIBRATED_TEMPLATES
    
    try:
        load_sprite_sheet()
    except (FileNotFoundError, IOError) as e:
        print(f"❌ Erro fatal: {e}")
        return

    print("🤖 Bot Campo Minado iniciando...")
    time.sleep(2)
    
    if not os.path.exists(os.path.join(CALIBRATED_TEMPLATES_DIR, "closed.png")):
        if not calibrate_templates():
            print("❌ Falha na calibração inicial. Encerrando.")
            return
    else:
        # Carrega o tamanho da célula a partir do template já calibrado
        closed_template = cv2.imread(os.path.join(CALIBRATED_TEMPLATES_DIR, "closed.png"))
        if closed_template is not None:
            global_cell_size = closed_template.shape[0]
            print(f"Tamanho da célula carregado da calibração anterior: {global_cell_size}x{global_cell_size}")
        else:
             if not calibrate_templates():
                print("❌ Falha ao tentar recalibrar. Encerrando.")
                return

    first_move_made = False
    
    while True:
        try:
            image, window_bbox = screenshot_game_window()
            if image is None:
                time.sleep(2)
                continue
            
            window_x, window_y, _, _ = window_bbox
            
            if is_game_over(image):
                print("🏁 Jogo terminou!")
                time.sleep(1) # Pausa para ver o resultado
                restart_game(image, window_x, window_y)
                
                # Reseta o estado do bot para o novo jogo
                first_move_made = False
                global_tabuleiro_offset_x = -1
                global_tabuleiro_offset_y = -1
                board = {}
                time.sleep(1.5) # Espera a animação de reinício
                continue
            
            board = build_board(image)
            if not board:
                print("⚠️ Não foi possível construir o tabuleiro. Tentando novamente...")
                time.sleep(1)
                continue
            
            if not first_move_made:
                if make_first_move(board, window_x, window_y):
                    first_move_made = True
                    time.sleep(0.5) # Espera o tabuleiro abrir após o primeiro clique
                continue
            
            board, made_deterministic_move = solve_deterministic(board, window_x, window_y)
            
            if made_deterministic_move:
                time.sleep(0.4) # Pequena pausa após um movimento seguro
                continue
            
            # Se não há mais jogadas seguras, faz um movimento aleatório
            if not make_random_move(board, window_x, window_y):
                print("✅ Não há mais células fechadas. O jogo deveria ter terminado.")
                time.sleep(3)
            else:
                time.sleep(0.5) # Pausa após um movimento aleatório

        except KeyboardInterrupt:
            print("\n🛑 Bot interrompido pelo usuário.")
            break
        except Exception as e:
            print(f"💥 Ocorreu um erro inesperado no loop principal: {e}")
            import traceback
            traceback.print_exc()
            break


if __name__ == "__main__":
    main()