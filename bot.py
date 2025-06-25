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

# --- Nova Variável para Velocidade de Clique ---
# Ajuste este valor para controlar a velocidade do bot.
# Valores menores = cliques mais rápidos; Valores maiores = cliques mais lentos.
# Para o mais rápido possível, tente 0.0001 ou 0.001.
CLICK_DELAY_SECONDS = 0.0001 # OTIMIZADO PARA VELOCIDADE MÁXIMA

# --- Caminho dinâmico para o Sprite Sheet ---
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SPRITE_SHEET_PATH = os.path.join(SCRIPT_DIR, "tiles", "cloneskin.png")
SPRITE_SHEET = None

# Título da janela (Verifique se o título da sua janela é exatamente este)
MINESWEEPER_WINDOW_TITLE = "Minesweeper"

# --- Coordenadas corrigidas para o seu sprite sheet 'cloneskin.png' ---
TILE_MAP_COORDS_ORIGINAL = {
    # Números de 1-8
    "1": (16, 0, 16, 16),
    "2": (32, 0, 16, 16),
    "3": (48, 0, 16, 16),
    "4": (64, 0, 16, 16),
    "5": (80, 0, 16, 16),
    "6": (96, 0, 16, 16),
    "7": (112, 0, 16, 16),
    "8": (128, 0, 16, 16), 
    
    # Estados das células
    "empty_zero_revealed": (0, 0, 16, 16),
    "closed": (0, 16, 16, 16),
    
    "flag": (48, 16, 16, 16), 
    
    "mine_unrevealed": (16, 16, 16, 16),
    "mine_red_exploded": (64, 16, 16, 16),
    "one_red_background": (32, 16, 16, 16), 
    "question_mark": (80, 16, 16, 16),
}

FACE_MAP_COORDS_ORIGINAL = {
    "face_happy":         (0, 55, 25, 25),
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
        return None # Retorna None se o nome do tile não for encontrado
    
    return SPRITE_SHEET[y:y+h, x:x+w]

def get_calibrated_template(tile_name):
    """Retorna um template calibrado."""
    if tile_name not in CALIBRATED_TEMPLATES:
        template_path = os.path.join(CALIBRATED_TEMPLATES_DIR, f"{tile_name}.png")
        
        if os.path.exists(template_path):
            template = cv2.imread(template_path)
        elif global_cell_size > 0:
            original = extract_template_from_sprite(tile_name)
            if original is None:
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
            template = extract_template_from_sprite(tile_name)

        CALIBRATED_TEMPLATES[tile_name] = template
    
    return CALIBRATED_TEMPLATES[tile_name]

def screenshot_game_window():
    """Captura screenshot da janela do jogo."""
    try:
        windows = pyautogui.getWindowsWithTitle(MINESWEEPER_WINDOW_TITLE)
        if not windows:
            print(f"AVISO: Janela com o título '{MINESWHEEPER_WINDOW_TITLE}' não encontrada.")
            return None, (0, 0, 0, 0)
        
        window = windows[0]
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
    
    pyautogui.moveTo(center_x, center_y, duration=0.01) # Movimento mais rápido
    pyautogui.click(button=button)
    time.sleep(CLICK_DELAY_SECONDS) # Usa a nova variável de atraso

def is_game_over(image):
    """Verifica se o jogo terminou e qual o resultado."""
    face_dead = get_calibrated_template("face_dead")
    face_win = get_calibrated_template("face_win")
    
    if face_dead is not None and len(find_template_matches(image, face_dead, 0.95)) > 0:
        return "lose"
    if face_win is not None and len(find_template_matches(image, face_win, 0.95)) > 0:
        return "win"
        
    return "playing" # Jogo em andamento

def build_board(image):
    """Constrói o tabuleiro a partir da imagem."""
    global global_tabuleiro_offset_x, global_tabuleiro_offset_y, global_cell_size
    
    cell_types = {
        "1": {"threshold": 0.8}, "2": {"threshold": 0.8},
        "3": {"threshold": 0.8}, "4": {"threshold": 0.8},
        "5": {"threshold": 0.8}, "6": {"threshold": 0.8},
        "7": {"threshold": 0.8}, "8": {"threshold": 0.8},
        "closed": {"threshold": 0.85}, 
        "empty_zero_revealed": {"threshold": 0.9}, 
        "flag": {"threshold": 0.85}, 
        "one_red_background": {"threshold": 0.8} 
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
            "empty_zero_revealed": 5, "flag": 7, "closed": 1, 
            "one_red_background": 6 
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
    
    # Regra 1: Marcar minas
    for row, col, cell in numbered_cells:
        number = int(cell["type"])
        neighbors = get_neighbors(board, row, col)
        
        closed_neighbors = [n for n in neighbors if n[2]["type"] == "closed"]
        flagged_neighbors = [n for n in neighbors if n[2]["type"] == "flag"]
        
        # Se o número de células fechadas restantes for igual ao número de minas a serem marcadas
        if len(closed_neighbors) == (number - len(flagged_neighbors)) and len(closed_neighbors) > 0:
            for nr, nc, ncell in closed_neighbors:
                if board[nr][nc]["type"] == "closed": 
                    click_center(ncell["x"], ncell["y"], ncell["w"], ncell["h"], 
                               window_x, window_y, "right") 
                    print(f"🚩 Marcado ({nr}, {nc}) como mina")
                    board[nr][nc]["type"] = "flag" 
                    made_move = True
                    return board, True 

    # Regra 2: Revelar células seguras
    for row, col, cell in numbered_cells:
        number = int(cell["type"])
        neighbors = get_neighbors(board, row, col)
        
        closed_neighbors = [n for n in neighbors if n[2]["type"] == "closed"]
        flagged_neighbors = [n for n in neighbors if n[2]["type"] == "flag"]
        
        # Se o número de bandeiras vizinhas for igual ao número da célula, 
        # as células fechadas restantes são seguras.
        if len(flagged_neighbors) == number and len(closed_neighbors) > 0:
            for nr, nc, ncell in closed_neighbors:
                click_center(ncell["x"], ncell["y"], ncell["w"], ncell["h"], 
                           window_x, window_y, "left")
                print(f"✅ Revelado ({nr}, {nc}) por lógica determinística")
                made_move = True
                return board, True 
    
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

    matches = find_template_matches(image, original_closed, 0.7)
    
    if not matches:
        print("Não foi possível encontrar células fechadas para calibração. Verifique se o jogo está aberto e visível.")
        return False
    
    x, y, w, h = matches[0]
    global_cell_size = w
    
    calibrated_closed = image[y:y+h, x:x+w]
    cv2.imwrite(os.path.join(CALIBRATED_TEMPLATES_DIR, "closed.png"), calibrated_closed)
    
    print(f"Templates calibrados! Tamanho da célula detectado: {global_cell_size}x{global_cell_size}")
    all_templates = list(TILE_MAP_COORDS_ORIGINAL.keys()) + list(FACE_MAP_COORDS_ORIGINAL.keys())
    for name in all_templates:
        get_calibrated_template(name)

    return True

# --- LOOP PRINCIPAL ---
def main():
    global global_tabuleiro_offset_x, global_tabuleiro_offset_y, global_cell_size, CALIBRATED_TEMPLATES
    
    try:
        load_sprite_sheet()
    except (FileNotFoundError, IOError) as e:
        print(f"❌ Erro fatal: {e}")
        return

    print("🤖 Bot Campo Minado iniciando...")
    time.sleep(0.1) # Reduzido

    
    if not os.path.exists(os.path.join(CALIBRATED_TEMPLATES_DIR, "closed.png")):
        if not calibrate_templates():
            print("❌ Falha na calibração inicial. Encerrando.")
            return
    else:
        # Se os templates já existem, não recalibra, apenas carrega o tamanho da célula
        closed_template = cv2.imread(os.path.join(CALIBRATED_TEMPLATES_DIR, "closed.png"))
        if closed_template is not None:
            global_cell_size = closed_template.shape[0]
            print(f"Tamanho da célula carregado da calibração anterior: {global_cell_size}x{global_cell_size}")
        else:
             # Caso o arquivo calibrated_templates/closed.png esteja corrompido, tenta recalibrar
             print("AVISO: Template 'closed.png' corrompido ou inválido. Tentando recalibrar.")
             if not calibrate_templates():
                print("❌ Falha ao tentar recalibrar. Encerrando.")
                return

    first_move_made = False
    
    while True:
        try:
            image, window_bbox = screenshot_game_window()
            if image is None:
                time.sleep(0.1) # Reduzido, mas mantido para evitar loop em caso de janela não encontrada
                continue
            
            window_x, window_y, _, _ = window_bbox
            
            game_status = is_game_over(image)
            if game_status == "win":
                print("🎉🎉🎉 JOGO VENCIDO! 🎉🎉🎉")
                print("Bot encerrado após a vitória.")
                break # Encerra o loop e o bot
            elif game_status == "lose":
                print("💀 JOGO PERDIDO! 💀")
                print("Bot encerrado após a derrota.")
                break # Encerra o loop e o bot
            
            # Se o jogo não terminou (game_status == "playing"), continua o fluxo
            
            board = build_board(image)
            if not board:
                # Sem um board válido, a única chance é que seja o primeiro movimento ou um erro de detecção
                # Aumentamos um pouco o atraso aqui para dar tempo para a janela do jogo carregar completamente
                # antes de tentar novamente, mas ainda rápido.
                print("⚠️ Não foi possível construir o tabuleiro. Tentando novamente...")
                time.sleep(0.05) # Reduzido
                continue
            
            if not first_move_made:
                if make_first_move(board, window_x, window_y):
                    first_move_made = True
                    # A pausa após o primeiro clique é crucial para o jogo reagir e revelar o tabuleiro inicial.
                    # Mantenha um pequeno atraso aqui, talvez um pouco mais que o CLICK_DELAY_SECONDS básico.
                    time.sleep(CLICK_DELAY_SECONDS * 5) # Ajuste para permitir que o tabuleiro se revele
                continue
            
            board, made_deterministic_move = solve_deterministic(board, window_x, window_y)
            
            if made_deterministic_move:
                # Não há necessidade de sleep extra aqui, a próxima iteração do loop fará um novo screenshot imediatamente.
                continue
            
            # Se não houve movimento determinístico
            closed_cells_remaining = sum(1 for row_data in board.values() for cell in row_data.values() if cell["type"] == "closed")
            if closed_cells_remaining == 0:
                print("✅ Não há mais células fechadas. O jogo deveria ter terminado.")
                # Pausa para o usuário ver o fim do jogo, se o bot não o detectar imediatamente.
                # Para velocidade máxima, pode ser 0, mas se o jogo não terminar em 100% dos casos, é bom para depurar.
                time.sleep(0.1) 
                # Não break, pois o loop principal vai verificar is_game_over novamente e vai parar se for win/lose
            else:
                # Se não há movimentos determinísticos e ainda há células fechadas, faz um movimento aleatório
                # Sem sleep extra aqui, a próxima iteração do loop fará um novo screenshot imediatamente.
                make_random_move(board, window_x, window_y)

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