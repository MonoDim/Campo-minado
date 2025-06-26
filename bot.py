import cv2
import numpy as np
import pyautogui
import time
import os
from itertools import combinations
from collections import defaultdict

# --- OTIMIZAÇÃO: Velocidade Bruta ---
# Remove a pausa padrão do PyAutoGUI entre as ações para máxima velocidade.
pyautogui.PAUSE = 0

# --- Configurações Globais ---
global_tabuleiro_offset_x = -1
global_tabuleiro_offset_y = -1
global_cell_size = -1

# OTIMIZAÇÃO: Atraso de clique já otimizado, mantido para controle.
CLICK_DELAY_SECONDS = 0.0001

# --- Caminhos e Constantes ---
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SPRITE_SHEET_PATH = os.path.join(SCRIPT_DIR, "tiles", "cloneskin.png")
SPRITE_SHEET = None
MINESWEEPER_WINDOW_TITLE = "Minesweeper"

# Dicionários de coordenadas (mantidos como você ajustou)
TILE_MAP_COORDS_ORIGINAL = {
    "1": (16, 0, 16, 16), "2": (32, 0, 16, 16), "3": (48, 0, 16, 16),
    "4": (64, 0, 16, 16), "5": (80, 0, 16, 16), "6": (96, 0, 16, 16),
    "7": (112, 0, 16, 16), "8": (128, 0, 16, 16),
    "empty_zero_revealed": (0, 0, 16, 16), "closed": (0, 16, 16, 16),
    "flag": (48, 16, 16, 16), "mine_unrevealed": (16, 16, 16, 16),
    "mine_red_exploded": (64, 16, 16, 16), "one_red_background": (32, 16, 16, 16),
    "question_mark": (80, 16, 16, 16),
}
FACE_MAP_COORDS_ORIGINAL = {
    "face_happy": (0, 55, 25, 25), "face_mouth_open": (27, 55, 25, 25),
    "face_dead": (54, 55, 25, 25), "face_win": (81, 55, 25, 25),
    "face_happy_pressed": (108, 55, 25, 25)
}

CALIBRATED_TEMPLATES_DIR = "calibrated_templates"
CALIBRATED_TEMPLATES = {}

# As funções de setup (load, extract, calibrate) permanecem as mesmas
# Elas são eficientes e rodam principalmente fora do loop de jogo.
def load_sprite_sheet():
    global SPRITE_SHEET
    if SPRITE_SHEET is None:
        if not os.path.exists(SPRITE_SHEET_PATH):
            raise FileNotFoundError(f"ERRO: Sprite sheet não encontrado em '{SPRITE_SHEET_PATH}'.")
        SPRITE_SHEET = cv2.imread(SPRITE_SHEET_PATH)
        if SPRITE_SHEET is None:
            raise IOError(f"Não foi possível carregar '{SPRITE_SHEET_PATH}'.")

def extract_template_from_sprite(tile_name):
    coords_map = TILE_MAP_COORDS_ORIGINAL if tile_name in TILE_MAP_COORDS_ORIGINAL else FACE_MAP_COORDS_ORIGINAL
    if tile_name not in coords_map: return None
    x, y, w, h = coords_map[tile_name]
    return SPRITE_SHEET[y:y+h, x:x+w]

def get_calibrated_template(tile_name):
    if tile_name not in CALIBRATED_TEMPLATES:
        template_path = os.path.join(CALIBRATED_TEMPLATES_DIR, f"{tile_name}.png")
        if os.path.exists(template_path):
            template = cv2.imread(template_path)
        else: # Gera se não existir
            original = extract_template_from_sprite(tile_name)
            if original is None:
                CALIBRATED_TEMPLATES[tile_name] = None
                return None
            
            if global_cell_size > 0:
                if tile_name in FACE_MAP_COORDS_ORIGINAL:
                    _, _, orig_w, orig_h = FACE_MAP_COORDS_ORIGINAL[tile_name]
                    template = cv2.resize(original, (orig_w, orig_h), interpolation=cv2.INTER_AREA)
                else:
                    template = cv2.resize(original, (global_cell_size, global_cell_size), interpolation=cv2.INTER_AREA)
                cv2.imwrite(template_path, template)
            else:
                template = original
        CALIBRATED_TEMPLATES[tile_name] = template
    return CALIBRATED_TEMPLATES[tile_name]

def screenshot_game_window():
    try:
        windows = pyautogui.getWindowsWithTitle(MINESWEEPER_WINDOW_TITLE)
        if not windows: return None, (0, 0, 0, 0)
        window = windows[0]
        region = (window.left, window.top, window.width, window.height)
        screenshot = pyautogui.screenshot(region=region)
        return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR), region
    except Exception:
        return None, (0, 0, 0, 0)

def find_template_matches(image, template, threshold=0.8):
    if template is None or template.size == 0: return []
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)
    matches = []
    for pt in zip(*locations[::-1]):
        matches.append([int(pt[0]), int(pt[1]), template.shape[1], template.shape[0]])
    if not matches: return []
    
    # Non-Maximal Suppression para evitar detecções duplicadas
    indices = cv2.dnn.NMSBoxes(matches, [result[y, x] for x, y, _, _ in matches], threshold, 0.4)
    return [tuple(matches[i]) for i in indices.flatten()] if len(indices) > 0 else []

def click_center(x, y, w, h, window_x=0, window_y=0, button='left'):
    center_x, center_y = window_x + x + w // 2, window_y + y + h // 2
    pyautogui.click(x=center_x, y=center_y, button=button, duration=0) # Duração 0 para clique instantâneo
    time.sleep(CLICK_DELAY_SECONDS)

def is_game_over(image):
    if get_calibrated_template("face_dead") is not None and len(find_template_matches(image, get_calibrated_template("face_dead"), 0.95)) > 0:
        return "lose"
    if get_calibrated_template("face_win") is not None and len(find_template_matches(image, get_calibrated_template("face_win"), 0.95)) > 0:
        return "win"
    return "playing"

def build_board(image):
    global global_tabuleiro_offset_x, global_tabuleiro_offset_y, global_cell_size
    cell_types = {"1": 0.8, "2": 0.8, "3": 0.8, "4": 0.8, "5": 0.8, "6": 0.8, "7": 0.8, "8": 0.8,
                  "closed": 0.85, "empty_zero_revealed": 0.9, "flag": 0.85}
    all_cells = []
    for cell_type, threshold in cell_types.items():
        template = get_calibrated_template(cell_type)
        if template is None: continue
        matches = find_template_matches(image, template, threshold)
        for (x, y, w, h) in matches:
            all_cells.append((x, y, w, h, cell_type))
    
    if not all_cells: return {}
    if global_cell_size == -1:
        closed_cells = [c for c in all_cells if c[4] == "closed"]
        if not closed_cells: return {}
        x, y, w, h, _ = closed_cells[0]
        global_tabuleiro_offset_x, global_tabuleiro_offset_y, global_cell_size = x, y, w
        print(f"Calibração: Offset=({x},{y}), Tamanho Célula={w}x{h}")

    board = {}
    priority = {"1": 9, "2": 9, "3": 9, "4": 9, "5": 9, "6": 9, "7": 9, "8": 9, "flag": 7, "empty_zero_revealed": 5, "closed": 1}
    for x, y, w, h, cell_type in all_cells:
        grid_col = round((x - global_tabuleiro_offset_x) / global_cell_size)
        grid_row = round((y - global_tabuleiro_offset_y) / global_cell_size)
        cell_data = {"type": "empty" if cell_type == "empty_zero_revealed" else cell_type, "x": x, "y": y, "w": w, "h": h}
        if grid_row not in board: board[grid_row] = {}
        if grid_col not in board[grid_row] or priority.get(cell_type, 0) > priority.get(board[grid_row][grid_col]["type"], 0):
            board[grid_row][grid_col] = cell_data
    return board

def get_neighbors(board, row, col):
    neighbors = []
    for r in range(row - 1, row + 2):
        for c in range(col - 1, col + 2):
            if (r == row and c == col) or r not in board or c not in board[r]: continue
            neighbors.append((r, c, board[r][c]))
    return neighbors

def solve_deterministic(board, window_x, window_y):
    for r, row_data in board.items():
        for c, cell in row_data.items():
            if not cell["type"].isdigit(): continue
            number = int(cell["type"])
            neighbors = get_neighbors(board, r, c)
            closed_neighbors = [n for n in neighbors if n[2]["type"] == "closed"]
            flagged_neighbors = [n for n in neighbors if n[2]["type"] == "flag"]
            
            # Regra 1: Marcar minas
            if len(closed_neighbors) > 0 and len(closed_neighbors) == (number - len(flagged_neighbors)):
                for nr, nc, ncell in closed_neighbors:
                    click_center(ncell["x"], ncell["y"], ncell["w"], ncell["h"], window_x, window_y, "right")
                    print(f"🚩 Marcado ({nr}, {nc}) por lógica básica")
                    board[nr][nc]["type"] = "flag"
                return board, True
            
            # Regra 2: Revelar células seguras
            if len(closed_neighbors) > 0 and len(flagged_neighbors) == number:
                for nr, nc, ncell in closed_neighbors:
                    click_center(ncell["x"], ncell["y"], ncell["w"], ncell["h"], window_x, window_y, "left")
                    print(f"✅ Revelado ({nr}, {nc}) por lógica básica")
                return board, True
    return board, False

# --- OTIMIZAÇÃO DE ESTRATÉGIA: Novo Solver Avançado ---
def solve_advanced_deterministic(board, window_x, window_y):
    """Implementa a 'Regra de Subconjunto' para resolver padrões mais complexos."""
    numbered_cells = {}
    for r, row_data in board.items():
        for c, cell in row_data.items():
            if cell["type"].isdigit():
                numbered_cells[(r, c)] = cell
    
    for (r1, c1), cell1 in numbered_cells.items():
        neighbors1_all = get_neighbors(board, r1, c1)
        unknown_neighbors1 = {(r, c) for r, c, n_cell in neighbors1_all if n_cell["type"] == 'closed'}
        if not unknown_neighbors1: continue
        
        mines_to_find1 = int(cell1["type"]) - len([n_cell for _, _, n_cell in neighbors1_all if n_cell["type"] == 'flag'])
        
        # Compara com os vizinhos que também são números
        for r2, c2, _ in neighbors1_all:
            if (r2, c2) not in numbered_cells: continue
            
            cell2 = numbered_cells[(r2, c2)]
            neighbors2_all = get_neighbors(board, r2, c2)
            unknown_neighbors2 = {(r, c) for r, c, n_cell in neighbors2_all if n_cell["type"] == 'closed'}
            if not unknown_neighbors2: continue

            mines_to_find2 = int(cell2["type"]) - len([n_cell for _, _, n_cell in neighbors2_all if n_cell["type"] == 'flag'])
            
            # Lógica do subconjunto
            if unknown_neighbors1.issubset(unknown_neighbors2):
                diff_neighbors = unknown_neighbors2 - unknown_neighbors1
                diff_mines = mines_to_find2 - mines_to_find1
                
                if len(diff_neighbors) > 0:
                    # Se o número de minas na diferença é igual ao tamanho da diferença, todos são minas
                    if diff_mines == len(diff_neighbors):
                        for r_d, c_d in diff_neighbors:
                            ncell = board[r_d][c_d]
                            click_center(ncell["x"], ncell["y"], ncell["w"], ncell["h"], window_x, window_y, "right")
                            print(f"🚩 Marcado ({r_d}, {c_d}) por lógica avançada (Subconjunto)")
                            board[r_d][c_d]["type"] = "flag"
                        return board, True
                    
                    # Se o número de minas na diferença é zero, todos são seguros
                    if diff_mines == 0:
                        for r_d, c_d in diff_neighbors:
                            ncell = board[r_d][c_d]
                            click_center(ncell["x"], ncell["y"], ncell["w"], ncell["h"], window_x, window_y, "left")
                            print(f"✅ Revelado ({r_d}, {c_d}) por lógica avançada (Subconjunto)")
                        return board, True
                        
    return board, False

def make_random_move(board, window_x, window_y):
    closed_cells = [cell for row in board.values() for cell in row.values() if cell["type"] == "closed"]
    if closed_cells:
        cell_to_click = closed_cells[0] # Pega a primeira da lista, um pouco mais previsível que random
        click_center(cell_to_click["x"], cell_to_click["y"], cell_to_click["w"], cell_to_click["h"], window_x, window_y, "left")
        print("🎲 Sem jogada segura, fazendo um chute.")
        return True
    return False

def calibrate_and_prepare():
    global global_cell_size
    load_sprite_sheet()
    if os.path.exists(CALIBRATED_TEMPLATES_DIR):
        import shutil
        shutil.rmtree(CALIBRATED_TEMPLATES_DIR)
    os.makedirs(CALIBRATED_TEMPLATES_DIR, exist_ok=True)
    
    print("Calibrando templates... Certifique-se que a janela do jogo está visível.")
    image, _ = screenshot_game_window()
    if image is None: return False
    
    original_closed = extract_template_from_sprite("closed")
    matches = find_template_matches(image, original_closed, 0.7)
    if not matches:
        print("Não foi possível encontrar células para calibração. Verifique se o jogo está aberto e visível.")
        return False
    
    x, y, w, h = matches[0]
    global_cell_size = w
    
    print(f"Templates calibrados! Tamanho da célula: {global_cell_size}x{global_cell_size}")
    all_templates = list(TILE_MAP_COORDS_ORIGINAL.keys()) + list(FACE_MAP_COORDS_ORIGINAL.keys())
    for name in all_templates: get_calibrated_template(name)
    return True

# --- OTIMIZAÇÃO: Loop Principal Enxuto e com Cronômetro ---
def main():
    if not calibrate_and_prepare():
        print("❌ Falha na calibração inicial. Encerrando.")
        return

    print("🤖 Bot Campo Minado Otimizado iniciando em 3 segundos...")
    time.sleep(3)

    first_move_made = False
    start_time = time.time() # Inicia o cronômetro

    while True:
        try:
            image, window_bbox = screenshot_game_window()
            if image is None:
                print("AVISO: Janela do jogo não encontrada.")
                time.sleep(1)
                continue

            game_status = is_game_over(image)
            if game_status != "playing":
                end_time = time.time()
                duration = end_time - start_time
                if game_status == "win":
                    print(f"\n🎉🎉🎉 JOGO VENCIDO! 🎉🎉🎉")
                else:
                    print(f"\n💀 JOGO PERDIDO! 💀")
                print(f"Tempo de partida: {duration:.2f} segundos.")
                break

            board = build_board(image)
            if not board:
                print("⚠️ Não foi possível construir o tabuleiro, tentando novamente...")
                continue
            
            window_x, window_y, _, _ = window_bbox

            if not first_move_made:
                center_row = sorted(board.keys())[len(board.keys()) // 2]
                center_col = sorted(board[center_row].keys())[len(board[center_row].keys()) // 2]
                cell = board[center_row][center_col]
                click_center(cell["x"], cell["y"], cell["w"], cell["h"], window_x, window_y, "left")
                print(f"🖱️ Primeira jogada em ({center_row}, {center_col})")
                first_move_made = True
                time.sleep(0.1) # Pausa estratégica para o tabuleiro se abrir
                continue
            
            # --- OTIMIZAÇÃO: Executa a cadeia de solvers ---
            # 1. Lógica Básica (mais rápida)
            board, made_move = solve_deterministic(board, window_x, window_y)
            if made_move:
                continue

            # 2. Lógica Avançada (se a básica falhar)
            board, made_move = solve_advanced_deterministic(board, window_x, window_y)
            if made_move:
                continue
            
            # 3. Chute (como último recurso)
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