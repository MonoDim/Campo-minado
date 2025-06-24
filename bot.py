import cv2
import numpy as np
import pyautogui
import time
import os
from itertools import combinations
from collections import defaultdict

# --- Variáveis Globais para offset do tabuleiro ---
global_tabuleiro_offset_x = -1
global_tabuleiro_offset_y = -1
global_cell_size = -1
total_minas = 99  # Ajuste conforme o nível de dificuldade

# --- Funções de Base ---
def screenshot():
    """Captura uma screenshot da tela e a converte para o formato BGR do OpenCV."""
    return cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR)

def localizar(imagem, template_path, precisao=0.97):
    """
    Localiza todas as ocorrências de um template em uma imagem.
    Retorna uma lista de tuplas (x, y, largura, altura) das correspondências encontradas.
    """
    template = cv2.imread(template_path)
    if template is None:
        print(f"Erro: Não foi possível carregar o template em {template_path}.")
        return []

    resultado = cv2.matchTemplate(imagem, template, cv2.TM_CCOEFF_NORMED)
    h, w = template.shape[:2]
    loc = np.where(resultado >= precisao)
    
    return [(pt[0], pt[1], w, h) for pt in zip(*loc[::-1])]

def clicar_centro(x, y, w, h, botao='left'):
    """Clica no centro de uma área especificada."""
    centro_x = x + w // 2
    centro_y = y + h // 2
    pyautogui.moveTo(centro_x, centro_y)
    pyautogui.click(button=botao)

def jogo_acabou(imagem):
    """Verifica se o jogo de Campo Minado terminou."""
    face_dead_found = localizar(imagem, "tiles/face_dead.png", 0.97)
    face_win_found = localizar(imagem, "tiles/face_win.png", 0.97) 
    return len(face_dead_found) > 0 or len(face_win_found) > 0

def primeira_jogada(tabuleiro):
    """Realiza a primeira jogada no centro ou em um canto do tabuleiro."""
    if not tabuleiro:
        return False
        
    max_lin = max(tabuleiro.keys())
    max_col = max(max(tabuleiro[lin].keys()) for lin in tabuleiro)
    
    # Tenta clicar no centro
    centro_lin = max_lin // 2
    centro_col = max_col // 2
    
    if centro_lin in tabuleiro and centro_col in tabuleiro[centro_lin]:
        celula = tabuleiro[centro_lin][centro_col]
        if celula['tipo'] == 'closed':
            clicar_centro(celula['x'], celula['y'], celula['w'], celula['h'])
            print(f"🔷 Primeira jogada no centro: ({centro_lin}, {centro_col})")
            return True
    
    # Se não conseguir no centro, tenta em um canto
    cantos = [(0, 0), (0, max_col), (max_lin, 0), (max_lin, max_col)]
    for lin, col in cantos:
        if lin in tabuleiro and col in tabuleiro[lin]:
            celula = tabuleiro[lin][col]
            if celula['tipo'] == 'closed':
                clicar_centro(celula['x'], celula['y'], celula['w'], celula['h'])
                print(f"🔷 Primeira jogada no canto: ({lin}, {col})")
                return True
    
    return False

def construir_tabuleiro(imagem):
    """
    Escaneia a imagem da tela para identificar e organizar as células do tabuleiro.
    Retorna um dicionário aninhado representando o tabuleiro.
    """
    global global_tabuleiro_offset_x, global_tabuleiro_offset_y, global_cell_size

    tipos = [str(i) for i in range(1, 9)] + ["flag", "closed", "empty", "mine"]
    celulas_encontradas_na_tela = []

    for tipo in tipos:
        path = f"tiles/{tipo}.png"
        encontrados = localizar(imagem, path, 0.95)
        for (x, y, w, h) in encontrados:
            celulas_encontradas_na_tela.append((x, y, w, h, tipo))

    if not celulas_encontradas_na_tela:
        return {}

    # Determina o tamanho da célula se ainda não foi feito
    if global_cell_size == -1:
        closed_tiles = localizar(imagem, "tiles/closed.png", 0.97)
        if closed_tiles:
            global_cell_size = closed_tiles[0][2]
        else:
            global_cell_size = celulas_encontradas_na_tela[0][2]
        print(f"Tamanho da célula detectado: {global_cell_size} pixels.")

    # Se a origem do tabuleiro ainda não foi definida
    if global_tabuleiro_offset_x == -1 or global_tabuleiro_offset_y == -1:
        min_x = min(c[0] for c in celulas_encontradas_na_tela)
        min_y = min(c[1] for c in celulas_encontradas_na_tela)
        global_tabuleiro_offset_x = min_x
        global_tabuleiro_offset_y = min_y
        print(f"Origem do tabuleiro detectada em: ({global_tabuleiro_offset_x}, {global_tabuleiro_offset_y})")

    # Mapeia as células para as coordenadas do tabuleiro
    tabuleiro_grid = {}
    for x, y, w, h, tipo in celulas_encontradas_na_tela:
        rel_x = x - global_tabuleiro_offset_x
        rel_y = y - global_tabuleiro_offset_y
        
        linha = round(rel_y / global_cell_size)
        coluna = round(rel_x / global_cell_size)
        
        tabuleiro_grid.setdefault(linha, {})
        tabuleiro_grid[linha][coluna] = {"tipo": tipo, "x": x, "y": y, "w": w, "h": h}

    return tabuleiro_grid

def pegar_vizinhos(tabuleiro, lin, col):
    """
    Retorna uma lista de vizinhos para uma célula específica no tabuleiro.
    Inclui os 8 vizinhos ao redor (incluindo diagonais).
    """
    vizinhos = []
    for i in range(lin - 1, lin + 2):
        for j in range(col - 1, col + 2):
            if i == lin and j == col:
                continue
            if i in tabuleiro and j in tabuleiro[i]:
                vizinhos.append((i, j, tabuleiro[i][j]))
    return vizinhos

def verificar_consistencia_tabuleiro(tabuleiro):
    """
    Verifica se o estado atual do tabuleiro é logicamente consistente.
    Retorna True se consistente, False caso contrário.
    """
    for lin in tabuleiro:
        for col in tabuleiro[lin]:
            celula = tabuleiro[lin][col]
            if celula['tipo'] in [str(i) for i in range(1, 9)]:
                valor_celula = int(celula['tipo'])
                vizinhos = pegar_vizinhos(tabuleiro, lin, col)
                
                bandeiras_vizinhas = 0
                fechadas_vizinhas = 0
                
                for v_lin, v_col, v_cell in vizinhos:
                    if v_cell['tipo'] == 'flag':
                        bandeiras_vizinhas += 1
                    elif v_cell['tipo'] == 'closed':
                        fechadas_vizinhas += 1
                
                if bandeiras_vizinhas > valor_celula:
                    print(f"🚫 ERRO DE CONSISTÊNCIA: Número {valor_celula} em ({lin}, {col}) tem {bandeiras_vizinhas} bandeiras ao redor.")
                    return False
                
                if bandeiras_vizinhas < valor_celula and fechadas_vizinhas == 0:
                    print(f"🚫 ERRO DE CONSISTÊNCIA: Número {valor_celula} em ({lin}, {col}) tem {bandeiras_vizinhas} bandeiras, mas faltam {valor_celula - bandeiras_vizinhas} minas.")
                    return False
    return True

def is_valid_config(tabuleiro_hipotetico, celulas_numeradas_para_validar):
    """
    Verifica se uma configuração hipotética de minas é consistente
    com os números revelados nas células envolvidas.
    """
    for lin_num, col_num, celula_orig_data in celulas_numeradas_para_validar:
        valor_esperado = int(celula_orig_data['tipo'])
        
        vizinhos_do_numero = pegar_vizinhos(tabuleiro_hipotetico, lin_num, col_num)
        
        minas_encontradas_hipotese = 0
        fechadas_restantes_hipotese = 0

        for v_lin, v_col, v_cell in vizinhos_do_numero:
            if v_cell['tipo'] == 'flag':
                minas_encontradas_hipotese += 1
            elif v_cell['tipo'] == 'closed':
                fechadas_restantes_hipotese += 1

        if minas_encontradas_hipotese > valor_esperado:
            return False
        
        if (minas_encontradas_hipotese + fechadas_restantes_hipotese) < valor_esperado:
            return False
            
    return True

def resolver_por_backtracking(tabuleiro):
    """
    Tenta resolver situações complexas usando backtracking limitado.
    Retorna células seguras para clicar e minas para marcar.
    """
    celulas_fechadas = []
    celulas_numeradas = []
    
    # Coleta todas as células relevantes
    for lin in tabuleiro:
        for col in tabuleiro[lin]:
            celula = tabuleiro[lin][col]
            if celula['tipo'] == 'closed':
                celulas_fechadas.append((lin, col))
            elif celula['tipo'] in [str(i) for i in range(1, 9)]:
                celulas_numeradas.append((lin, col, celula))
    
    if not celulas_fechadas or not celulas_numeradas:
        return [], []
    
    # Limita o número de células fechadas para evitar explosão combinatória
    if len(celulas_fechadas) > 8:
        return [], []
    
    # Calcula minas restantes
    minas_restantes = sum(1 for lin in tabuleiro for col in tabuleiro[lin] if tabuleiro[lin][col]['tipo'] == 'flag')
    
    configs_validas = []
    for k in range(0, min(len(celulas_fechadas), total_minas - minas_restantes) + 1):
        for combo in combinations(celulas_fechadas, k):
            tabuleiro_hipotetico = {r: {c: dict(data) for c, data in row.items()} for r, row in tabuleiro.items()}
            
            # Marca as minas na configuração hipotética
            for lin, col in combo:
                tabuleiro_hipotetico[lin][col]['tipo'] = 'flag'
            
            if is_valid_config(tabuleiro_hipotetico, celulas_numeradas):
                configs_validas.append(combo)
    
    # Encontra células que são sempre minas ou sempre seguras
    sempre_minas = []
    sempre_seguras = []
    
    if configs_validas:
        for celula in celulas_fechadas:
            is_mina = all(celula in config for config in configs_validas)
            is_segura = all(celula not in config for config in configs_validas)
            
            if is_mina:
                sempre_minas.append(celula)
            if is_segura:
                sempre_seguras.append(celula)
    
    return sempre_seguras, sempre_minas

def calcular_probabilidades(tabuleiro):
    """
    Calcula a probabilidade de cada célula fechada ser uma mina.
    Retorna um dicionário com as probabilidades.
    """
    celulas_fechadas = []
    celulas_numeradas = []
    
    for lin in tabuleiro:
        for col in tabuleiro[lin]:
            celula = tabuleiro[lin][col]
            if celula['tipo'] == 'closed':
                celulas_fechadas.append((lin, col))
            elif celula['tipo'] in [str(i) for i in range(1, 9)]:
                celulas_numeradas.append((lin, col, celula))
    
    if not celulas_fechadas:
        return {}
    
    # Inicializa probabilidades
    prob = {celula: 0.0 for celula in celulas_fechadas}
    contagem = {celula: 0 for celula in celulas_fechadas}
    
    # Calcula influência de cada número nas células vizinhas
    for lin, col, celula in celulas_numeradas:
        valor = int(celula['tipo'])
        vizinhos = pegar_vizinhos(tabuleiro, lin, col)
        fechadas_vizinhas = [v for v in vizinhos if v[2]['tipo'] == "closed"]
        bandeiras_vizinhas = [v for v in vizinhos if v[2]['tipo'] == "flag"]
        
        minas_restantes = valor - len(bandeiras_vizinhas)
        if minas_restantes > 0 and fechadas_vizinhas:
            prob_mina = minas_restantes / len(fechadas_vizinhas)
            for v_lin, v_col, _ in fechadas_vizinhas:
                prob[(v_lin, v_col)] += prob_mina
                contagem[(v_lin, v_col)] += 1
    
    # Calcula a média das probabilidades
    for celula in prob:
        if contagem[celula] > 0:
            prob[celula] /= contagem[celula]
        else:
            # Para células sem informações, usa a probabilidade global
            minas_marcadas = sum(1 for lin in tabuleiro for col in tabuleiro[lin] if tabuleiro[lin][col]['tipo'] == 'flag')
            celulas_fechadas_total = len(celulas_fechadas)
            prob[celula] = (total_minas - minas_marcadas) / celulas_fechadas_total if celulas_fechadas_total > 0 else 0.5
    
    return prob

def detectar_padroes(tabuleiro):
    """
    Detecta padrões comuns como 1-1, 1-2, etc. que podem revelar jogadas seguras.
    Retorna células seguras para clicar e minas para marcar.
    """
    seguras = []
    minas = []
    
    for lin in tabuleiro:
        for col in tabuleiro[lin]:
            celula = tabuleiro[lin][col]
            if celula['tipo'] in ['1', '2']:  # Padrões geralmente aparecem com números baixos
                vizinhos = pegar_vizinhos(tabuleiro, lin, col)
                fechadas = [v for v in vizinhos if v[2]['tipo'] == "closed"]
                bandeiras = [v for v in vizinhos if v[2]['tipo'] == "flag"]
                
                # Padrão 1-1
                if celula['tipo'] == '1' and len(bandeiras) == 1 and len(fechadas) == 2:
                    for v_lin, v_col, v_cel in vizinhos:
                        if v_cel['tipo'] == '1' and (v_lin, v_col) != (lin, col):
                            vizinhos_vizinho = pegar_vizinhos(tabuleiro, v_lin, v_col)
                            fechadas_vizinho = [v for v in vizinhos_vizinho if v[2]['tipo'] == "closed"]
                            bandeiras_vizinho = [v for v in vizinhos_vizinho if v[2]['tipo'] == "flag"]
                            
                            if len(bandeiras_vizinho) == 1 and len(fechadas_vizinho) == 2:
                                celulas_comuns = set((v[0], v[1]) for v in fechadas) & set((v[0], v[1]) for v in fechadas_vizinho)
                                if len(celulas_comuns) == 1:
                                    celula_comum = celulas_comuns.pop()
                                    for v in fechadas:
                                        if (v[0], v[1]) != celula_comum:
                                            seguras.append((v[0], v[1]))
                                    for v in fechadas_vizinho:
                                        if (v[0], v[1]) != celula_comum:
                                            seguras.append((v[0], v[1]))
    
    return list(set(seguras)), list(set(minas))

def aplicar_logica(tabuleiro):
    """
    Aplica as regras determinísticas do Campo Minado em ordem de prioridade.
    Retorna o tabuleiro atualizado e um booleano indicando se alguma jogada foi feita.
    """
    jogou_algo_nesta_rodada = False
    
    while True:
        mudanca_feita_nesta_iteracao = False

        # Prepara a lista de células numeradas
        celulas_numeradas = []
        for lin in tabuleiro:
            for col in tabuleiro[lin]:
                celula = tabuleiro[lin][col]
                if celula['tipo'] in [str(i) for i in range(1, 9)]:
                    celulas_numeradas.append((lin, col, celula))

        # --- PRIORIDADE 1: Marcar Minas Determinísticas ---
        for lin, col, celula in celulas_numeradas:
            valor = int(celula['tipo'])
            vizinhos = pegar_vizinhos(tabuleiro, lin, col)
            
            fechadas_vizinhas = [v for v in vizinhos if v[2]['tipo'] == "closed"]
            bandeiras_vizinhas = [v for v in vizinhos if v[2]['tipo'] == "flag"]

            minas_faltantes_para_esta_celula = valor - len(bandeiras_vizinhas)

            if len(fechadas_vizinhas) == minas_faltantes_para_esta_celula and minas_faltantes_para_esta_celula > 0:
                for (i, j, cell) in fechadas_vizinhas:
                    if tabuleiro[i][j]['tipo'] == "closed":
                        clicar_centro(cell['x'], cell['y'], cell['w'], cell['h'], botao='right')
                        tabuleiro[i][j]['tipo'] = 'flag'
                        mudanca_feita_nesta_iteracao = True
                        jogou_algo_nesta_rodada = True
                        print(f"🚩 Bandeira marcada em: {i}, {j} (Mina garantida pelo número {valor})")
            
        if mudanca_feita_nesta_iteracao:
            continue 

        # --- PRIORIDADE 2: Clicar em Células Seguras (Chord) ---
        mudanca_feita_nesta_iteracao = False
        for lin, col, celula in celulas_numeradas:
            valor = int(celula['tipo'])
            vizinhos = pegar_vizinhos(tabuleiro, lin, col)
            
            fechadas_vizinhas = [v for v in vizinhos if v[2]['tipo'] == "closed"]
            bandeiras_vizinhas = [v for v in vizinhos if v[2]['tipo'] == "flag"]

            if len(bandeiras_vizinhas) == valor and len(fechadas_vizinhas) > 0:
                clicar_centro(celula['x'], celula['y'], celula['w'], celula['h'], botao='left')
                mudanca_feita_nesta_iteracao = True
                jogou_algo_nesta_rodada = True
                print(f"✅ Chord aplicado em {lin}, {col}. Revelando vizinhos seguros.")
                return tabuleiro, True 

        if mudanca_feita_nesta_iteracao:
            continue 

        # --- PRIORIDADE 3: Inferência por Backtracking ---
        seguras, minas = resolver_por_backtracking(tabuleiro)
        if seguras or minas:
            for lin, col in minas:
                celula = tabuleiro[lin][col]
                clicar_centro(celula['x'], celula['y'], celula['w'], celula['h'], botao='right')
                tabuleiro[lin][col]['tipo'] = 'flag'
                print(f"🕵️ Backtracking: Mina marcada em ({lin}, {col})")
            
            for lin, col in seguras:
                celula = tabuleiro[lin][col]
                clicar_centro(celula['x'], celula['y'], celula['w'], celula['h'], botao='left')
                tabuleiro[lin][col]['tipo'] = 'empty'
                print(f"🕵️ Backtracking: Célula segura clicada em ({lin}, {col})")
            
            if seguras or minas:
                return tabuleiro, True

        # --- PRIORIDADE 4: Detecção de Padrões ---
        seguras_padroes, minas_padroes = detectar_padroes(tabuleiro)
        if seguras_padroes or minas_padroes:
            for lin, col in minas_padroes:
                celula = tabuleiro[lin][col]
                clicar_centro(celula['x'], celula['y'], celula['w'], celula['h'], botao='right')
                tabuleiro[lin][col]['tipo'] = 'flag'
                print(f"🧩 Padrão: Mina marcada em ({lin}, {col})")
            
            for lin, col in seguras_padroes:
                celula = tabuleiro[lin][col]
                clicar_centro(celula['x'], celula['y'], celula['w'], celula['h'], botao='left')
                tabuleiro[lin][col]['tipo'] = 'empty'
                print(f"🧩 Padrão: Célula segura clicada em ({lin}, {col})")
            
            if seguras_padroes or minas_padroes:
                return tabuleiro, True

        if not mudanca_feita_nesta_iteracao:
            break 

    return tabuleiro, jogou_algo_nesta_rodada

def clicar_aleatorio(tabuleiro):
    """
    Clica em uma célula 'closed' com a menor probabilidade de ser mina.
    Retorna True se uma célula foi clicada, False caso contrário.
    """
    celulas_fechadas = []
    for lin in tabuleiro:
        for col in tabuleiro[lin]:
            celula = tabuleiro[lin][col]
            if celula['tipo'] == "closed":
                celulas_fechadas.append((lin, col, celula))
    
    if not celulas_fechadas:
        return False
    
    # Calcula probabilidades
    probabilidades = calcular_probabilidades(tabuleiro)
    
    if probabilidades:
        # Encontra a célula com menor probabilidade
        celula_mais_segura = min(
            ((lin, col, cel) for lin, col, cel in celulas_fechadas),
            key=lambda x: probabilidades.get((x[0], x[1]), 1.0)  # Default 1.0 se não encontrado
        )
        
        lin, col, celula = celula_mais_segura
        prob = probabilidades.get((lin, col), (total_minas - sum(1 for r in tabuleiro for c in tabuleiro[r] if tabuleiro[r][c]['tipo'] == 'flag')) / len(celulas_fechadas))
        
        clicar_centro(celula['x'], celula['y'], celula['w'], celula['h'])
        print(f"🎲 Chute probabilístico em ({lin}, {col}) com {prob*100:.1f}% de chance de ser mina")
    else:
        # Fallback: clica em qualquer célula fechada
        lin, col, celula = celulas_fechadas[0]
        clicar_centro(celula['x'], celula['y'], celula['w'], celula['h'])
        print(f"🖱️ Clicando aleatoriamente em ({lin}, {col})")
    
    return True

# --- LOOP PRINCIPAL ---
if __name__ == "__main__":
    print("Bot iniciando em 3 segundos...")
    time.sleep(3)
    
    primeira_jogada_feita = False
    
    try:
        dummy_closed_tile = cv2.imread("tiles/closed.png")
        if dummy_closed_tile is None:
            raise FileNotFoundError("Não foi possível carregar 'tiles/closed.png'.")
    except Exception as e:
        print(f"Erro ao carregar template inicial: {e}")
        exit()

    while True:
        tela = screenshot()

        if jogo_acabou(tela):
            if len(localizar(tela, "tiles/face_win.png", 0.97)) > 0:
                print("🏆 O jogo foi vencido!")
            else:
                print("💀 O jogo terminou em derrota.")
            break

        tabuleiro = construir_tabuleiro(tela)
        
        if not tabuleiro:
            print("Nenhuma célula detectada. Verificando novamente...")
            time.sleep(2)
            continue

        # Primeira jogada estratégica
        if not primeira_jogada_feita:
            if primeira_jogada(tabuleiro):
                primeira_jogada_feita = True
                time.sleep(0.5)
                continue

        if not verificar_consistencia_tabuleiro(tabuleiro):
            print("❌ Tabuleiro inconsistente. Parando...")
            break

        # Aplica lógica determinística
        tabuleiro_atualizado, jogou_determinismo = aplicar_logica(tabuleiro.copy())
        
        if not jogou_determinismo:
            # Se nenhuma jogada determinística foi feita, tenta um chute probabilístico
            if not clicar_aleatorio(tabuleiro_atualizado):
                print("✅ Jogo provavelmente vencido ou sem mais movimentos válidos!")
                break
        
        time.sleep(0.3)