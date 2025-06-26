# Bot Inteligente para Campo Minado

![Minesweeper Logo]([https://raw.githubusercontent.com/sanches-programador/Bot-Campo-Minado/main/tiles/cloneskin.png](https://github.com/MonoDim/Campo-minado/blob/main/tiles/cloneskin.png))

## 📖 Descrição

Este é um bot de alta performance desenvolvido em Python que joga o clássico Campo Minado da Microsoft de forma autônoma. Utilizando visão computacional com OpenCV e controle de mouse com PyAutoGUI, o bot é capaz de analisar o tabuleiro e tomar decisões lógicas para vencer o jogo no menor tempo possível.

O projeto foi otimizado para velocidade e eficiência, incorporando não apenas uma lógica determinística básica, mas também um solver avançado que resolve padrões complexos, minimizando a necessidade de chutes aleatórios.

## ✨ Funcionalidades Principais

-   **Gameplay 100% Autônomo:** O bot controla o mouse para jogar do início ao fim.
-   **Visão Computacional:** Usa OpenCV para identificar os números, células vazias e bandeiras no tabuleiro.
-   **Calibração Automática:** Detecta o tamanho das células do jogo na primeira execução para se adaptar a diferentes resoluções.
-   **Solver de Lógica Dupla:**
    1.  **Solver Básico:** Resolve as jogadas mais óbvias (marcar minas certas e revelar casas seguras).
    2.  **Solver Avançado (Regra de Subconjunto):** Analisa padrões complexos entre casas numeradas adjacentes para deduzir jogadas que a lógica básica não consegue, aumentando drasticamente a taxa de sucesso.
-   **Alta Performance:** O código é otimizado para respostas rápidas, com atrasos mínimos entre as jogadas, visando recordes de tempo.
-   **Cronômetro de Partida:** Mede e exibe o tempo total para resolver cada tabuleiro.

## 🛠️ Tecnologias e Dependências

-   **Python 3.x**
-   **OpenCV:** Para todo o processamento de imagem e visão computacional.
-   **NumPy:** Dependência principal do OpenCV para manipulação de arrays.
-   **PyAutoGUI:** Para automação da interface gráfica e controle do mouse.

## ⚙️ Instalação e Configuração

Siga os passos abaixo para colocar o bot em funcionamento.

**1. Clone o Repositório:**
```bash
git clone [https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git](https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git)
cd SEU_REPOSITORIO
