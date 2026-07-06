# Introdução a Deep Learning

[![CI](https://github.com/lucaskup/deep-learning-course/actions/workflows/latex-ci.yml/badge.svg)](https://github.com/lucaskup/deep-learning-course/actions/workflows/latex-ci.yml)
[![LICENSE](https://img.shields.io/github/license/lucaskup/deep-learning-course)](https://github.com/lucaskup/deep-learning-course/blob/main/LICENSE)
[![commit](https://img.shields.io/github/last-commit/lucaskup/deep-learning-course?color=blue)](https://github.com/lucaskup/deep-learning-course/commits/main)
[![PR](https://img.shields.io/badge/PRs-Welcome-red)](https://github.com/lucaskup/deep-learning-course/pulls)
[![GitHub Repo stars](https://img.shields.io/github/stars/lucaskup/deep-learning-course)](https://github.com/lucaskup/deep-learning-course)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21212167.svg)](https://doi.org/10.5281/zenodo.21212167)

Material completo de uma disciplina de Introdução a Deep Learning: slides, listas de exercícios, provas com gabarito e demonstrações interativas.

**🌐 Acesse o material compilado em [lucaskup.github.io/deep-learning-course](https://lucaskup.github.io/deep-learning-course/)**, com todos os PDFs e demos navegáveis por aula.

## 👀 Sobre a disciplina

A disciplina é pensada para estudantes de final de graduação ou início de pós-graduação, cobrindo desde os fundamentos de redes neurais até arquiteturas modernas: Transformers, LLMs, modelos generativos e tópicos atuais como GNNs, modelos multimodais, fairness e machine unlearning. É ministrada em português (pt-BR) pelo Prof. Lucas S. Kupssinskü na PUCRS.

Se você vai ministrar uma disciplina de Deep Learning, fique à vontade para usar e adaptar este material! 🤓👌

## 📦 O que tem aqui

| Material | Quantidade | Onde |
|---|---|---|
| 🖥️ Aulas em Beamer (LaTeX) | 28 apresentações | [`slides/`](slides/) |
| ✏️ Listas de exercícios | 24 listas | [`problems/`](problems/) |
| 📝 Provas com gabarito | 8 provas + 2 formulários | [`exams/`](exams/) |
| 🎛️ Demos interativas | 26 demonstrações | [`docs/`](docs/) |

As demonstrações interativas são páginas em JavaScript puro (sem dependências externas, rodam direto no navegador) para construir intuição sobre os principais conceitos: perceptron, backpropagation, descida de gradiente, funções de ativação, inicialização de pesos, regularização, convoluções, conexões residuais, RNNs/LSTMs, modelagem de linguagem, self-attention, codificação posicional, tokenização, modelos decoder-only e encoder-only, treinamento de LLMs, GANs, normalizing flows, autoencoders/VAEs, modelos de difusão, GNNs, CLIP, fairness e unlearning. Cada demo vive em `docs/<tópico>/` e está vinculada à aula correspondente na página do curso.

## 🧠 Ementa

O curso está organizado em seis unidades:

### Unidade 1: Fundamentos e Redes Neurais (MLP)
Pastas: [`slides/01-mlp`](slides/01-mlp) · [`problems/01-mlp`](problems/01-mlp)

- Introdução a Aprendizado de Máquina, regressão linear e logística
- Perceptron e o problema do XOR
- Multi‑Layer Perceptron (MLP), backpropagation e grafos computacionais
- Descida de Gradiente e variantes (SGD, Momentum, RMSProp, Adam)
- Funções de custo e de ativação, softmax e entropia cruzada
- Estratégias de inicialização de pesos
- Regularização (L1, L2, Dropout) e mitigação de overfitting

### Unidade 2: Redes Neurais Convolucionais (CNNs)
Pastas: [`slides/02-cnn`](slides/02-cnn) · [`problems/02-cnn`](problems/02-cnn)

- Convoluções, pooling e arquiteturas clássicas (LeNet, AlexNet, VGG)
- Aplicações em visão computacional
- Batch Normalization e conexões residuais (ResNets)

### Unidade 3: Modelos Sequenciais
Pastas: [`slides/03-rnn`](slides/03-rnn) · [`problems/03-rnn`](problems/03-rnn)

- Redes Neurais Recorrentes (RNNs)
- LSTMs (Long Short‑Term Memory) e GRUs
- Modelagem de linguagem com redes recorrentes

### Unidade 4: Atenção, Transformers e LLMs
Pastas: [`slides/04-transformers`](slides/04-transformers) · [`problems/04-transformers`](problems/04-transformers)

- Mecanismos de Self‑Attention e a arquitetura Transformer
- Tokenização (BPE, WordPiece, SentencePiece)
- Codificação posicional (absoluta, relativa, RoPE)
- Modelos decoder‑only (família GPT) e encoder‑only (família BERT)
- Treinamento de LLMs: pré‑treinamento, fine‑tuning, PEFT e alinhamento

### Unidade 5: Modelos Generativos
Pastas: [`slides/05-geradores`](slides/05-geradores) · [`problems/05-geradores`](problems/05-geradores)

- GANs (Generative Adversarial Networks)
- Normalizing Flows
- Autoencoders e Variational Autoencoders (VAEs)
- Modelos de Difusão

### Unidade 6: Tópicos Avançados
Pastas: [`slides/06-topicos`](slides/06-topicos) · [`problems/06-topicos`](problems/06-topicos)

- Graph Neural Networks (GNNs)
- CLIP e modelos multimodais
- Fairness e viés em modelos de Deep Learning
- Machine Unlearning em modelos de linguagem

## 📁 Estrutura do repositório

```text
deep-learning-course/
├── 📂 slides/                 Apresentações em LaTeX/Beamer, uma por aula
│   ├── 01-mlp/ … 06-topicos/  Aulas organizadas por unidade
│   ├── SimplePlusTheme/       Tema Beamer customizado
│   ├── code/                  Exemplos em Python referenciados nos slides
│   ├── img/                   Figuras e diagramas
│   ├── ppt/                   Materiais de origem (PowerPoint, TikZ)
│   ├── utils/                 Preâmbulo e comandos LaTeX compartilhados
│   └── reference.bib          Bibliografia unificada do curso
│
├── 📂 problems/               Listas de exercícios (listaNN_*.tex por unidade)
│   ├── 01-mlp/ … 06-topicos/
│   └── style/                 Classe LaTeX compartilhada (dlproblemset.cls)
│
├── 📂 exams/                  Provas, gabaritos (_gabarito.tex) e formulários
│
├── 📂 docs/                   Site estático (GitHub Pages) com demos interativas
│
├── 📂 scripts/                Utilitários em Python que geram figuras dos slides
│
├── 📂 build/                  PDFs compilados (gerado pelo latexmk)
│
├── Makefile                   Alvos para compilar listas e provas
├── latexmkrc                  Configuração do latexmk (LuaLaTeX + shell-escape)
└── LICENSE                    Creative Commons BY 4.0
```

## 🛠️ Como compilar

Os documentos usam **LuaLaTeX** e **latexmk**, com `-shell-escape` habilitado pelo `latexmkrc` (necessário para o pacote `minted`). Uma instalação TeX completa (por exemplo, `texlive-full` ou MiKTeX) é suficiente. Sempre compile a partir da **raiz do repositório**: os documentos referenciam caminhos relativos à raiz.

```bash
make problems          # compila todas as listas de exercícios
make exams             # compila todas as provas e gabaritos
make clean-problems    # limpa artefatos das listas
make clean-exams       # limpa artefatos das provas
```

Slides são compilados individualmente (a mesma invocação vale para qualquer documento):

```bash
latexmk -lualatex slides/01-mlp/02-perceptron.tex
```

Os PDFs são gerados em `build/`.

As demos interativas não têm etapa de build: basta servir `docs/` com qualquer servidor estático, por exemplo `python3 -m http.server`.

## 🚀 CI e publicação

O workflow de CI ([`latex-ci.yml`](.github/workflows/latex-ci.yml)) compila todos os documentos LaTeX a cada push e pull request que altere material do curso (mudanças apenas em arquivos markdown não disparam compilação). A publicação no GitHub Pages acontece quando um release é criado no GitHub: nesse momento os PDFs e o site com as demos são compilados e publicados. Os gabaritos das provas são compilados no CI, mas excluídos da publicação.

## 🤝 Como contribuir

Contribuições são muito bem‑vindas! Se você encontrar algum erro de digitação, um problema nos exercícios ou tiver alguma sugestão para melhorar o material, sinta‑se à vontade para abrir uma Pull Request.

Para contribuir:

- Faça um Fork deste repositório.
- Crie uma nova Branch (`git checkout -b feature/sua-melhoria`).
- Faça o Commit das suas alterações (`git commit -m 'Adiciona melhoria X'`).
- Faça o Push para a sua Branch (`git push origin feature/sua-melhoria`).
- Abra uma Pull Request.

## 📖 Como citar

Este material está arquivado no Zenodo com o DOI [10.5281/zenodo.21212167](https://doi.org/10.5281/zenodo.21212167), que resolve sempre para a versão mais recente do material. Se você usar ou adaptar o material em uma disciplina, artigo ou outro trabalho, cite:

> Kupssinskü, L. S. (2026). *Introdução a Deep Learning: material de curso*. Zenodo. https://doi.org/10.5281/zenodo.21212167

Em BibTeX:

```bibtex
@misc{kupssinsku2026deeplearning,
  author    = {Kupssinsk{\"u}, Lucas Silveira},
  title     = {Introdu{\c{c}}{\~a}o a Deep Learning: material de curso},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21212167},
  url       = {https://doi.org/10.5281/zenodo.21212167}
}
```

## ⚖️ Licença

Este trabalho está licenciado sob a Creative Commons Attribution 4.0 International License (CC BY 4.0).

Isso significa que você tem a liberdade de:

- **Utilizar**: usar o material para ministrar uma disciplina de Deep Learning.
- **Compartilhar**: copiar e redistribuir o material em qualquer suporte ou formato.
- **Adaptar**: remixar, transformar e criar a partir do material para qualquer fim, mesmo que comercial.

Desde que você atribua o devido crédito, forneça um link para a licença e indique se foram feitas alterações.
