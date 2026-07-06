# Listas de exercícios

Listas de exercícios teóricos da disciplina, organizadas por unidade (`01-mlp/` … `06-topicos/`), seguindo a mesma numeração das pastas de slides.

## Convenções

- Cada lista é um arquivo `listaNN_topico.tex`, numerado pela ordem das aulas dentro da unidade.
- Todas usam a classe compartilhada [`style/dlproblemset.cls`](style/dlproblemset.cls), que define fonte, cabeçalho, rodapé e os pacotes básicos. Cada lista declara apenas `\title{}` e `\subtitle{}` e começa direto no conteúdo.
- O texto das questões é em português (pt-BR); nomes de arquivo ficam em inglês/snake_case.

## Compilação

Sempre a partir da **raiz do repositório** (a classe é resolvida via `TEXINPUTS` configurado no `latexmkrc`):

```bash
make problems                                    # todas as listas
latexmk -lualatex problems/01-mlp/lista01_mlp.tex  # uma lista específica
```

Os PDFs são gerados em `build/` e publicados no GitHub Pages quando um release é criado.
