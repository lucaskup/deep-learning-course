# Provas

Provas da disciplina, com gabaritos e folhas de fórmulas.

## Convenções

- As provas são organizadas em pastas por semestre (`2026-1/`, `2026-2/`, …).
- Cada prova `<semestre>_<unidades>.tex` tem um par `*_gabarito.tex` com as respostas e justificativas. Provas G2 levam o sufixo `_G2` no nome do arquivo.
- Os nomes de arquivo mantêm o prefixo de semestre (`2026_1_…`) mesmo dentro da pasta do semestre: a publicação no GitHub Pages copia os PDFs de forma achatada, e o prefixo evita colisão de nomes entre semestres.
- Os arquivos `*_formulario.tex` são folhas de fórmulas entregues junto com a prova, organizadas por blocos temáticos. Ficam na raiz de `exams/` por serem compartilhados entre semestres.
- As provas são documentos `article` independentes, sem classe compartilhada.
- Os **gabaritos são compilados no CI, mas excluídos da publicação** no GitHub Pages: apenas provas e formulários vão para o site público.

## Compilação

Sempre a partir da **raiz do repositório**:

```bash
make exams                                             # todas as provas e gabaritos
latexmk -lualatex exams/2026-1/2026_1_unidade1.tex     # um documento específico
```

Os PDFs são gerados em `build/`.
