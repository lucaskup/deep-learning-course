$lualatex = 'lualatex -shell-escape -interaction=nonstopmode -halt-on-error %O %S';
$biber = 'biber --input-directory slides %O %B';
$bibtex_use = 2;
$out_dir = 'build';
ensure_path('TEXINPUTS', './problems/style//');
