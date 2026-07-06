$lualatex = 'lualatex -shell-escape %O %S';
$biber = 'biber --input-directory slides %O %B';
$bibtex_use = 2;
$out_dir = 'build';
ensure_path('TEXINPUTS', './problems/style//');
