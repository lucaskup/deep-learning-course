\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{dlproblemset}[2026/05/20 PUCRS Intro to Deep Learning problem set class]

% Forward unknown options to scrartcl, then preset our defaults.
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{scrartcl}}
\PassOptionsToClass{a4paper,11pt,parskip=half,numbers=noenddot}{scrartcl}
\ProcessOptions\relax
\LoadClass{scrartcl}

% Fonts: XCharter serif + Heros sans + Latin Modern mono, math via newtxmath[xcharter].
\RequirePackage{fontspec}
\setmainfont{XCharter}[
    UprightFont    = *-Roman,
    BoldFont       = *-Bold,
    ItalicFont     = *-Italic,
    BoldItalicFont = *-BoldItalic,
    Numbers        = OldStyle,
]
\setsansfont{TeX Gyre Heros}
\setmonofont{Latin Modern Mono}[Scale=0.95]

\RequirePackage[brazilian]{babel}
\RequirePackage[a4paper,
    left=2.2cm, right=2.2cm,
    top=3cm, bottom=3cm,
    headheight=40pt, headsep=20pt,
    footskip=1.4cm
]{geometry}
\RequirePackage{enumerate}
\RequirePackage{amsmath,amssymb,amsthm}
\RequirePackage[xcharter,bigdelims,vvarbb]{newtxmath}
% newtxmath's operators font requests the fontspec family `XCharter(0)' in OT1;
% without these substitutions math digits and operator names silently fall back
% to Computer Modern (LaTeX Font Warning: Font shape `OT1/XCharter(0)/m/n' undefined).
\DeclareFontFamily{OT1}{XCharter(0)}{}
\DeclareFontShape{OT1}{XCharter(0)}{m}{n}{<->ssub * XCharter-TLF/m/n}{}
\DeclareFontShape{OT1}{XCharter(0)}{m}{it}{<->ssub * XCharter-TLF/m/it}{}
\DeclareFontShape{OT1}{XCharter(0)}{b}{n}{<->ssub * XCharter-TLF/b/n}{}
\DeclareFontShape{OT1}{XCharter(0)}{bx}{n}{<->ssub * XCharter-TLF/b/n}{}
\RequirePackage{graphicx}
\RequirePackage{float}
\RequirePackage{subfig}
\RequirePackage{emoji}
\RequirePackage{xcolor}
\RequirePackage{microtype}
\RequirePackage{scrlayer-scrpage}
\RequirePackage[colorlinks=true,allcolors=blue]{hyperref}

% Section typography matching the approved demo.
\setkomafont{section}{\rmfamily\bfseries\fontsize{13}{15}\selectfont}
\setkomafont{subsection}{\rmfamily\bfseries\fontsize{11}{13}\selectfont}
\setkomafont{subsubsection}{\rmfamily\bfseries\itshape\normalsize}
\setkomafont{disposition}{\rmfamily\bfseries}

% Gray 1pt rules above the running header and below the footer.
\KOMAoptions{headsepline=1pt, footsepline=1pt}
\setkomafont{headsepline}{\color{gray}}
\setkomafont{footsepline}{\color{gray}}
\setkomafont{pagehead}{\normalfont\sffamily\fontsize{8}{10}\selectfont\color{gray}}
\setkomafont{pagefoot}{\normalfont\sffamily\fontsize{8}{10}\selectfont\color{gray}}

\pagestyle{scrheadings}
\clearpairofpagestyles
\ifoot*{Prof. Lucas Silveira Kupssinkü}
\ofoot*{\thepage}

% Header content: title on the left, subtitle on the right. Resolved at \begin{document}
% so \@title / \@subtitle reflect whatever the lista set.
\makeatletter
\providecommand{\@subtitle}{}
\AtBeginDocument{%
    \ihead*{\@title}%
    \ohead*{\@subtitle}%
}
\makeatother

% Default empty subtitle so a lista that forgets \subtitle{} still compiles.
\providecommand{\subtitle}[1]{\gdef\@subtitle{#1}}
\subtitle{}
