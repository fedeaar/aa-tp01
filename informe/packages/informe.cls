\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{informe}
%\DeclareOption{handout}{\PassOptionsToClass{\CurrentOption}{amsart}}
%\ProcessOptions\relax
\LoadClass[12pt, oneside,reqno]{amsart}
\RequirePackage[margin=1in]{geometry} 
\RequirePackage{lmodern, enumitem, needspace}
\RequirePackage{graphicx, wrapfig, pgffor, ifthen}
\RequirePackage[normalem]{ulem}
\RequirePackage[font=small]{caption}
\RequirePackage[caption=false]{subfig}
\RequirePackage{hyperref}
\RequirePackage{listings}
\RequirePackage{xcolor}
\hypersetup {
  pdfstartview=FitH,
  colorlinks=true,
  citecolor=black,
  filecolor=black,
  linkcolor=black,
  urlcolor=black
}

\renewcommand{\contentsname}{contenidos}
\renewcommand{\refname}{Referencias}
% \renewcommand\qedsymbol{$\blacksquare$}
\renewcommand{\lstlistingname}{Algoritmo}
\renewcommand{\figurename}{Figura}
\renewcommand\thesubfigure{\roman{subfigure}}
\captionsetup[subfloat]{labelformat=empty}

\lstdefinelanguage{pseudo}{
  morekeywords={
    pre, post, if, while, for, in, out, proc, inout, do, and, not, return, break, raise},
  sensitive=false, 
  morecomment=[l]{//},
  morecomment=[s]{/*}{*/},
  keywordstyle=\color{black}\textbf,
  commentstyle=\color{gray},
  morestring=[b]',
  morestring=[b]",
  inputencoding=utf8,
  extendedchars=true,
  aboveskip=\baselineskip,
  numberstyle=\ttfamily\tiny,
  basicstyle=\ttfamily\footnotesize,
  breakatwhitespace=false,
  breaklines=true,
  captionpos=b,
  keepspaces=true,
  numbers=left,
  xleftmargin=2em,
  numbersep=5pt,
  showspaces=false,
  showstringspaces=false,
  showtabs=false,
  tabsize=2,
  frame=tb,
  literate=
    {λ}{$\lambda$}1
    {α}{$\alpha$}1
    {ε}{$\epsilon$}1
    {Λ}{$\Lambda$}1
    {>=}{$\geq$}2
    {<=}{$\leq$}2
    {<-}{$\leftarrow$}2
    {->}{$\rightarrow$}2
}

\newcommand{\TODO}[1]{
  {\noindent \color{red} [#1]}
}
\DeclareRobustCommand{\bb}[1]{\mathbb{#1}}
\DeclareRobustCommand{\cal}[1]{\mathcal{#1}}
\DeclareRobustCommand{\sc}[1]{\textsc{#1}}
\newcommand{\nil}{\varnothing}
\newcommand{\R}{\bb{R}}
\newcommand{\Q}{\bb{Q}}
\newcommand{\Z}{\bb{Z}}
\newcommand{\N}{\bb{N}}
\newcommand{\ep}{\varepsilon}
\renewcommand{\phi}{\varphi}
\newcommand{\ceil}[1]{\left\lceil#1\right\rceil}
\newcommand{\floor}[1]{\left\lfloor#1\right\rfloor}
\newcommand{\arr}[1]{\left\langle#1\right\rangle}
\newcommand{\paren}[1]{\left(#1\right)}
\newcommand{\brk}[1]{\left[#1\right]}
\newcommand{\abs}[1]{\left|#1\right|}
\newcommand{\curl}[1]{\left\{#1\right\}}
