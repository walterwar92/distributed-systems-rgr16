# Конфигурация latexmk для курсовой работы по ТАУ.
# Проект использует пакет fontspec (шрифт Times New Roman, кириллица),
# который требует движок XeTeX. Поэтому latexmk всегда собирает PDF
# через xelatex — даже если запущен без флага -xelatex.
$pdf_mode = 5;   # 5 = xelatex (1 = pdflatex, 4 = lualatex)

# Не останавливаться на ошибках в интерактивном режиме.
$xelatex = 'xelatex -interaction=nonstopmode -synctex=1 %O %S';
