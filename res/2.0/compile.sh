# Builds 35 different quizes naming them testXX, XX = 01 -> 35

for i in 0{1..9} {10..35}; do
    rubber --pdf --jobname test$i quiz.tex
done
