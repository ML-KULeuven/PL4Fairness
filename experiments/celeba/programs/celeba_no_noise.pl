fair_Y(Img, 1) :: fair_Y(Img, 1).

tpr_a(sensitive, 1) :: tpr_a(sensitive, 1).
tpr_not_a(sensitive, 1) :: tpr_not_a(sensitive, 1).
fpr_a(sensitive, 1) :: fpr_a(sensitive, 1).
fpr_not_a(sensitive, 1) :: fpr_not_a(sensitive, 1).

f(sensitive, 0) :: f(sensitive, 0).
a :- f(sensitive, 0).

PY(Img, Features) :- fair_Y(Img, 1), not(a), tpr_not_a(sensitive, 1).
PY(Img, Features) :- fair_Y(Img, 1), a, tpr_a(sensitive, 1).
PY(Img, Features) :- not(fair_Y(Img, 1)), not(a), fpr_not_a(sensitive, 1).
PY(Img, Features) :- not(fair_Y(Img, 1)), a, fpr_a(sensitive, 1).

?- PY(img, sensitive).
