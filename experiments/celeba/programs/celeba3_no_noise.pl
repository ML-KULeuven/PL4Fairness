fair_Y(Img, 1) :: fair_Y(Img, 1).

tpr_a1(sensitive, 1) :: tpr_a1(sensitive, 1).
tpr_not_a1(sensitive, 1) :: tpr_not_a1(sensitive, 1).
fpr_a1(sensitive, 1) :: fpr_a1(sensitive, 1).
fpr_not_a1(sensitive, 1) :: fpr_not_a1(sensitive, 1).

tpr_a2(sensitive, 1) :: tpr_a2(sensitive, 1).
tpr_not_a2(sensitive, 1) :: tpr_not_a2(sensitive, 1).
fpr_a2(sensitive, 1) :: fpr_a2(sensitive, 1).
fpr_not_a2(sensitive, 1) :: fpr_not_a2(sensitive, 1).

tpr_a3(sensitive, 1) :: tpr_a3(sensitive, 1).
tpr_not_a3(sensitive, 1) :: tpr_not_a3(sensitive, 1).
fpr_a3(sensitive, 1) :: fpr_a3(sensitive, 1).
fpr_not_a3(sensitive, 1) :: fpr_not_a3(sensitive, 1).

f(sensitive, 0) :: f(sensitive, 0).
f(sensitive, 1) :: f(sensitive, 1).
f(sensitive, 2) :: f(sensitive, 2).
a1 :- f(sensitive, 0).
a2 :- f(sensitive, 1).
a3 :- f(sensitive, 2).

PY(Img) :- fair_Y(Img, 1), not(a1), tpr_not_a1(sensitive, 1).
PY(Img) :- fair_Y(Img, 1), a1, tpr_a1(sensitive, 1).
PY(Img) :- not(fair_Y(Img, 1)), not(a1), fpr_not_a1(sensitive, 1).
PY(Img) :- not(fair_Y(Img, 1)), a1, fpr_a1(sensitive, 1).

PY2(Img) :- PY(Img), not(a2), tpr_not_a2(sensitive, 1).
PY2(Img) :- PY(Img), a2, tpr_a2(sensitive, 1).
PY2(Img) :- not(PY(Img)), not(a2), fpr_not_a2(sensitive, 1).
PY2(Img) :- not(PY(Img)), a2, fpr_a2(sensitive, 1).

PY3(Img, Features) :- PY2(Img), not(a3), tpr_not_a3(sensitive, 1).
PY3(Img, Features) :- PY2(Img), a3, tpr_a3(sensitive, 1).
PY3(Img, Features) :- not(PY2(Img)), not(a3), fpr_not_a3(sensitive, 1).
PY3(Img, Features) :- not(PY2(Img)), a3, fpr_a3(sensitive, 1).

?- PY3(img, sensitive).
