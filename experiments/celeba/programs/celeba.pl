fair_Y(Img, 1) :: fair_Y(Img, 1).

tpr_a(sensitive, 1) :: tpr_a(sensitive, 1).
tpr_not_a(sensitive, 1) :: tpr_not_a(sensitive, 1).
fpr_a(sensitive, 1) :: fpr_a(sensitive, 1).
fpr_not_a(sensitive, 1) :: fpr_not_a(sensitive, 1).

noise_a(sensitive, 1) :: noise_a(sensitive, 1).
f(sensitive, 0) :: f(sensitive, 0).
noisy_a(sensitive) :- f(sensitive, 0).

a(sensitive) :- noise_a(sensitive, 1), not(noisy_a(sensitive)).
a(sensitive) :- not(noise_a(sensitive, 1)), noisy_a(sensitive).

PY(Img, Features) :- fair_Y(Img, 1), not(a(sensitive)), tpr_not_a(sensitive, 1).
PY(Img, Features) :- fair_Y(Img, 1), a(sensitive), tpr_a(sensitive, 1).
PY(Img, Features) :- not(fair_Y(Img, 1)), not(a(sensitive)), fpr_not_a(sensitive, 1).
PY(Img, Features) :- not(fair_Y(Img, 1)), a(sensitive), fpr_a(sensitive, 1).

?- PY(img, sensitive).
