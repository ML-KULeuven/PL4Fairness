fair_Y(Features, 1) :: fair_Y(Features, 1).

tpr_a(features(0,0,0,0), 1) :: tpr_a(features(0,0,0,0), 1).
tpr_not_a(features(0,0,0,0), 1) :: tpr_not_a(features(0,0,0,0), 1).
fpr_a(features(0,0,0,0), 1) :: fpr_a(features(0,0,0,0), 1).
fpr_not_a(features(0,0,0,0), 1) :: fpr_not_a(features(0,0,0,0), 1).

f(features(0,0,0,0), 0) :: f(features(0,0,0,0), 0).
a :- f(features(0,0,0,0), 0).

PY(Features) :- fair_Y(Features, 1), not(a), tpr_not_a(features(0,0,0,0), 1).
PY(Features) :- fair_Y(Features, 1), a, tpr_a(features(0,0,0,0), 1).
PY(Features) :- not(fair_Y(Features, 1)), not(a), fpr_not_a(features(0,0,0,0), 1).
PY(Features) :- not(fair_Y(Features, 1)), a, fpr_a(features(0,0,0,0), 1).

?- PY(features(0,0,0,0)).
