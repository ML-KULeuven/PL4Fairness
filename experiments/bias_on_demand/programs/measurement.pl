fair_Y(Features, 1) :: fair_Y(Features, 1).

do_flip(features(0,X,Y,Z), 1, features(1,X,Y,Z)).
do_flip(features(R,0,Y,Z), 2, features(R,1,Y,Z)).
do_flip(features(R,X,0,Z), 3, features(R,X,1,Z)).
do_flip(features(R,X,Y,0), 4, features(R,X,Y,1)).

ppv_a(features(0,0,0,0), N) :: ppv_a(features(0,0,0,0), N).
ppv_not_a(features(0,0,0,0), N) :: ppv_not_a(features(0,0,0,0), N).
npv_a(features(0,0,0,0), N) :: npv_a(features(0,0,0,0), N).
npv_not_a(features(0,0,0,0), N) :: npv_not_a(features(0,0,0,0), N).

f(features(0,0,0,0), N) :: f(features(0,0,0,0), N).
a :- f(features(0,0,0,0), 0).

flip(Features, N) :- a, f(features(0,0,0,0), N), not(ppv_a(features(0,0,0,0), N)).
flip(Features, N) :- not(a), f(features(0,0,0,0), N), not(ppv_not_a(features(0,0,0,0), N)).
flip(Features, N) :- a, not(f(features(0,0,0,0), N)), not(npv_a(features(0,0,0,0), N)).
flip(Features, N) :- not(a), not(f(features(0,0,0,0), N)), not(npv_not_a(features(0,0,0,0), N)).

fair_features(Features, 0, Features).
fair_features(Features, N, FairFeatures) :- >(N,0), flip(Features, N), do_flip(Features, N, FlippedFeatures), is(N_, -(N,1)), fair_features(FlippedFeatures, N_, FairFeatures).
fair_features(Features, N, FairFeatures) :- >(N,0), not(flip(Features, N)), is(N_, -(N,1)), fair_features(Features, N_, FairFeatures).

PY(Features) :- fair_features(Features, 4, FairFeatures), fair_Y(FairFeatures, 1).

?- PY(features(0,0,0,0)).
