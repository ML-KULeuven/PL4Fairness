from pathlib import Path

import re

import torch.nn as nn
from deeplog.formula import DeepLogModuleFactory, get_network_predicate
from deeplog.shape import get_all_symbols
from deeplog.symbol import symbol_to_str
from deeplog.systems.deepproblog import SimpleEngine, compile_to_module, str_to_rules

from src.build_module import BinaryProbability, FeatureFlipper, FeatureSelector, RateModule

PROGRAMS = Path(__file__).parent / "programs"


def compile_program(program, modules):
    """Compile a DeepProbLog program that models the bias together with the classifier modules into a circuit that can compute a biased prediction.
    """
    atom_builders = {(name, 2, "probability"): get_network_predicate(name, 2, "probability", module)
                     for name, module in modules.items()}
    factory = DeepLogModuleFactory(atom_builders=atom_builders)
    result = SimpleEngine().get_query_result(tuple(str_to_rules(program)), factory)
    circuit_module = compile_to_module(result, factory)
    symbols = [symbol_to_str(s) for s in get_all_symbols(circuit_module.get_input_shape())]
    return circuit_module, symbols


class LabelBiasModule(nn.Module):
    """Maps input features ``X`` to probability of biased label being 1 ``P(biased label = 1)``,
     for a batch shape ``(batch,)``.

    ``classifier`` predicts ``P(fair label = 1)`` with shape ``(batch, 1)``.
    ``tpr_not_a``, ``tpr_a``, ``fpr_not_a``, ``fpr_a`` are the rates ``P(Y=1 | Y_fair=1)`` and ``P(Y=1 | Y_fair=0)`` for group a (a=1) and group not_a (a=1),
    where a is the binary sensitive attribute.
    ``sensitive_index``  is the column of ``X`` holding the binary sensitive
    attribute, where 1 means the unprivileged group ``a``.
    """

    def __init__(self, classifier, tpr_not_a, tpr_a, fpr_not_a, fpr_a, sensitive_index=0):
        super().__init__()
        self.classifier = classifier
        self.sensitive_index = sensitive_index

        # One network predicate per annotated functor in the program.
        # BinaryProbability exposes 1 - p as column 0 so fair_Y(Input, 1) can read
        # column 1; FeatureSelector lets f(sensitive, 0) read the sensitive column.
        rates = {"tpr_not_a": tpr_not_a, "tpr_a": tpr_a, "fpr_not_a": fpr_not_a, "fpr_a": fpr_a}
        modules = {"fair_Y": BinaryProbability(classifier), "f": FeatureSelector()}
        modules |= {name: RateModule({1: rate}) for name, rate in rates.items()}
        self.circuit_module, self.input_symbols = compile_program(
            (PROGRAMS / "label_bias.pl").read_text(), modules)

    def get_classifier(self):
        """Return only the classifier that is included in this bias module"""
        return self.classifier

    def forward(self, X):
        a = X[:, self.sensitive_index].unsqueeze(1)
        return self.circuit_module(*[a if s == "sensitive" else X for s in self.input_symbols]).flatten()


class MeasurementBiasModule(nn.Module):
    """Maps features ``X`` to ``P(label = 1)`` under measurement bias, for a batch shape ``(batch,)``.
    The first entry of input features ``X`` is assumed to be the sensitive attribute, where
    1 means the unprivileged group ``a``.
    Potentially biased features are in columns 1..n.

    ``classifier`` predicts ``P(label = 1)`` from a *fair* feature vector, with
    shape ``(batch, 1)``.

    Each rate argument is a sequence with one entry per potentially biased feature column (colums 1..n of ``X``).
    ``ppv_a[i]`` is the positive predictive value for group ``a``,
     i.e. ``P(fair feature = 1 | observed feature = 1)`` for group ``a`` and column ``i + 1`` of X (i+1 because X[0] is reserved for the sensitive attribute),
     and ``npv_a[i]`` is ``P(fair feature = 0 | observed feature = 0)`` for group ``a`` for the same feature column.

    """

    def __init__(self, classifier, ppv_not_a, ppv_a, npv_not_a, npv_a):
        super().__init__()
        rates = {"ppv_not_a": ppv_not_a, "ppv_a": ppv_a, "npv_not_a": npv_not_a, "npv_a": npv_a}
        lengths = {len(rate) for rate in rates.values()}
        if len(lengths) != 1:
            raise ValueError(f"All four rate sequences need one entry per feature, got {lengths}")
        self.n_features = lengths.pop()

        self.classifier = classifier
        self.flipper = FeatureFlipper()


        modules = {"fair_Y": BinaryProbability(classifier), "f": FeatureSelector()}
        # enumerate rates starting with index 1 because biased features are 1..n and the 0-index is for the sensitive attribute.
        modules |= {name: RateModule(dict(enumerate(rate, start=1))) for name, rate in rates.items()}
        self.circuit_module, self.input_symbols = compile_program(
            self._program(self.n_features), modules)

    @staticmethod
    def _program(n):
        """Fill in the parts of ``measurement_bias.pl`` that depend on the feature count:
        one ``do_flip`` fact per column, the recursion depth, and the query term."""
        variables = [f"V{i}" for i in range(1, n + 1)]
        do_flip = []
        for column in range(1, n + 1):
            observed = list(variables)
            observed[column - 1] = "0"
            fair = list(variables)
            fair[column - 1] = "1"
            do_flip.append(f"do_flip(features({','.join(observed)}), {column}, "
                           f"features({','.join(fair)})).")
        return (PROGRAMS / "measurement_bias.pl").read_text().format(
            do_flip="\n".join(do_flip), n=n, query=f"features({','.join(['0'] * n)})")

    def get_classifier(self):
        """Return only the classifier that is included in this bias module"""
        return self.classifier

    def forward(self, X):
        # Create combinations of flipped columns and match them to symbols
        variants = self.flipper(X)
        inputs = []
        for symbol in self.input_symbols:
            match = re.fullmatch(r"features\(([01](?:,[01])*)\)", symbol)
            inputs.append(variants[:, int(match.group(1).replace(",", ""), 2)] if match else X)
        # Feed the flipped combinations into the circuit that combines them into a prediction
        return self.circuit_module(*inputs).flatten()


class HistoricalBiasModule(nn.Module):
    """Maps features ``X`` to ``P(biased label = 1)`` under historical bias, shape ``(batch,)``.

    Historical bias is when the sensitive attribute influenced the features, and
    through them the label. This can be modelled as label bias, measurement bias, or both
    at once.

    ``classifier`` predicts ``P(fair label = 1)`` from a fair feature vector, with
    shape ``(batch, 1)``.


    Pass the relevant rates of how you model the historical bias. Add rates for labels, features, or both.
    - ``label_rates``: a dict with keys ``tpr_not_a``, ``tpr_a``, ``fpr_not_a``,
      ``fpr_a``, each a single rate, as for :class:`LabelBiasModule`.
    - ``measurement_rates``: a dict with keys ``ppv_not_a``, ``ppv_a``,
      ``npv_not_a``, ``npv_a``, each a sequence with one rate per biased feature
      column, as for :class:`MeasurementBiasModule`.

    The first entry of input features ``X`` is assumed to be the sensitive attribute, where
    1 means the unprivileged group ``a``.
    """

    def __init__(self, classifier, label_rates=None, measurement_rates=None):
        super().__init__()
        if label_rates is None and measurement_rates is None:
            raise ValueError("Give label_rates, measurement_rates or both")
        self.classifier = classifier

        module = classifier
        if label_rates is not None:
            _check_keys("label_rates", label_rates, LABEL_RATES)
            module = LabelBiasModule(classifier, **label_rates)
        if measurement_rates is not None:
            _check_keys("measurement_rates", measurement_rates, MEASUREMENT_RATES)
            # MeasurementBiasModule wants a classifier output of shape (batch, 1)
            inner = classifier if label_rates is None else _Unflatten(module)
            module = MeasurementBiasModule(inner, **measurement_rates)
        self.module = module

    def get_classifier(self):
        """Return only the classifier that is included in this bias module"""
        return self.classifier

    def forward(self, X):
        return self.module(X)


LABEL_RATES = ("tpr_not_a", "tpr_a", "fpr_not_a", "fpr_a")
MEASUREMENT_RATES = ("ppv_not_a", "ppv_a", "npv_not_a", "npv_a")


def _check_keys(name, rates, expected):
    if set(rates) != set(expected):
        raise ValueError(f"{name} needs exactly the keys {list(expected)}, got {list(rates)}")


class _Unflatten(nn.Module):
    """Reshapes a ``(batch,)`` output to ``(batch, 1)``."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, X):
        return self.module(X).unsqueeze(1)
