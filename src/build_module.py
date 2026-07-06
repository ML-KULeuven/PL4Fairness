"""Builds the DeepLog circuit module from a ProbLog (.pl) program."""

import re

import torch
import torch.nn as nn
from deeplog.formula import DeepLogModuleFactory, get_network_predicate

from deeplog.shape import get_all_symbols
from deeplog.symbol import symbol_to_str
from deeplog.systems.deepproblog import SimpleEngine, compile_to_module, str_to_rules

from src.utils import get_binary, clamp_hook


class BinaryProbability(nn.Module):
    """Wrap a module emitting a single success probability ``p`` of shape
    ``(batch, 1)`` so it returns ``(batch, 2) = [1 - p, p]``.

    Network predicates select an output *column* by the atom's second argument;
    binary facts use column ``1`` for "true", so the module must expose both
    ``1 - p`` (column 0) and ``p`` (column 1)."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, x):
        p = self.base(x)
        return torch.cat([1.0 - p, p], dim=1)


class FeatureSelector(nn.Module):
    """Passes features through unchanged; ``f(Features, i)`` then reads column
    ``i`` (e.g. column 0 is the sensitive attribute)."""

    def __init__(self, pos_vals=None):
        super().__init__()
        self.pos_vals = [1] if pos_vals is None else pos_vals

    @staticmethod
    def forward(features):
        return features


class FeatureFlipper(nn.Module):
    """Expands a feature row into all 2^l flips of its last ``l`` columns,
    returning ``(batch, 2^l, n_features)`` — one row per flip combination
    (column 0, the sensitive attribute, is never flipped)."""

    @staticmethod
    def forward(x):
        lst = [x]
        l = x.shape[1] - 1
        for i in range(1, 2 ** l):
            flip_mask = torch.tensor([0] + get_binary(i, l), device=x.device)
            lst.append(x * (1 - flip_mask) + (1 - x) * flip_mask)
        return torch.stack(lst).swapaxes(0, 1)


class RateModule(nn.Module):
    """Emits a (clamped) probability vector, broadcast over the batch.

    ``values`` maps an output column index to a fixed probability, or to
    ``None`` for a learnable parameter (initialised at 0.5). The output has
    ``max(index) + 1`` columns so that ``atom(In, K)`` can read column ``K``;
    unspecified columns are 0 and are never indexed by the program. The input is
    ignored apart from its batch size."""

    def __init__(self, values, minimum=0.0, maximum=1.0):
        super().__init__()
        width = max(values) + 1
        learnable = torch.zeros(width, dtype=torch.float32)
        fixed = torch.zeros(width, dtype=torch.float32)
        mask = torch.zeros(width, dtype=torch.bool)  # True where learnable
        for idx, value in values.items():
            if value is None:
                learnable[idx] = 0.5
                mask[idx] = True
            else:
                fixed[idx] = float(value)
        self.param = nn.Parameter(learnable, requires_grad=True)
        self.register_buffer("fixed", fixed)
        self.register_buffer("mask", mask)
        # Keep learnable probabilities in [minimum, maximum] each step.
        self.register_forward_pre_hook(lambda m, _: clamp_hook(m, minimum, maximum))

    def forward(self, x):
        row = torch.where(self.mask, self.param, self.fixed)
        return row.repeat(x.shape[0], 1)  # broadcast over batch


def build_module(unbiased_classifier, cfg, images=False):
    """Compile ``cfg.program_path`` into a module mapping ``(X, a)`` (features /
    image and sensitive attribute) to the fair-label probability predicted via
    the logic program."""
    assert not (cfg.flip_features and images), \
        "Feature flipping is not supported for image data."

    with open(cfg.program_path, 'r') as f:
        code = f.read()

    # One network predicate per annotated functor, keyed (functor, 2, structure).
    atom_builders = {}

    # Fair classifier: outputs P(fair label = 1) in column 1.
    classifier = BinaryProbability(unbiased_classifier)
    atom_builders[('fair_Y', 2, 'probability')] = \
        get_network_predicate('fair_Y', 2, 'probability', classifier)

    # Feature selector: f(In, i) reads feature column i.
    atom_builders[('f', 2, 'probability')] = \
        get_network_predicate('f', 2, 'probability', FeatureSelector())

    # Rate parameters (tpr/fpr/ppv/npv/noise): group config entries by functor;
    # each grounded column index K maps to one rate value.
    rates_by_functor = {}
    for rate, (name, output) in zip(cfg.rates, cfg.names):
        rates_by_functor.setdefault(name, {})[int(output)] = rate
    for name, values in rates_by_functor.items():
        atom_builders[(name, 2, 'probability')] = \
            get_network_predicate(name, 2, 'probability', RateModule(values))

    # Ground the program and compile the shared boolean circuit to probability.
    factory = DeepLogModuleFactory(atom_builders=atom_builders)
    result = SimpleEngine().get_query_result(tuple(str_to_rules(code)), factory)
    circuit_module = compile_to_module(result, factory)

    return _FairModule(circuit_module, cfg)


class _FairModule(nn.Module):
    """Routes the ``(X, a)`` forward inputs to the named tensors the compiled
    circuit expects, then runs the circuit.

    The circuit's inputs are the symbols the program grounded on:
    ``features(0,0,0,0)`` / ``img`` for the classifier, the flattened input
    constants for the rates / selector, and (in flip mode) one
    ``features(b0,b1,b2,b3)`` symbol per flipped feature variant. The sensitive
    tensor ``a`` feeds the ``sensitive`` symbol; the feature/image tensor ``X``
    feeds everything else."""

    def __init__(self, circuit_module, cfg):
        super().__init__()
        self.circuit_module = circuit_module
        self.flip = cfg.flip_features
        self.input_symbols = [symbol_to_str(s) for s in get_all_symbols(circuit_module.get_input_shape())]
        if self.flip:
            self.flipper = FeatureFlipper()

    def forward(self, X, a):
        variants = self.flipper(X) if self.flip else None  # (B, 2^l, n_features)
        inputs = []
        for sym in self.input_symbols:
            m = re.fullmatch(r'features\((\d),(\d),(\d),(\d)\)', sym)
            if self.flip and m:
                inputs.append(variants[:, int(''.join(m.groups()), 2)])
            elif sym == 'sensitive':
                inputs.append(a)
            else:
                inputs.append(X)
        return self.circuit_module(*inputs)
