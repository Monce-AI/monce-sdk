import json
import logging
import math
import sys
from random import choice, choices, sample, random
from time import time

try:
    from ._accel import (apply_literal_fast, apply_clause_fast,
                         traverse_chain_fast, get_lookalikes_fast,
                         batch_predict_fast, batch_get_lookalikes_fast,
                         filter_ts_remainder_fast, minimize_clause_fast,
                         filter_indices_by_literal_fast,
                         filter_consequence_fast)
    _HAS_ACCEL = True
except ImportError:
    _HAS_ACCEL = False

from .candle import Candle, compute_candle
from .expansion import ExpansionMixin

################################################################
#                                                              #
#    Algorithme.ai : Snake         Author : Charles Dana       #
#                                                              #
#   v5.5.1 — SAT classifier + auto domain extension (MI)       #
#                                                              #
################################################################

_BANNER = """################################################################
#                                                              #
#    Algorithme.ai : Snake         Author : Charles Dana       #
#                                                              #
#   v5.5.1 — SAT classifier + auto domain extension (MI)       #
#                                                              #
################################################################
"""

# ---------------------------------------------------------------------------
# Oppose profile constants
# ---------------------------------------------------------------------------
_VALID_PROFILES = ("auto", "balanced", "linguistic", "industrial",
                   "cryptographic", "scientific", "categorical", "hef")

# ---------------------------------------------------------------------------
# Helper functions for new literal types (module-level, pure Python)
# ---------------------------------------------------------------------------

def _levenshtein(a, b):
    """String distance: exact DP for short strings, O(n) bag-of-chars for long ones.
    Always returns an int. Preserves ordering so midpoint splits work."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Exact Wagner-Fischer DP up to 256 chars (covers all practical use cases)
    if len(a) <= 256 and len(b) <= 256:
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a):
            curr = [i + 1] + [0] * len(b)
            for j, cb in enumerate(b):
                curr[j + 1] = min(prev[j + 1] + 1, curr[j] + 1,
                                  prev[j] + (0 if ca == cb else 1))
            prev = curr
        return prev[-1]
    # Long strings: O(n) char-frequency distance
    # = chars you'd need to insert + delete to transform a into b
    # Lower bound on true levenshtein, preserves midpoint ordering
    fa, fb = {}, {}
    for c in a:
        fa[c] = fa.get(c, 0) + 1
    for c in b:
        fb[c] = fb.get(c, 0) + 1
    shared = sum(min(fa.get(c, 0), fb.get(c, 0)) for c in set(fa) | set(fb))
    return (len(a) - shared) + (len(b) - shared)


def _jaccard_bigrams(a, b):
    """Jaccard similarity on character bigrams. Returns 0.0-1.0."""
    if len(a) < 2 and len(b) < 2:
        return 1.0 if a == b else 0.0
    sa = {a[i:i+2] for i in range(max(0, len(a)-1))}
    sb = {b[i:i+2] for i in range(max(0, len(b)-1))}
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / len(union)


def _text_score(field, ref):
    """Subsequence match ratio: fraction of ref's chars found in field in order.
    O(len(field)), single pass, continuous [0,1]. Combines T's substring logic
    with N's midpoint thresholding for fuzzy part-number matching."""
    if not ref:
        return 0.0
    j = 0
    n = len(ref)
    for c in field:
        if j < n and c == ref[j]:
            j += 1
    return j / n


def _common_prefix_len(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _common_suffix_len(a, b):
    return _common_prefix_len(a[::-1], b[::-1])


def _entropy(s):
    """Shannon entropy of string character distribution."""
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((f/n) * math.log2(f/n) for f in freq.values())


def _hex_ratio(s):
    """Proportion of hex-valid characters."""
    if not s:
        return 0.0
    hex_chars = set("0123456789abcdefABCDEF")
    return sum(1 for c in s if c in hex_chars) / len(s)


def _repeat_period_score(s):
    """How periodic is the string? 1.0 = perfectly repeating, 0.0 = no pattern."""
    if len(s) > 64:
        s = s[:64]  # cap period search space, not the signal
    if len(s) < 4:
        return 0.0
    best = 0.0
    for period in range(1, len(s) // 2 + 1):
        matches = sum(1 for i in range(period, len(s)) if s[i] == s[i % period])
        score = matches / (len(s) - period)
        best = max(best, score)
    return best


def _count_upper(s):
    return sum(1 for c in s if c.isupper())


def _count_digits(s):
    return sum(1 for c in s if c.isdigit())


def _count_special(s):
    return sum(1 for c in s if not c.isalnum() and not c.isspace())

_snake_instance_counter = 0


class _StringBufferHandler(logging.Handler):
    """Logging handler that accumulates formatted records into an in-memory string buffer."""
    def __init__(self):
        super().__init__()
        self.buffer = ""

    def emit(self, record):
        self.buffer += self.format(record) + "\n"


"""
When working with strings of floating point, handles the mistakes by replacing the value to 0.0 when floating parse error
"""
def floatconversion(txt):
    try:
        result = float(txt)
        return result
    except ValueError:
        return 0.0


def _is_missing(v):
    """True for a missing value from any source: None, float NaN, or a pandas
    NA/NaT sentinel. Used to fill NaN by design — 0.0 for numeric fields, "" for
    text fields — at both training and inference. Zero-dependency: NaN is the
    only float not equal to itself, and pandas NA/NaT reuse that contract."""
    if v is None:
        return True
    try:
        return bool(v != v)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Bucket helpers (pure functions, no self)
# ---------------------------------------------------------------------------

def _unique_vals(targets, indices):
    """Deduplicate target values for a set of indices, handling unhashable types."""
    seen = set()
    result = []
    for i in indices:
        t = targets[i]
        try:
            k = t
            if k not in seen:
                seen.add(k)
                result.append(t)
        except TypeError:
            k = json.dumps(t, sort_keys=True)
            if k not in seen:
                seen.add(k)
                result.append(t)
    return result


def build_condition(matching, population, targets, bucket, oppose_fn, apply_literal_fn, max_retries=50, log_fn=None, header=None):
    """AND of oppose literals that peels ~bucket elements from matching."""
    condition = []
    retries = 0
    t_start = time()
    n_literals = 0
    while len(matching) > 2 * bucket and retries < max_retries:
        target_vals = _unique_vals(targets, matching)
        if len(target_vals) < 2:
            if log_fn:
                log_fn(f"#     [condition] only 1 target left in {len(matching)} samples, stopping")
            break
        t_a = choice(target_vals)
        t_b = choice([t for t in target_vals if t != t_a])
        A = population[choice([i for i in matching if targets[i] == t_a])]
        B = population[choice([i for i in matching if targets[i] == t_b])]
        literal = oppose_fn(A, B)
        if literal is None:
            retries += 1
            continue
        if _HAS_ACCEL and header is not None:
            satisfying = filter_indices_by_literal_fast(matching, population, literal, header)
        else:
            satisfying = [i for i in matching if apply_literal_fn(population[i], literal)]
        if len(satisfying) < bucket:
            retries += 1
            continue
        n_literals += 1
        before = len(matching)
        condition.append(literal)
        matching = satisfying
        retries = 0
        if log_fn:
            log_fn(f"#     [condition] literal #{n_literals}: {before} -> {len(matching)} samples (type={literal[3]}, retries_left={max_retries})")
    elapsed = time() - t_start
    if log_fn:
        log_fn(f"#     [condition] built {len(condition)} literals, {len(matching)} samples remaining ({elapsed:.3f}s)")
    return condition, matching


def build_bucket_chain(population, targets, bucket, oppose_fn, apply_literal_fn, noise=0.25, log_fn=None, header=None):
    """Sequential IF/ELIF/ELSE peeling into buckets."""
    chain = []
    remaining = list(range(len(population)))
    t_chain_start = time()
    branch_idx = 0
    if log_fn:
        log_fn(f"#   [bucket_chain] START building chain: {len(population)} samples, bucket_size={bucket}, noise={noise}")
    while len(remaining) > 2 * bucket:
        t_branch = time()
        if log_fn:
            n_targets_remaining = len(_unique_vals(targets, remaining))
            log_fn(f"#   [bucket_chain] --- BRANCH {branch_idx} --- {len(remaining)} remaining, {n_targets_remaining} unique targets")
        condition, selected = build_condition(
            remaining, population, targets, bucket,
            oppose_fn, apply_literal_fn, log_fn=log_fn, header=header
        )
        if not condition:
            if log_fn:
                log_fn(f"#   [bucket_chain] BRANCH {branch_idx}: no condition found, stopping chain")
            break
        core_set = set(selected)
        rest = [i for i in remaining if i not in core_set]
        full_noise_pool = [i for i in range(len(population)) if i not in core_set]
        members = list(selected)
        noise_added = 0
        if noise > 0 and len(full_noise_pool) > 0:
            noise_count = max(1, int(noise * len(selected)))
            noise_added = min(noise_count, len(full_noise_pool))
            members += sample(full_noise_pool, noise_added)
        origins = ["c"] * len(selected) + ["n"] * noise_added
        chain.append({"condition": condition, "members": members, "origins": origins})
        elapsed_branch = time() - t_branch
        if log_fn:
            log_fn(f"#   [bucket_chain] BRANCH {branch_idx}: IF({len(condition)} literals) -> {len(selected)} core + {noise_added} noise = {len(members)} members ({elapsed_branch:.3f}s)")
        remaining = rest
        branch_idx += 1
    if remaining:
        chain.append({"condition": None, "members": remaining, "origins": ["c"] * len(remaining)})
        if log_fn:
            log_fn(f"#   [bucket_chain] ELSE bucket: {len(remaining)} remaining members")
    elapsed_chain = time() - t_chain_start
    if log_fn:
        sizes = [len(e["members"]) for e in chain]
        log_fn(f"#   [bucket_chain] DONE: {len(chain)} buckets, sizes={sizes}, total={elapsed_chain:.3f}s")
    return chain


def traverse_chain(chain, X, apply_literal_fn):
    """Walk the IF/ELIF/ELSE chain, return the first matching bucket."""
    for entry in chain:
        if entry["condition"] is None:
            return entry
        if all(apply_literal_fn(X, lit) for lit in entry["condition"]):
            return entry
    return chain[-1] if chain else None


"""
Snake() of data will provide insights
"""
class Snake(ExpansionMixin):
    def __init__(self, Knowledge, target_index=0, excluded_features_index=(),
                 n_layers=5, bucket=250, noise=0.25, vocal=False, saved=False,
                 progress_file=None, workers=1, oppose_profile="auto", lookahead=5,
                 datatypes=None, expand="auto"):
        # --- logging setup ---
        global _snake_instance_counter
        _snake_instance_counter += 1
        self._logger = logging.getLogger(f"snake.{_snake_instance_counter}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False
        # Remove any leftover handlers (safety for reused logger names)
        self._logger.handlers.clear()

        # Buffer handler — always attached, captures everything to self.log
        self._buffer_handler = _StringBufferHandler()
        self._buffer_handler.setLevel(logging.DEBUG)
        self._buffer_handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(self._buffer_handler)

        # Console handler — only when vocal
        self._console_handler = None
        v = 1 if vocal is True else (vocal if vocal else 0)
        if v >= 2:
            self._console_handler = logging.StreamHandler(sys.stdout)
            self._console_handler.setLevel(logging.DEBUG)
            self._console_handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(self._console_handler)
        elif v >= 1:
            self._console_handler = logging.StreamHandler(sys.stdout)
            self._console_handler.setLevel(logging.INFO)
            self._console_handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(self._console_handler)

        # Initialize buffer with banner
        self._buffer_handler.buffer = _BANNER
        if vocal:
            print(_BANNER)

        self.population = []
        self.header = []
        self.target = None
        self.targets = []
        self.datatypes = []
        self.layers = []
        self.clauses = []
        self.lookalikes = {}
        self.n_layers = n_layers
        self.bucket = bucket
        self.noise = noise
        self.vocal = vocal
        self.progress_file = progress_file
        self.workers = workers
        self.oppose_profile = oppose_profile
        self.expand = expand              # v5.5.0 domain extension: "auto" | False | {col: family}
        self.expansions = []              # fitted expansion records (TOKENSET, ...)
        self._derived_names = set()       # derived column names (machine marker)
        self._col_stats = {}
        self._feature_mi = {}
        self._feature_bins = {}     # occupied-bin count per feature (for MI bias correction)
        self.lookahead = lookahead
        self._enforced_datatypes = datatypes  # if set, skip type detection
        self._t0 = 0
        self._avg_per_layer = 0
        self._current_layer = 0
        # --- v5.4.8: stripped serialization + parallel batch inference ---
        self._stripped = False            # True only when loaded from a stripped JSON (no population)
        self._parallel_threshold = 64     # batches smaller than this run inline (IPC not worth it)
        self._max_workers = None          # None => os.cpu_count(); cap the inference pool size
        self._dtype_hint = None           # {col: "N"|"T"} read off a DataFrame's dtypes (NaN-proof type detection)

        # Detect input type and dispatch
        if isinstance(Knowledge, str) and Knowledge.endswith(".json"):
            self.from_json(Knowledge)
            return

        if isinstance(Knowledge, str) and Knowledge.endswith(".csv"):
            self._init_from_csv(Knowledge, target_index, list(excluded_features_index), saved)
        else:
            self._init_from_data(Knowledge, target_index)

    # ------------------------------------------------------------------
    # CSV flow (original, preserved)
    # ------------------------------------------------------------------
    def _init_from_csv(self, csv_path, target_index, excluded_features_index, saved):
        self.qprint(f"# Initiated Snake with {self.n_layers} layers and vocal mode {self.vocal} from csv {csv_path}")
        with open(csv_path, "r") as f:
            header = self.make_bloc_from_line(f.readlines()[0])
        with open(csv_path, "r") as f:
            rows = f.readlines()[1:]
        target_column = header[target_index]
        self.target = target_column
        train_columns = [header[i] for i in range(len(header)) if not i in (excluded_features_index + [target_index])]
        header_index = [target_index] + [i for i in range(len(header)) if not i in (excluded_features_index + [target_index])]
        self.header = [target_column] + train_columns
        self.qprint(f"# Analysis train columns {train_columns}")
        self.qprint(f"# Analysis header {self.header}")
        self.datatypes = []
        targets = [self.make_bloc_from_line(row)[target_index] for row in rows]
        self._detect_target_type(targets)
        occurences_vector = self._target_counts()
        self.qprint(f"# Algorithme.ai : Occurence Vector {occurences_vector}")
        for t in range(1, len(self.header)):
            hi = header_index[t]
            dtt = "N"
            values = [self.make_bloc_from_line(row)[hi] for row in rows]
            universe = set("".join(values))
            if [c for c in universe if not c in "+-.0123456789e"] == []:
                dtt = "N"
            else:
                dtt = "T"
            if dtt == "N":
                h = header[hi]
                self.qprint(f"#\t[{h}] numeric field")
            if dtt == "T":
                h = header[hi]
                self.qprint(f"#\t[{h}] text field")
            self.datatypes += [dtt]
        self.qprint(f"# Analysis datatypes {self.datatypes}")
        pp = self.make_population(csv_path, drop=True)
        self.target = self.header[0]
        self.targets = [dp[self.target] for dp in pp]
        self.population = pp
        unique = len(self._unique_targets())
        self.qprint(f"# Population ready: {len(pp)} samples, {unique} unique targets, {len(self.header)-1} features")
        self._detect_expansions()   # v5.5.0: grow derived columns before MI/oppose see them
        self._init_oppose_profile()
        self._train(saved)

    # ------------------------------------------------------------------
    # Universal data flow (list/dict/DataFrame)
    # ------------------------------------------------------------------
    def _init_from_data(self, Knowledge, target_index):
        header, rows, ti = self._normalize_input(Knowledge, target_index)
        self.header = [header[ti]] + [header[i] for i in range(len(header)) if i != ti]
        self.target = self.header[0]
        self.qprint(f"# Initiated Snake with {self.n_layers} layers from in-memory data ({len(rows)} rows)")
        self.qprint(f"# Analysis header {self.header}")

        targets = [row[ti] for row in rows]
        self.datatypes = []
        if self._enforced_datatypes and len(self._enforced_datatypes) > 0:
            # Use enforced target type
            self.datatypes = [self._enforced_datatypes[0]]
            self.qprint(f"# Target type: {self.datatypes[0]} (enforced)")
        else:
            # Check for complex (dict/list) targets before stringifying
            has_complex = any(isinstance(t, (dict, list)) for t in targets)
            if has_complex:
                self._detect_target_type(targets, raw=True)
            else:
                self._detect_target_type([str(t) for t in targets])

        # Detect feature types (or use enforced datatypes)
        header_index = [ti] + [i for i in range(len(header)) if i != ti]
        if self._enforced_datatypes and len(self._enforced_datatypes) >= len(self.header):
            # Use enforced datatypes — skip detection entirely
            for t in range(1, len(self.header)):
                dtt = self._enforced_datatypes[t]
                self.qprint(f"#\t[{self.header[t]}] {'numeric' if dtt == 'N' else 'text'} field (enforced)")
                self.datatypes += [dtt]
            self.qprint(f"# Analysis datatypes {self.datatypes} (enforced)")
        else:
            for t in range(1, len(self.header)):
                hi = header_index[t]
                h = self.header[t]
                if self._dtype_hint and h in self._dtype_hint:
                    # Trust pandas' own dtype — NaN-proof. A numeric column stays
                    # numeric even with missing values; only the string sniffer
                    # was ever fooled by str(nan) == "nan".
                    dtt = self._dtype_hint[h]
                    src = " (from DataFrame dtype)"
                else:
                    values = [str(row[hi]) for row in rows if not _is_missing(row[hi])]
                    universe = set("".join(values))
                    if universe and [c for c in universe if not c in "+-.0123456789e"] == []:
                        dtt = "N"
                    else:
                        dtt = "T"
                    src = ""
                self.qprint(f"#\t[{h}] {'numeric' if dtt == 'N' else 'text'} field{src}")
                self.datatypes += [dtt]
            self.qprint(f"# Analysis datatypes {self.datatypes}")

        # Build population dicts
        pp = []
        hashes = set()
        for row in rows:
            item = {}
            item_hash = ""
            reordered = [row[hi] for hi in header_index]
            for i in range(len(self.header)):
                h = self.header[i]
                dtt = self.datatypes[i]
                val = reordered[i]
                missing = _is_missing(val)
                if dtt == "J":
                    item[h] = val
                elif dtt == "B":
                    sv = str(val)
                    if sv in ("True", "TRUE", "true"):
                        item[h] = 1
                    elif sv in ("False", "FALSE", "false"):
                        item[h] = 0
                    else:
                        item[h] = 0 if missing else int(floatconversion(sv))
                elif dtt in "NI":
                    # Fill NaN by design: missing numeric -> 0.0 (note
                    # floatconversion("nan") would otherwise return nan).
                    if missing:
                        item[h] = 0.0 if dtt == "N" else 0
                    else:
                        item[h] = floatconversion(str(val)) if dtt == "N" else int(floatconversion(str(val)))
                else:
                    # Fill NaN by design: missing text -> "".
                    item[h] = "" if missing else str(val)
                if i > 0:
                    item_hash += str(item[h])
            if item_hash not in hashes:
                hashes.add(item_hash)
                pp.append(item)
            else:
                self.qprint(f"# Algorithme.ai : Dropped conflicting row {item}")

        self.population = pp
        self.target = self.header[0]
        self.targets = [dp[self.target] for dp in pp]
        unique = len(self._unique_targets())
        self.qprint(f"# Population ready: {len(pp)} samples, {unique} unique targets, {len(self.header)-1} features")
        self.qprint(f"# Deduplication: {len(rows)} rows -> {len(pp)} unique ({len(rows) - len(pp)} dropped)")
        self._detect_expansions()   # v5.5.0: grow derived columns before MI/oppose see them
        self._init_oppose_profile()
        self._train(False)

    # ------------------------------------------------------------------
    # Normalization: any Knowledge → (header, rows, target_index)
    # ------------------------------------------------------------------
    def _normalize_input(self, Knowledge, target_index):
        # Duck-typed DataFrame (anything with a callable to_dict)
        if hasattr(Knowledge, 'to_dict') and callable(Knowledge.to_dict):
            # Read types off the frame itself, on a copy, BEFORE stringifying.
            # A pandas numeric column stays float64 even with NaN in it, so a
            # single missing value can no longer flip a numeric column to text
            # (str(nan) == "nan" would otherwise poison the universe check). For
            # a minimal duck-typed frame without .dtypes, the hint is None and we
            # fall back to (NaN-aware) string sniffing.
            frame = Knowledge.copy() if hasattr(Knowledge, 'copy') else Knowledge
            self._dtype_hint = self._dtype_hint_from_frame(frame)
            records = frame.to_dict('records')
            if not records:
                raise ValueError("Empty DataFrame")
            header = list(records[0].keys())
            rows = [list(r.values()) for r in records]
            if isinstance(target_index, str):
                ti = header.index(target_index)
            else:
                ti = target_index
            return header, rows, ti

        if not isinstance(Knowledge, list) or len(Knowledge) == 0:
            raise ValueError("Knowledge must be a non-empty list, CSV path, JSON path, or DataFrame")

        first = Knowledge[0]

        # list[dict]
        if isinstance(first, dict):
            header = list(first.keys())
            rows = [list(d.get(k, "") for k in header) for d in Knowledge]
            if isinstance(target_index, str):
                ti = header.index(target_index)
            else:
                ti = 0
            return header, rows, ti

        # list[tuple|list] — check if uniform or variable length
        if isinstance(first, (tuple, list)):
            max_len = max(len(r) for r in Knowledge)
            # Pad variable-length rows with defaults
            rows = []
            for r in Knowledge:
                padded = list(r) + [""] * (max_len - len(r))
                rows.append(padded)
            header = ["target"] + [f"f{i}" for i in range(1, max_len)]
            ti = 0
            return header, rows, ti

        # list[str|int|float] — self-classing
        header = ["target", "f1"]
        rows = [[v, v] for v in Knowledge]
        ti = 0
        return header, rows, ti

    @staticmethod
    def _dtype_hint_from_frame(frame):
        """Map a DataFrame's columns to Snake datatypes from pandas' OWN dtypes,
        not from stringified values. numeric kind (b/i/u/f/c) -> "N", else "T".
        Duck-typed: uses only frame.dtypes.items() and each dtype's .kind, so
        Snake still imports nothing. Returns {col_name: "N"|"T"} or None."""
        dtypes = getattr(frame, "dtypes", None)
        if dtypes is None or not hasattr(dtypes, "items"):
            return None
        hint = {}
        for col, dt in dtypes.items():
            kind = getattr(dt, "kind", None)
            hint[col] = "N" if kind in ("b", "i", "u", "f", "c") else "T"
        return hint

    # ------------------------------------------------------------------
    # Target type detection (shared by both flows)
    # ------------------------------------------------------------------
    def _detect_target_type(self, targets, raw=False):
        """Detect target column type from string values (or raw values if raw=True). Appends to self.datatypes and sets self.targets."""
        if raw and any(isinstance(t, (dict, list)) for t in targets):
            self.datatypes = ["J"]
            self.targets = list(targets)
            n_unique = len(self._unique_targets())
            self.qprint(f"# Algorithme.ai : Snake Analysis on {self.target} a complex JSON target problem ({n_unique} unique)")
            return
        universe = set("".join(targets))
        if sorted(list(set(targets))) == ["0", "1"]:
            self.datatypes = ["B"]
            self.targets = [int(trg) for trg in targets]
            self.qprint(f"# Algorithme.ai : Snake Analysis on {self.target} a binary problem 0/1")
        elif sorted(list(set(targets))) in [["False", "True"], ["FALSE", "TRUE"]]:
            self.datatypes = ["B"]
            self.targets = [int("T" in trg or "t" in trg) for trg in targets]
            self.qprint(f"# Algorithme.ai : Snake Analysis on {self.target} a binary problem True/False")
        elif [c for c in universe if not c in "0123456789"] == []:
            self.datatypes = ["I"]
            self.targets = [int("0" + trg) for trg in targets]
            unique_targets = sorted(list(set(targets)))
            label = "/".join(unique_targets)
            self.qprint(f"# Algorithme.ai : Snake Analysis on {self.target} a multiclass integers problem {label}")
        elif [c for c in universe if not c in "+-.0123456789e"] == []:
            self.datatypes = ["N"]
            unique_targets = sorted(list(set(targets)))
            label = "/".join(unique_targets)
            self.targets = [floatconversion(trg) for trg in targets]
            self.qprint(f"# Algorithme.ai : Snake Analysis on {self.target} a multiclass floating point problem {label}")
        else:
            unique_targets = sorted(list(set(targets)))
            label = "/".join(unique_targets)
            self.targets = list(targets)
            self.qprint(f"# Algorithme.ai : Snake Analysis on {self.target} a multiclass text field problem {label}")
            self.datatypes = ["T"]

    # ------------------------------------------------------------------
    # Training (shared)
    # ------------------------------------------------------------------
    def _train(self, saved):
        self.layers = []
        self.clauses = []
        self.lookalikes = {str(l): [] for l in range(len(self.population))}

        unique_targets = self._sorted_unique_targets()
        target_counts = self._target_counts()

        self.qprint(f"#")
        self.qprint(f"# ============================================================")
        self.qprint(f"#   TRAINING START")
        self.qprint(f"# ============================================================")
        self.qprint(f"#   Population:    {len(self.population)} samples")
        self.qprint(f"#   Features:      {len(self.header) - 1} ({sum(1 for d in self.datatypes[1:] if d == 'T')} text, {sum(1 for d in self.datatypes[1:] if d == 'N')} numeric)")
        self.qprint(f"#   Target:        {self.target} ({self.datatypes[0]} type)")
        self.qprint(f"#   Classes:       {len(unique_targets)} unique values")
        self.qprint(f"#   Layers:        {self.n_layers}")
        self.qprint(f"#   Bucket size:   {self.bucket}")
        self.qprint(f"#   Noise:         {self.noise}")
        self.qprint(f"#   Profile:       {self.oppose_profile}")
        self.qprint(f"#   Lookahead:     {self.lookahead}")
        self.qprint(f"#   Vocal:         {self.vocal}")
        self.qprint(f"#")
        top_5 = sorted(target_counts, key=lambda x: -x[1])[:5]
        self.qprint(f"#   Top classes:   {', '.join(f'{t}({c})' for t, c in top_5)}")
        min_class = min(c for _, c in target_counts)
        max_class = max(c for _, c in target_counts)
        self.qprint(f"#   Class range:   min={min_class}, max={max_class}, avg={len(self.population)/len(unique_targets):.1f}")
        self.qprint(f"# ============================================================")
        self.qprint(f"#")

        self._t0 = time()

        if self.workers > 1:
            # === PARALLEL LAYER CONSTRUCTION ===
            import multiprocessing
            ctx = multiprocessing.get_context("fork")
            n_workers = min(self.workers, self.n_layers)
            self.qprint(f"# Parallel mode: {n_workers} workers for {self.n_layers} layers")

            base_seed = int(time() * 1000) % (2**31)
            # Lightweight per-job args — large data sent once via initializer
            jobs = [
                (self.bucket, self.noise, base_seed + i, i, self.n_layers)
                for i in range(self.n_layers)
            ]

            with ctx.Pool(n_workers, initializer=_init_worker,
                          initargs=(self.population, self.targets,
                                    self.header, self.datatypes,
                                    self.oppose_profile, self._col_stats,
                                    self._feature_mi, self.lookahead)) as pool:
                for i, layer in enumerate(pool.imap_unordered(_build_layer_worker, jobs)):
                    self.layers.append(layer)
                    self._verify_layer(layer, len(self.layers))
                    elapsed = time() - self._t0
                    layers_done = len(self.layers)
                    layers_left = self.n_layers - layers_done
                    avg = elapsed / layers_done
                    eta = avg * layers_left / n_workers if layers_left > 0 else 0

                    n_buckets = len(layer)
                    n_clauses = sum(len(entry["clauses"]) for entry in layer)
                    self.qprint(f"# <<< LAYER {layers_done}/{self.n_layers} DONE (parallel) — {n_buckets} buckets, {n_clauses} clauses, elapsed={elapsed:.2f}s, ETA={eta:.2f}s")
                    self._write_progress(layers_done, eta, elapsed)

        else:
            # === SEQUENTIAL (original) ===
            self._avg_per_layer = 0
            self._current_layer = 0
            for i in range(self.n_layers):
                self._current_layer = i
                t_layer_start = time()
                self.qprint(f"#")
                self.qprint(f"# >>> LAYER {i+1}/{self.n_layers} — starting construction...")
                self.construct_layer()
                self._verify_layer(self.layers[-1], i + 1)
                t_layer_end = time()
                layer_time = t_layer_end - t_layer_start
                elapsed_total = t_layer_end - self._t0
                layers_done = i + 1
                layers_left = self.n_layers - layers_done
                self._avg_per_layer = elapsed_total / layers_done
                eta = self._avg_per_layer * layers_left

                # Count total clauses and buckets in this layer
                layer = self.layers[-1]
                n_buckets = len(layer)
                n_clauses = sum(len(entry["clauses"]) for entry in layer)

                self.qprint(f"# <<< LAYER {i+1}/{self.n_layers} DONE in {layer_time:.2f}s — {n_buckets} buckets, {n_clauses} clauses")
                self.qprint(f"#     elapsed={elapsed_total:.2f}s, avg/layer={self._avg_per_layer:.2f}s, ETA={eta:.2f}s ({layers_left} layers left)")

                self._write_progress(layers_done, eta, elapsed_total)

        total_time = time() - self._t0
        total_clauses = sum(len(entry["clauses"]) for layer in self.layers for entry in layer)
        total_buckets = sum(len(layer) for layer in self.layers)
        self.qprint(f"#")
        self.qprint(f"# ============================================================")
        self.qprint(f"#   TRAINING COMPLETE")
        self.qprint(f"# ============================================================")
        self.qprint(f"#   Total time:    {total_time:.2f}s")
        self.qprint(f"#   Total layers:  {self.n_layers}")
        self.qprint(f"#   Total buckets: {total_buckets}")
        self.qprint(f"#   Total clauses: {total_clauses}")
        self.qprint(f"#   Avg clauses/layer: {total_clauses/self.n_layers:.1f}")
        self.qprint(f"# ============================================================")

        if saved:
            self.to_json()

    def _verify_layer(self, layer, layer_idx):
        """Clause contract verification — runs in main process after layer assembly.
        For each bucket: evaluate each clause ONCE per member, then check
        that no member gets a wrong-class lookalike."""
        for b_idx, bucket in enumerate(layer):
            members = bucket["members"]
            clauses = bucket["clauses"]
            lookalikes_map = bucket["lookalikes"]
            n = len(members)
            local_targets = [self.targets[members[i]] for i in range(n)]

            # Evaluate all clauses on all members ONCE: clause_results[mi] = set of negated clause indices
            member_negated = []
            for mi in range(n):
                member = self.population[members[mi]]
                negated = set()
                for ci, clause in enumerate(clauses):
                    if not self.apply_clause(member, clause):
                        negated.add(ci)
                member_negated.append(negated)

            # Check: no member gets a wrong-class lookalike
            for mi in range(n):
                negated = member_negated[mi]
                member_target = local_targets[mi]
                for li_str in lookalikes_map:
                    li = int(li_str)
                    la_target = local_targets[li]
                    if la_target == member_target:
                        continue
                    for condition in lookalikes_map[li_str]:
                        if all(ci in negated for ci in condition):
                            import json as _j
                            _dump = {
                                "error": "wrong-class lookalike",
                                "layer": layer_idx, "bucket": b_idx,
                                "member_local_idx": mi,
                                "member_global_idx": members[mi],
                                "member_target": str(member_target),
                                "lookalike_local_idx": li,
                                "lookalike_global_idx": members[li],
                                "lookalike_target": str(la_target),
                                "condition_clause_indices": condition,
                                "failing_clauses": [{
                                    "clause_idx": ci,
                                    "clause": str(clauses[ci]),
                                    "per_literal": [
                                        {"literal": str(lit), "eval": self.apply_literal(self.population[members[mi]], lit)}
                                        for lit in clauses[ci]
                                    ]
                                } for ci in condition if ci in negated],
                            }
                            with open("/tmp/snake_fatal.json", "w") as _ef:
                                _ef.write(_j.dumps(_dump, indent=2))
                            exit(f"FATAL: layer {layer_idx} bucket {b_idx} — member[{mi}] class {member_target} "
                                 f"gets lookalike[{li}] class {la_target} — /tmp/snake_fatal.json")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def qprint(self, txt, level=1):
        if level >= 2:
            self._logger.debug(str(txt))
        else:
            self._logger.info(str(txt))

    def _write_progress(self, layers_done, eta_seconds, elapsed_seconds,
                        bucket=None, n_buckets=None, eta_bucket_seconds=None):
        """Write training progress to progress_file (if set)."""
        if not self.progress_file:
            return
        try:
            data = {
                "layer": layers_done,
                "n_layers": self.n_layers,
                "elapsed_seconds": round(elapsed_seconds, 1),
                "eta_seconds": round(eta_seconds, 1),
            }
            if bucket is not None:
                data["bucket"] = bucket
                data["n_buckets"] = n_buckets
                data["eta_bucket_seconds"] = round(eta_bucket_seconds, 1)
            with open(self.progress_file, "w") as _pf:
                json.dump(data, _pf)
        except Exception:
            pass

    @property
    def log(self):
        return self._buffer_handler.buffer

    @log.setter
    def log(self, value):
        self._buffer_handler.buffer = value

    def __repr__(self):
        n = len(self.population) if isinstance(self.population, list) else 0
        return f"Snake(target={self.target!r}, population={n}, layers={len(self.layers)})"

    # ------------------------------------------------------------------
    # Target-key helpers (support unhashable dict/list targets)
    # ------------------------------------------------------------------

    def _target_key(self, t):
        """Return a hashable key for a target value. Simple types pass through; dicts/lists get JSON-serialized."""
        if isinstance(t, (dict, list)):
            return json.dumps(t, sort_keys=True)
        return t

    def _unique_targets(self):
        """Return deduplicated list of targets preserving order."""
        seen = set()
        result = []
        for t in self.targets:
            k = self._target_key(t)
            if k not in seen:
                seen.add(k)
                result.append(t)
        return result

    def _target_counts(self):
        """Return list of (target, count) tuples using _target_key for hashing."""
        counts_by_key = {}
        first_val = {}
        for t in self.targets:
            k = self._target_key(t)
            counts_by_key[k] = counts_by_key.get(k, 0) + 1
            if k not in first_val:
                first_val[k] = t
        return [(first_val[k], c) for k, c in counts_by_key.items()]

    def _sorted_unique_targets(self):
        """Return sorted unique targets. Falls back to _target_key for non-orderable types."""
        unique = self._unique_targets()
        try:
            return sorted(unique)
        except TypeError:
            return sorted(unique, key=lambda t: self._target_key(t))

    """
    Will parse a .csv line properly,
    returning an array of string, handling triple quotes elegantly
    """
    def make_bloc_from_line(self, line):
        line = line.replace('\n', '')
        if '"' in line:
            quoted = False
            bloc = []
            txt = ''
            for c in line:
                if c == '"':
                    quoted = not quoted
                else:
                    if c == ',' and not quoted:
                        bloc += [txt]
                        txt = ''
                    else:
                        txt += c
            bloc += [txt]
            return bloc
        return line.split(',')

    """
    Will effectively parse any .csv properly formated by pandas
    """
    def read_csv(self, fname):
        if not '.csv' in fname:
            self.qprint("Algorithme.ai: Please input a .csv file")
            return 0, 0
        with open(fname, "r") as f:
            lines = f.readlines()
        header = self.make_bloc_from_line(lines[0])
        data = [self.make_bloc_from_line(lines[t]) for t in range(1, len(lines))]
        return header, data

    """
    Makes the population available to the user
    """
    def make_population(self, fname, drop=False):
        POPULATION = []
        data_header, data = self.read_csv(fname)
        mapping_table = {h : -1 for h in self.header}
        for h in mapping_table:
            if h in data_header:
                mapping_table[h] = min((t for t in range(len(data_header)) if data_header[t] == h))
        hashes = set()
        for row in data:
            item_hash = ""
            item = {}
            for i in range(len(self.header)):
                h = self.header[i]
                dtt = self.datatypes[i]
                if mapping_table[h] == -1:
                    if dtt in "NIB":
                        item[h] = 0
                    if dtt == "T":
                        item[h] = ""
                else:
                    raw_val = row[mapping_table[h]]
                    if dtt == "B":
                        if raw_val in ("True", "TRUE", "true"):
                            item[h] = 1
                        elif raw_val in ("False", "FALSE", "false"):
                            item[h] = 0
                        else:
                            item[h] = int(floatconversion(raw_val))
                    elif dtt == "I":
                        item[h] = int(raw_val)
                    elif dtt == "N":
                        item[h] = floatconversion(raw_val)
                    elif dtt == "T":
                        item[h] = str(raw_val)
                if i > 0:
                    item_hash += str(item[h])
            if drop and item_hash in hashes:
                self.qprint(f"# Algorithme.ai : Dropped conflicting row {item}")
            if drop and not item_hash in hashes:
                hashes.add(item_hash)
                POPULATION += [item]
            if not drop:
                POPULATION += [item]
        return POPULATION

    # ------------------------------------------------------------------
    # Core SAT methods (unchanged)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Oppose profiles
    # ------------------------------------------------------------------

    def _init_oppose_profile(self):
        """Resolve oppose_profile and set self._active_oppose."""
        if self.oppose_profile not in _VALID_PROFILES:
            self.oppose_profile = "auto"
        if self.oppose_profile == "auto":
            self.oppose_profile = self._detect_profile()
        self.qprint(f"# Oppose profile: {self.oppose_profile}")
        self._active_oppose = getattr(self, f"_oppose_{self.oppose_profile}")
        if self.oppose_profile == "scientific":
            self._precompute_col_stats()
        self._precompute_feature_mi()
        # v5.5.0: prune low-signal derived columns so expansion never floods the
        # oppose() candidate pool. Gate is relative to original-feature MI; runs
        # after MI is computed over the full (raw + derived) set.
        if getattr(self, "expansions", None):
            self._apply_mi_gate()

    def _detect_profile(self):
        """Scan population and datatypes to pick the best profile."""
        text_cols = [i for i in range(1, len(self.datatypes)) if self.datatypes[i] == "T"]
        num_cols = [i for i in range(1, len(self.datatypes)) if self.datatypes[i] == "N"]

        if not text_cols:
            return "scientific"

        if not num_cols:
            # Analyze text characteristics
            sample_size = min(200, len(self.population))
            rows = sample(self.population, sample_size) if len(self.population) > sample_size else self.population
            total_len = 0
            lengths = []
            digit_count = 0
            upper_count = 0
            special_count = 0
            total_chars = 0
            split_count = 0
            for row in rows:
                for ci in text_cols:
                    h = self.header[ci]
                    v = str(row.get(h, ""))
                    vlen = len(v)
                    total_len += vlen
                    lengths.append(vlen)
                    total_chars += max(vlen, 1)
                    digit_count += _count_digits(v)
                    upper_count += _count_upper(v)
                    special_count += _count_special(v)
                    split_count += len(v.split(",")) + len(v.split(".")) - 2

            n_vals = max(len(lengths), 1)
            avg_len = total_len / n_vals
            len_mean = avg_len
            len_var = sum((l - len_mean) ** 2 for l in lengths) / n_vals if n_vals > 1 else 0
            digit_ratio = digit_count / total_chars
            upper_ratio = upper_count / total_chars
            special_ratio = special_count / total_chars
            avg_splits = split_count / n_vals

            if avg_len > 30 and len_var > 100:
                return "linguistic"
            if avg_len < 15 and digit_ratio > 0.3:
                return "industrial"
            if special_ratio > 0.2 or upper_ratio > 0.6:
                return "cryptographic"
            if avg_splits > 2:
                return "categorical"
            return "balanced"

        # Mixed text + numeric
        return "balanced"

    def _precompute_col_stats(self):
        """Precompute per-column statistics for scientific profile."""
        self._col_stats = {}
        for i in range(1, len(self.datatypes)):
            if self.datatypes[i] == "N":
                h = self.header[i]
                vals = [row[h] for row in self.population if isinstance(row.get(h), (int, float)) and row[h] == row[h]]
                if vals:
                    mu = sum(vals) / len(vals)
                    std = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5
                    sorted_vals = sorted(vals)
                    self._col_stats[i] = {"mean": mu, "std": max(std, 1e-10), "median": sorted_vals[len(sorted_vals) // 2]}

    def _precompute_feature_mi(self):
        """Compute mutual information MI(feature; target) for each feature. O(n*m)."""
        self._feature_mi = {}
        n = len(self.population)
        if n < 2:
            return
        # Target distribution
        target_counts = {}
        for t in self.targets:
            k = self._target_key(t)
            target_counts[k] = target_counts.get(k, 0) + 1
        for col in range(1, len(self.header)):
            h = self.header[col]
            if self.datatypes[col] == "N":
                vals = []
                for row in self.population:
                    v = row.get(h)
                    if isinstance(v, (int, float)) and v == v:
                        vals.append(v)
                    else:
                        vals.append(0.0)
                sorted_unique = sorted(set(vals))
                n_bins = min(20, len(sorted_unique))
                if n_bins < 2:
                    self._feature_mi[col] = 0.0
                    continue
                if len(sorted_unique) <= 20 and n / len(sorted_unique) >= 2.0:
                    # Low-cardinality numeric WITH recurrence (incl. one-hot 0/1
                    # expansion columns): bin by EXACT value. Quantile boundaries put
                    # the cut at the max unique value, so `v <= bnd` collapses a
                    # 2-value column into a single bin and reports MI=0 — exactly
                    # wrong for a perfect discriminator like mod2==0. The recurrence
                    # guard (>= 2 rows per distinct value) is essential: with all-
                    # unique values exact bins give every row its own bin and MI
                    # saturates to H(target) for signal AND noise alike. Non-
                    # recurrent columns fall through to quantile binning, which
                    # regularizes by merging neighbors.
                    bins = vals
                else:
                    boundaries = []
                    for b in range(1, n_bins):
                        idx = min(int(b * len(sorted_unique) / n_bins), len(sorted_unique) - 1)
                        boundaries.append(sorted_unique[idx])
                    bins = []
                    for v in vals:
                        assigned = len(boundaries)
                        for bi, bnd in enumerate(boundaries):
                            if v <= bnd:
                                assigned = bi
                                break
                        bins.append(assigned)
            else:
                raw_vals = [str(row.get(h, "")) for row in self.population]
                val_counts = {}
                for v in raw_vals:
                    val_counts[v] = val_counts.get(v, 0) + 1
                if len(val_counts) > 200:
                    top = sorted(val_counts, key=lambda x: -val_counts[x])[:199]
                    keep = set(top)
                    bins = [v if v in keep else "__other__" for v in raw_vals]
                else:
                    bins = raw_vals
            # Joint histogram → MI
            joint = {}
            feat_counts = {}
            for i_row in range(n):
                b = bins[i_row]
                tk = self._target_key(self.targets[i_row])
                key = (b, tk)
                joint[key] = joint.get(key, 0) + 1
                feat_counts[b] = feat_counts.get(b, 0) + 1
            mi = 0.0
            for (b, tk), count in joint.items():
                p_joint = count / n
                p_feat = feat_counts[b] / n
                p_target = target_counts[tk] / n
                if p_joint > 0 and p_feat > 0 and p_target > 0:
                    mi += p_joint * math.log2(p_joint / (p_feat * p_target))
            self._feature_mi[col] = max(mi, 0.0)
            self._feature_bins[col] = len(feat_counts)
        if self._feature_mi:
            sorted_mi = sorted(self._feature_mi.items(), key=lambda x: -x[1])
            top5 = [(self.header[col], f"{mi:.4f}") for col, mi in sorted_mi[:5]]
            self.qprint(f"# Feature MI (top 5): {top5}")

    def _weighted_feature_choice(self, candidates):
        """Pick a feature index from candidates, weighted by MI. Falls back to uniform."""
        if not self._feature_mi or len(candidates) <= 1:
            return choice(candidates)
        weights = [self._feature_mi.get(i, 0.0) + 1e-4 for i in candidates]
        return choices(candidates, weights=weights, k=1)[0]

    def _get_differing_candidates(self, T, F):
        """Return list of feature indices where T and F differ (excluding NaN numerics)."""
        candidates = [i for i in range(1, len(self.header)) if T[self.header[i]] != F[self.header[i]]
                       and not (self.datatypes[i] == "N" and (T[self.header[i]] != T[self.header[i]] or F[self.header[i]] != F[self.header[i]]))]
        return candidates

    # --- Text literal generators (shared by profiles) ---

    def _gen_text_substring(self, index, T, F):
        """Generate a substring (T) literal. Includes FA/TA single-char matching."""
        h = self.header[index]
        tv, fv = str(T[h]), str(F[h])
        possibilities = []
        # FA/TA: single chars unique to one string (most discriminating)
        chars_only_in_t = [c for c in set(tv) if c not in fv]
        chars_only_in_f = [c for c in set(fv) if c not in tv]
        possibilities += [[index, c, False, "T"] for c in chars_only_in_t]
        possibilities += [[index, c, True, "T"] for c in chars_only_in_f]
        # Separator-based tokens
        pros = set()
        cons = set()
        for sep in [" ", "/", ":", "-"]:
            for label in tv.split(sep):
                pros.add(label.split("'")[0].split('"')[0])
            for label in fv.split(sep):
                cons.add(label.split("'")[0].split('"')[0])
        clean_pros = [label for label in pros if len(label) > 1 and len(label) < max(2, len(tv)) and label not in fv]
        clean_cons = [label for label in cons if len(label) > 1 and len(label) < max(2, len(fv)) and label not in tv]
        possibilities += [[index, label, False, "T"] for label in clean_pros]
        possibilities += [[index, label, True, "T"] for label in clean_cons]
        if possibilities:
            return choice(possibilities)
        if tv != fv and tv not in fv:
            return [index, tv, False, "T"]
        if tv != fv and fv not in tv:
            return [index, fv, True, "T"]
        return None

    def _gen_text_structural(self, index, T, F):
        """Generate TN or TLN literal. Returns literal or None."""
        h = self.header[index]
        tv, fv = str(T[h]), str(F[h])
        possible = []
        if len(fv) != len(tv):
            possible.append("TN")
        if len(set(fv)) != len(set(tv)):
            possible.append("TLN")
        if not possible:
            return None
        todo = choice(possible)
        if todo == "TN":
            return [index, (len(fv) + len(tv)) / 2, len(tv) > len(fv), "TN"]
        return [index, (len(set(fv)) + len(set(tv))) / 2, len(set(tv)) > len(set(fv)), "TLN"]

    def _gen_text_splits(self, index, T, F):
        """Generate TWS, TPS, or TSS literal. Returns literal or None."""
        h = self.header[index]
        tv, fv = str(T[h]), str(F[h])
        possible = []
        if len(fv.split(" ")) != len(tv.split(" ")):
            possible.append("TWS")
        if len(fv.split(",")) != len(tv.split(",")):
            possible.append("TPS")
        if len(fv.split(".")) != len(tv.split(".")):
            possible.append("TSS")
        if not possible:
            return None
        todo = choice(possible)
        if todo == "TWS":
            return [index, (len(fv.split(" ")) + len(tv.split(" "))) / 2, len(tv.split(" ")) > len(fv.split(" ")), "TWS"]
        if todo == "TPS":
            return [index, (len(fv.split(",")) + len(tv.split(","))) / 2, len(tv.split(",")) > len(fv.split(",")), "TPS"]
        return [index, (len(fv.split(".")) + len(tv.split("."))) / 2, len(tv.split(".")) > len(fv.split(".")), "TSS"]

    def _gen_text_distance(self, index, T, F):
        """Generate LEV or JAC literal. Returns literal or None.
        LEV uses O(n) bag-of-chars on long strings — no truncation."""
        h = self.header[index]
        tv, fv = str(T[h]), str(F[h])
        if tv == fv:
            return None
        tag = choice(["LEV", "JAC"])
        if tag == "LEV":
            d = _levenshtein(tv, fv)
            if d == 0:
                return None
            threshold = d / 2
            return [index, [tv, threshold], True, "LEV"]
        else:
            j = _jaccard_bigrams(tv, fv)
            threshold = (j + 1.0) / 2
            return [index, [tv, threshold], True, "JAC"]

    def _gen_text_similarity(self, index, T, F):
        """Generate SIM literal — subsequence score with N-style midpoint.
        T's substring generation + N's continuous thresholding.
        Ref is picked from T or F, score = subsequence match ratio."""
        h = self.header[index]
        tv, fv = str(T[h]), str(F[h])
        if tv == fv:
            return None
        ref = choice([tv, fv])
        score_t = _text_score(tv, ref)
        score_f = _text_score(fv, ref)
        if score_t == score_f:
            return None
        threshold = (score_t + score_f) / 2
        return [index, [ref, threshold], score_t > score_f, "SIM"]

    def _gen_text_positional(self, index, T, F):
        """Generate PFX or SFX literal. Returns literal or None."""
        h = self.header[index]
        tv, fv = str(T[h]), str(F[h])
        if tv == fv:
            return None
        tag = choice(["PFX", "SFX"])
        if tag == "PFX":
            pfx = _common_prefix_len(fv, tv)
            threshold = (pfx + len(tv)) / 2
            return [index, [tv, threshold], True, "PFX"]
        else:
            sfx = _common_suffix_len(fv, tv)
            threshold = (sfx + len(tv)) / 2
            return [index, [tv, threshold], True, "SFX"]

    def _gen_text_charclass(self, index, T, F):
        """Generate TUC, TDC, or TSC literal. Returns literal or None."""
        h = self.header[index]
        tv, fv = str(T[h]), str(F[h])
        possible = []
        if _count_upper(tv) != _count_upper(fv):
            possible.append("TUC")
        if _count_digits(tv) != _count_digits(fv):
            possible.append("TDC")
        if _count_special(tv) != _count_special(fv):
            possible.append("TSC")
        if not possible:
            return None
        tag = choice(possible)
        if tag == "TUC":
            return [index, (_count_upper(fv) + _count_upper(tv)) / 2, _count_upper(tv) > _count_upper(fv), "TUC"]
        if tag == "TDC":
            return [index, (_count_digits(fv) + _count_digits(tv)) / 2, _count_digits(tv) > _count_digits(fv), "TDC"]
        return [index, (_count_special(fv) + _count_special(tv)) / 2, _count_special(tv) > _count_special(fv), "TSC"]

    # --- Numeric literal generators ---

    def _gen_numeric_midpoint(self, index, T, F):
        """Generate N literal (midpoint split)."""
        h = self.header[index]
        return [index, (F[h] + T[h]) / 2, T[h] > F[h], "N"]

    def _gen_numeric_digit_count(self, index, T, F):
        """Generate ND literal (digit count). Returns literal or None.

        Skipped on v5.5.0 derived columns: 'digits in a token count / has()
        flag' is a meaningless str()-based feature that only muddies the audit.
        Derived numerics get the clean midpoint split instead."""
        if self._is_derived(index):
            return None
        h = self.header[index]
        tv, fv = str(T[h]), str(F[h])
        dt = _count_digits(tv)
        df = _count_digits(fv)
        if dt == df:
            return None
        return [index, (dt + df) / 2, dt > df, "ND"]

    def _gen_numeric_zscore(self, index, T, F):
        """Generate NZ literal (z-score threshold). Returns literal or None."""
        h = self.header[index]
        stats = self._col_stats.get(index)
        if not stats:
            return self._gen_numeric_midpoint(index, T, F)
        mu, std = stats["mean"], stats["std"]
        zt = (T[h] - mu) / std
        zf = (F[h] - mu) / std
        if zt == zf:
            return None
        threshold = (zt + zf) / 2
        return [index, [mu, std, threshold], zt > zf, "NZ"]

    def _gen_numeric_logscale(self, index, T, F):
        """Generate NL literal (log-scale midpoint). Returns literal or None."""
        h = self.header[index]
        tv, fv = T[h], F[h]

        def _signed_log(x):
            if x == 0:
                return 0.0
            return math.copysign(math.log(abs(x) + 1), x)

        lt = _signed_log(tv)
        lf = _signed_log(fv)
        if lt == lf:
            return None
        return [index, (lt + lf) / 2, lt > lf, "NL"]

    def _gen_numeric_magnitude(self, index, T, F):
        """Generate NMG literal (order of magnitude). Returns literal or None."""
        h = self.header[index]
        tv, fv = T[h], F[h]

        def _mag(x):
            if x == 0:
                return 0.0
            return math.floor(math.log10(abs(x) + 1e-300))

        mt = _mag(tv)
        mf = _mag(fv)
        if mt == mf:
            return None
        return [index, (mt + mf) / 2, mt > mf, "NMG"]

    def _gen_numeric_zero(self, index, T, F):
        """Generate NZR literal (zero test). Returns literal or None."""
        h = self.header[index]
        tv, fv = T[h], F[h]
        if (tv == 0) == (fv == 0):
            return None  # both zero or both nonzero — test won't discriminate
        return [index, 0, tv == 0, "NZR"]

    def _gen_numeric_range(self, index, T, F):
        """Generate NRG literal (range). Returns literal or None."""
        h = self.header[index]
        tv, fv = T[h], F[h]
        if tv == fv:
            return None
        lo, hi = min(tv, fv), max(tv, fv)
        return [index, [lo, hi], tv < fv, "NRG"]

    def _gen_text_exact(self, index, T, F):
        """Generate TEQ literal (exact match). Returns literal or None."""
        h = self.header[index]
        tv, fv = str(T[h]), str(F[h])
        if tv == fv:
            return None
        if random() < 0.5:
            return [index, tv, False, "TEQ"]
        return [index, fv, True, "TEQ"]

    def _gen_text_startswith(self, index, T, F):
        """Generate TSW literal (starts with). Returns literal or None."""
        h = self.header[index]
        tv, fv = str(T[h]), str(F[h])
        # Find a prefix of tv that fv doesn't start with (or vice versa)
        for length in range(1, min(len(tv), 8) + 1):
            prefix = tv[:length]
            if not fv.startswith(prefix):
                return [index, prefix, False, "TSW"]
        for length in range(1, min(len(fv), 8) + 1):
            prefix = fv[:length]
            if not tv.startswith(prefix):
                return [index, prefix, True, "TSW"]
        return None

    def _gen_text_endswith(self, index, T, F):
        """Generate TEW literal (ends with). Returns literal or None."""
        h = self.header[index]
        tv, fv = str(T[h]), str(F[h])
        for length in range(1, min(len(tv), 8) + 1):
            suffix = tv[-length:]
            if not fv.endswith(suffix):
                return [index, suffix, False, "TEW"]
        for length in range(1, min(len(fv), 8) + 1):
            suffix = fv[-length:]
            if not tv.endswith(suffix):
                return [index, suffix, True, "TEW"]
        return None

    def _gen_text_vowel_ratio(self, index, T, F):
        """Generate TVR literal (vowel ratio). Returns literal or None."""
        h = self.header[index]
        tv, fv = str(T[h]), str(F[h])
        vowels = set("aeiouAEIOU")
        vt = sum(1 for c in tv if c in vowels) / max(len(tv), 1)
        vf = sum(1 for c in fv if c in vowels) / max(len(fv), 1)
        if vt == vf:
            return None
        return [index, (vt + vf) / 2, vt > vf, "TVR"]

    # --- Crypto-specific generators ---

    def _gen_crypto_entropy(self, index, T, F):
        """Generate ENT literal (Shannon entropy). Returns literal or None."""
        h = self.header[index]
        tv, fv = str(T[h]), str(F[h])
        et = _entropy(tv)
        ef = _entropy(fv)
        if et == ef:
            return None
        return [index, (et + ef) / 2, et > ef, "ENT"]

    def _gen_crypto_hex(self, index, T, F):
        """Generate HEX literal (hex char ratio). Returns literal or None."""
        h = self.header[index]
        tv, fv = str(T[h]), str(F[h])
        ht = _hex_ratio(tv)
        hf = _hex_ratio(fv)
        if ht == hf:
            return None
        return [index, (ht + hf) / 2, ht > hf, "HEX"]

    def _gen_crypto_repeat(self, index, T, F):
        """Generate REP literal (repeat period score). Returns literal or None."""
        h = self.header[index]
        tv, fv = str(T[h]), str(F[h])
        rt = _repeat_period_score(tv)
        rf = _repeat_period_score(fv)
        if rt == rf:
            return None
        return [index, (rt + rf) / 2, rt > rf, "REP"]

    def _gen_crypto_charfreq(self, index, T, F):
        """Generate CFC literal (char freq chi-squared). Returns literal or None."""
        h = self.header[index]
        tv, fv = str(T[h]), str(F[h])
        if not tv:
            return None
        # Build reference freq from T
        ref_freq = {}
        for c in tv:
            ref_freq[c] = ref_freq.get(c, 0) + 1
        nt = len(tv)
        ref_freq = {c: cnt / nt for c, cnt in ref_freq.items()}
        # Compute chi-sq of F against T's frequency
        nf = max(len(fv), 1)
        f_freq = {}
        for c in fv:
            f_freq[c] = f_freq.get(c, 0) + 1
        chi_sq = 0.0
        all_chars = set(ref_freq) | set(f_freq)
        for c in all_chars:
            observed = f_freq.get(c, 0) / nf
            expected = ref_freq.get(c, 0.001)
            chi_sq += (observed - expected) ** 2 / max(expected, 0.001)
        if chi_sq == 0:
            return None
        threshold = chi_sq / 2
        return [index, [ref_freq, threshold], True, "CFC"]

    # --- Profile oppose methods (flat inline, no dispatch overhead) ---

    def _oppose_balanced(self, T, F):
        """Equal probability across all families. No assumptions."""
        candidates = self._get_differing_candidates(T, F)
        if not candidates:
            return None
        index = self._weighted_feature_choice(candidates)
        h = self.header[index]
        if self.datatypes[index] == "T":
            r = random()
            if r < 0.167:
                lit = self._gen_text_substring(index, T, F)
            elif r < 0.333:
                lit = self._gen_text_structural(index, T, F)
            elif r < 0.500:
                lit = self._gen_text_splits(index, T, F)
            elif r < 0.667:
                lit = self._gen_text_distance(index, T, F)
            elif r < 0.833:
                lit = self._gen_text_positional(index, T, F)
            else:
                lit = self._gen_text_charclass(index, T, F)
            return lit if lit is not None else self._gen_text_substring(index, T, F)
        if self.datatypes[index] == "N":
            r = random()
            if r < 0.5:
                return self._gen_numeric_midpoint(index, T, F)
            lit = self._gen_numeric_digit_count(index, T, F)
            return lit if lit is not None else self._gen_numeric_midpoint(index, T, F)
        return None

    def _oppose_linguistic(self, T, F):
        """Heavy on edit distance + positional. For NLP / free text."""
        candidates = self._get_differing_candidates(T, F)
        if not candidates:
            return None
        index = self._weighted_feature_choice(candidates)
        if self.datatypes[index] == "T":
            r = random()
            if r < 0.35:
                lit = self._gen_text_distance(index, T, F)
            elif r < 0.55:
                lit = self._gen_text_positional(index, T, F)
            elif r < 0.70:
                lit = self._gen_text_charclass(index, T, F)
            elif r < 0.85:
                lit = self._gen_text_structural(index, T, F)
            elif r < 0.95:
                lit = self._gen_text_splits(index, T, F)
            else:
                lit = self._gen_text_substring(index, T, F)
            return lit if lit is not None else self._gen_text_substring(index, T, F)
        if self.datatypes[index] == "N":
            return self._gen_numeric_midpoint(index, T, F)
        return None

    def _oppose_industrial(self, T, F):
        """Heavy on substring. For product codes, SKUs, short labels."""
        candidates = self._get_differing_candidates(T, F)
        if not candidates:
            return None
        index = self._weighted_feature_choice(candidates)
        if self.datatypes[index] == "T":
            r = random()
            if r < 0.40:
                lit = self._gen_text_substring(index, T, F)
                if lit is not None:
                    return lit
            if r < 0.60:
                lit = self._gen_text_structural(index, T, F)
                if lit is not None:
                    return lit
            if r < 0.75:
                lit = self._gen_text_splits(index, T, F)
                if lit is not None:
                    return lit
            if r < 0.85:
                lit = self._gen_text_positional(index, T, F)
                if lit is not None:
                    return lit
            if r < 0.95:
                lit = self._gen_text_charclass(index, T, F)
                if lit is not None:
                    return lit
            return self._gen_text_substring(index, T, F)
        if self.datatypes[index] == "N":
            return self._gen_numeric_midpoint(index, T, F)
        return None

    def _oppose_cryptographic(self, T, F):
        """Entropy, hex ratio, char frequency, repeat patterns. For hashes/IDs."""
        candidates = self._get_differing_candidates(T, F)
        if not candidates:
            return None
        index = self._weighted_feature_choice(candidates)
        if self.datatypes[index] == "T":
            r = random()
            if r < 0.15:
                lit = self._gen_text_charclass(index, T, F)
            elif r < 0.25:
                lit = self._gen_crypto_entropy(index, T, F)
            elif r < 0.30:
                lit = self._gen_crypto_hex(index, T, F)
            elif r < 0.35:
                lit = self._gen_crypto_repeat(index, T, F)
            elif r < 0.60:
                lit = self._gen_text_structural(index, T, F)
            elif r < 0.75:
                lit = self._gen_text_distance(index, T, F)
            elif r < 0.85:
                lit = self._gen_text_splits(index, T, F)
            elif r < 0.95:
                lit = self._gen_text_positional(index, T, F)
            else:
                lit = self._gen_text_substring(index, T, F)
            return lit if lit is not None else self._gen_text_substring(index, T, F)
        if self.datatypes[index] == "N":
            return self._gen_numeric_midpoint(index, T, F)
        return None

    def _oppose_scientific(self, T, F):
        """Z-score, log-scale, magnitude. For numeric-heavy data."""
        candidates = self._get_differing_candidates(T, F)
        if not candidates:
            return None
        index = self._weighted_feature_choice(candidates)
        if self.datatypes[index] == "N":
            r = random()
            if r < 0.40:
                return self._gen_numeric_midpoint(index, T, F)
            if r < 0.60:
                lit = self._gen_numeric_zscore(index, T, F)
                if lit is not None:
                    return lit
            if r < 0.75:
                lit = self._gen_numeric_logscale(index, T, F)
                if lit is not None:
                    return lit
            if r < 0.90:
                lit = self._gen_numeric_magnitude(index, T, F)
                if lit is not None:
                    return lit
            lit = self._gen_numeric_digit_count(index, T, F)
            return lit if lit is not None else self._gen_numeric_midpoint(index, T, F)
        if self.datatypes[index] == "T":
            lit = self._gen_text_substring(index, T, F)
            return lit if lit is not None else self._gen_text_structural(index, T, F)
        return None

    def _oppose_categorical(self, T, F):
        """Heavy on splits + substring. For surveys, tags, enums."""
        candidates = self._get_differing_candidates(T, F)
        if not candidates:
            return None
        index = self._weighted_feature_choice(candidates)
        if self.datatypes[index] == "T":
            r = random()
            if r < 0.30:
                lit = self._gen_text_splits(index, T, F)
                if lit is not None:
                    return lit
            if r < 0.60:
                lit = self._gen_text_substring(index, T, F)
                if lit is not None:
                    return lit
            if r < 0.75:
                lit = self._gen_text_structural(index, T, F)
                if lit is not None:
                    return lit
            if r < 0.85:
                lit = self._gen_text_distance(index, T, F)
                if lit is not None:
                    return lit
            if r < 0.95:
                lit = self._gen_text_positional(index, T, F)
                if lit is not None:
                    return lit
            return self._gen_text_substring(index, T, F)
        if self.datatypes[index] == "N":
            return self._gen_numeric_midpoint(index, T, F)
        return None

    def _oppose_hef(self, T, F):
        """SIM-dominant profile for part number matching.
        Subsequence scoring (SIM) as primary, with TEQ/JAC/PFX/SFX/T support.
        Designed for short alphanumeric strings (5-30 chars) where
        customers send variant refs for the same catalog article."""
        candidates = self._get_differing_candidates(T, F)
        if not candidates:
            return None
        index = self._weighted_feature_choice(candidates)
        if self.datatypes[index] == "T":
            r = random()
            if r < 0.35:
                lit = self._gen_text_similarity(index, T, F)
                if lit is not None:
                    return lit
            if r < 0.50:
                lit = self._gen_text_exact(index, T, F)
                if lit is not None:
                    return lit
            if r < 0.65:
                lit = self._gen_text_positional(index, T, F)
                if lit is not None:
                    return lit
            if r < 0.78:
                lit = self._gen_text_distance(index, T, F)
                if lit is not None:
                    return lit
            if r < 0.88:
                lit = self._gen_text_substring(index, T, F)
                if lit is not None:
                    return lit
            if r < 0.95:
                lit = self._gen_text_charclass(index, T, F)
                if lit is not None:
                    return lit
            return self._gen_text_similarity(index, T, F) or self._gen_text_substring(index, T, F)
        if self.datatypes[index] == "N":
            return self._gen_numeric_midpoint(index, T, F)
        return None

    """
    Will return
    - For text fields: words to be or not to be included
    - For numeric fields: splits to be greater or not to be greater
    """
    def oppose(self, T, F):
        candidates = [i for i in range(1, len(self.header)) if T[self.header[i]] != F[self.header[i]]
                       and not (self.datatypes[i] == "N" and (T[self.header[i]] != T[self.header[i]] or F[self.header[i]] != F[self.header[i]]))]
        if not candidates:
            exit("Snake.oppose() — T and F are identical on all features. Dedup failed somewhere upstream. You should never see this. I'm out.")
        index = self._weighted_feature_choice(candidates)
        h = self.header[index]
        if self.datatypes[index] == "T":
            if choice(["Do it", "Don't"]) == "Do it":
                possible = []
                length = (len(F[h]) != len(T[h]))
                if length:
                    possible += ["TN"]
                alphabet = (len(list(set(F[h]))) != len(list(set(T[h]))))
                if alphabet:
                    possible += ["TLN"]
                alphabet_false = len([c for c in list(set(F[h])) if not c in T[h]]) > 0
                if alphabet_false:
                    possible += ["FA"]
                alphabet_true = len([c for c in list(set(T[h])) if not c in F[h]]) > 0
                if alphabet_true:
                    possible += ["TA"]
                word_splits = (len(F[h].split(" ")) != len(T[h].split(" ")))
                if word_splits:
                    possible += ["TWS"]
                part_splits = (len(F[h].split(",")) != len(T[h].split(",")))
                if part_splits:
                    possible += ["TPS"]
                sent_splits = (len(F[h].split(".")) != len(T[h].split(".")))
                if sent_splits:
                    possible += ["TSS"]
                if len(possible):
                    todo = choice(possible)
                    if todo == "TN":
                        return [index, (len(F[h]) + len(T[h])) / 2, len(T[h]) > len(F[h]), "TN"]
                    if todo == "TLN":
                        return [index, (len(list(set(F[h]))) + len(list(set(T[h])))) / 2, len(list(set(T[h]))) > len(list(set(F[h]))), "TLN"]
                    if todo == "FA":
                        return [index, choice([c for c in list(set(F[h])) if not c in T[h]]), True, "T"]
                    if todo == "TA":
                        return [index, choice([c for c in list(set(T[h])) if not c in F[h]]), False, "T"]
                    if todo == "TWS":
                        return [index, (len(F[h].split(" ")) + len(T[h].split(" "))) / 2, len(T[h].split(" ")) > len(F[h].split(" ")), "TWS"]
                    if todo == "TPS":
                        return [index, (len(F[h].split(",")) + len(T[h].split(","))) / 2, len(T[h].split(",")) > len(F[h].split(",")), "TPS"]
                    if todo == "TSS":
                        return [index, (len(F[h].split(".")) + len(T[h].split("."))) / 2, len(T[h].split(".")) > len(F[h].split(".")), "TSS"]
            pros = set()
            cons = set()
            for sep in [" ", "/", ":", "-"]:
                for label in T[h].split(sep):
                    pros.add(label.split("\'")[0].split('\"')[0])
                for label in F[h].split(sep):
                    cons.add(label.split("\'")[0].split('\"')[0])
            clean_pros = [label for label in pros if len(label) and len(label) < max(2,len(T[h])) and not label in F[h]]
            clean_cons = [label for label in cons if len(label) and len(label) < max(2,len(F[h])) and not label in T[h]]
            possibilities = [[index, label, False, "T"] for label in clean_pros] + [[index, label, True, "T"] for label in clean_cons]
            if len(possibilities):
                return choice(possibilities)
            else:
                if T[h] != F[h] and not T[h] in F[h]:
                    return [index, T[h], False, "T"]
                if T[h] != F[h] and not F[h] in T[h]:
                    return [index, F[h], True, "T"]
        if self.datatypes[index] == "N":
            if F[h] != F[h] or T[h] != T[h]:  # NaN guard: NaN != NaN is True
                exit("Snake.oppose() — NaN slipped past floatconversion into a numeric feature. Impressive failure. I'm out.")
            return [index, (F[h] + T[h]) / 2, T[h] > F[h], "N"]
        exit("Snake.oppose() — feature datatype is neither 'N' nor 'T'. Someone broke the type detector. I'm out.")

    """
    Will return:
    - True if the datapoint satisfies the literal
    - False if the datapoint misses the header value or do not satisfy the literal
    Robust.
    """
    def apply_literal(self, X, literal):
        if _HAS_ACCEL:
            return apply_literal_fast(X, literal, self.header)
        index = literal[0]
        value = literal[1]
        negat = literal[2]
        datat = literal[3]
        if self.header[index] not in X:
            return False
        field = X[self.header[index]]
        if datat == "TWS":
            if negat:
                return value <= len(field.split(" "))
            return value > len(field.split(" "))
        elif datat == "TPS":
            if negat:
                return value <= len(field.split(","))
            return value > len(field.split(","))
        elif datat == "TSS":
            if negat:
                return value <= len(field.split("."))
            return value > len(field.split("."))
        elif datat == "TLN":
            if negat:
                return value <= len(list(set(field)))
            return value > len(list(set(field)))
        elif datat == "TN":
            if negat:
                return value <= len(field)
            return value > len(field)
        elif datat == "T":
            if negat:
                return value not in field
            return value in field
        elif datat == "N":
            if negat:
                return value <= field
            return value > field
        # --- New literal types (v5.2.1) ---
        elif datat == "ND":
            dc = _count_digits(str(field))
            return value <= dc if negat else value > dc
        elif datat == "TUC":
            uc = _count_upper(str(field))
            return value <= uc if negat else value > uc
        elif datat == "TDC":
            dc = _count_digits(str(field))
            return value <= dc if negat else value > dc
        elif datat == "TSC":
            sc = _count_special(str(field))
            return value <= sc if negat else value > sc
        elif datat == "LEV":
            ref, threshold = value[0], value[1]
            d = _levenshtein(str(field), ref)
            return d <= threshold if negat else d > threshold
        elif datat == "JAC":
            ref, threshold = value[0], value[1]
            j = _jaccard_bigrams(str(field), ref)
            return j >= threshold if negat else j < threshold
        elif datat == "SIM":
            ref, threshold = value[0], value[1]
            s = _text_score(str(field), ref)
            return s >= threshold if negat else s < threshold
        elif datat == "PFX":
            ref, threshold = value[0], value[1]
            p = _common_prefix_len(str(field), ref)
            return p >= threshold if negat else p < threshold
        elif datat == "SFX":
            ref, threshold = value[0], value[1]
            s = _common_suffix_len(str(field), ref)
            return s >= threshold if negat else s < threshold
        elif datat == "ENT":
            e = _entropy(str(field))
            return value <= e if negat else value > e
        elif datat == "HEX":
            h = _hex_ratio(str(field))
            return value <= h if negat else value > h
        elif datat == "REP":
            r = _repeat_period_score(str(field))
            return value <= r if negat else value > r
        elif datat == "CFC":
            ref_freq, threshold = value[0], value[1]
            sf = str(field)
            nf = max(len(sf), 1)
            f_freq = {}
            for c in sf:
                f_freq[c] = f_freq.get(c, 0) + 1
            chi_sq = 0.0
            all_chars = set(ref_freq) | set(f_freq)
            for c in all_chars:
                observed = f_freq.get(c, 0) / nf
                expected = ref_freq.get(c, 0.001)
                chi_sq += (observed - expected) ** 2 / max(expected, 0.001)
            return chi_sq < threshold if negat else chi_sq >= threshold
        elif datat == "NZ":
            mu, std, threshold = value[0], value[1], value[2]
            z = (field - mu) / std
            return z >= threshold if negat else z < threshold
        elif datat == "NL":
            def _signed_log(x):
                if x == 0:
                    return 0.0
                return math.copysign(math.log(abs(x) + 1), x)
            lv = _signed_log(field)
            return value <= lv if negat else value > lv
        elif datat == "NMG":
            def _mag(x):
                if x == 0:
                    return 0.0
                return math.floor(math.log10(abs(x) + 1e-300))
            m = _mag(field)
            return value <= m if negat else value > m
        elif datat == "NZR":
            return (field != 0) if negat else (field == 0)
        elif datat == "NRG":
            lo, hi = value[0], value[1]
            inside = lo < field <= hi
            return inside if negat else not inside
        elif datat == "TEQ":
            sfield = str(field)
            return (sfield != value) if negat else (sfield == value)
        elif datat == "TSW":
            sfield = str(field)
            return (not sfield.startswith(value)) if negat else sfield.startswith(value)
        elif datat == "TEW":
            sfield = str(field)
            return (not sfield.endswith(value)) if negat else sfield.endswith(value)
        elif datat == "TVR":
            sfield = str(field)
            vowels = set("aeiouAEIOU")
            vr = sum(1 for c in sfield if c in vowels) / max(len(sfield), 1)
            return value <= vr if negat else value > vr
        return False

    """
    Applies an or Statement on the literals
    """
    def apply_clause(self, X, clause):
        if _HAS_ACCEL:
            return apply_clause_fast(X, clause, self.header)
        for literal in clause:
            if self.apply_literal(X, literal):
                return True
        return False

    def _oppose_lookahead(self, Ts, F):
        """Generate K oppose literals, return the one covering the most Ts.
        Guard: reject any literal that is True on F OR not True on T."""
        k = getattr(self, 'lookahead', 5)
        _oppose = self._active_oppose if hasattr(self, '_active_oppose') else self.oppose
        if k <= 1:
            T = choice(Ts)
            lit = _oppose(T, F)
            if lit is not None and (self.apply_literal(F, lit) or not self.apply_literal(T, lit)):
                return None
            return lit
        best_lit, best_cov = None, -1
        for _ in range(k):
            T = choice(Ts)
            lit = _oppose(T, F)
            if lit is None:
                continue
            if self.apply_literal(F, lit) or not self.apply_literal(T, lit):
                continue
            cov = sum(1 for t in Ts if self.apply_literal(t, lit))
            if cov > best_cov:
                best_cov = cov
                best_lit = lit
        return best_lit

    """
    Constructs a minimal clause to discriminate F relative to Ts
    - True on all Ts
    - False on at least F
    - Minimal
    """
    def construct_clause(self, F, Ts):
        lit = self._oppose_lookahead(Ts, F)
        if lit is None:
            lit = self.oppose(choice(Ts), F)
        clause = [lit]
        Ts_remainder = [T for T in Ts if not self.apply_literal(T, clause[-1])]
        while len(Ts_remainder):
            lit = self._oppose_lookahead(Ts_remainder, F)
            if lit is None:
                lit = self.oppose(choice(Ts_remainder), F)
            clause.append(lit)
            Ts_remainder = [T for T in Ts_remainder if not self.apply_literal(T, clause[-1])]
        i = 0
        while i < len(clause):
            sub_clause = [clause[j] for j in range(len(clause)) if i != j]
            minimal_test = False
            for T in Ts:
                if not self.apply_clause(T, sub_clause):
                    minimal_test = True
                    break
            if minimal_test:
                i += 1
            else:
                clause = sub_clause
        return clause

    """
    Constructs a minimal SAT Instance for a target value
    """
    def construct_sat(self, target_value):
        Fs = [self.population[i] for i in range(len(self.population)) if self.targets[i] == target_value]
        Ts = [self.population[i] for i in range(len(self.population)) if self.targets[i] != target_value]
        sat = []
        while len(Fs):
            F = choice(Fs)
            clause = self.construct_clause(F, Ts)
            if not clause:
                Fs = [f for f in Fs if f is not F]
                self.qprint(f"# WARNING: empty clause in construct_sat for target [{target_value}], {len(Fs)} Fs remaining", level=2)
                continue
            consequence = [i for i in range(len(self.population)) if self.targets[i] == target_value and not self.apply_clause(self.population[i], clause)]
            Fs = [F for F in Fs if self.apply_clause(F, clause)]
            sat += [[clause, consequence]]
        return sat

    # ------------------------------------------------------------------
    # Bucketed layer construction
    # ------------------------------------------------------------------

    def _construct_local_sat(self, member_indices):
        """Run construct_sat scoped to a bucket's member indices. Returns (clauses, lookalikes) with 0-based local indexing."""
        local_pop = [self.population[i] for i in member_indices]
        local_targets = [self.targets[i] for i in member_indices]
        # Deduplicate local targets using _target_key for unhashable types
        seen_keys = set()
        unique_local = []
        for t in local_targets:
            k = self._target_key(t)
            if k not in seen_keys:
                seen_keys.add(k)
                unique_local.append(t)
        try:
            target_values = sorted(unique_local)
        except TypeError:
            target_values = sorted(unique_local, key=lambda t: self._target_key(t))
        local_clauses = []
        local_lookalikes = {str(l): [] for l in range(len(local_pop))}

        n_local = len(local_pop)
        m = len(self.header) - 1
        self.qprint(f"#     [SAT] local SAT: {n_local} samples, {len(target_values)} targets, O(m*n^2)={m * n_local * n_local:,}")

        t_sat_all = time()
        for tv_idx, target_value in enumerate(target_values):
            t_target = time()
            Fs = [local_pop[i] for i in range(len(local_pop)) if local_targets[i] == target_value]
            Ts = [local_pop[i] for i in range(len(local_pop)) if local_targets[i] != target_value]
            if not Ts:
                self.qprint(f"#     [SAT] target {tv_idx+1}/{len(target_values)} [{target_value}]: skipped (no negatives)", level=2)
                continue
            n_fs_start = len(Fs)
            sat = []
            while len(Fs):
                F = choice(Fs)
                clause = self.construct_clause(F, Ts)
                if not clause:
                    Fs = [f for f in Fs if f is not F]
                    self.qprint(f"# WARNING: empty clause for target [{target_value}], {len(Fs)} Fs remaining", level=2)
                    continue
                # --- Inline clause contract check ---
                broken = [T for T in Ts if not self.apply_clause(T, clause)]
                if broken:
                    import json as _j
                    _dump = {
                        "error": "clause FALSE on T immediately after construct_clause",
                        "target_value": str(target_value),
                        "n_broken": len(broken),
                        "clause_len": len(clause),
                        "clause": str(clause),
                        "F": {k: str(v) for k, v in F.items()},
                        "broken_T": {k: str(v) for k, v in broken[0].items()},
                        "broken_T_class": str(broken[0].get(self.target, "?")),
                        "per_literal": [
                            {"literal": str(lit), "on_F": self.apply_literal(F, lit), "on_broken_T": self.apply_literal(broken[0], lit)}
                            for lit in clause
                        ],
                    }
                    with open("/tmp/snake_fatal.json", "w") as _ef:
                        _ef.write(_j.dumps(_dump, indent=2))
                    exit(f"FATAL: clause FALSE on {len(broken)} Ts — /tmp/snake_fatal.json")
                # --- End check ---
                if _HAS_ACCEL:
                    consequence, _ = filter_consequence_fast(local_pop, local_targets, target_value, clause, self.header)
                else:
                    consequence = [i for i in range(len(local_pop)) if local_targets[i] == target_value and not self.apply_clause(local_pop[i], clause)]
                Fs = [f for f in Fs if self.apply_clause(f, clause)]
                sat += [[clause, consequence]]

            lookalikes_for_target = {str(l): [] for l in range(len(local_pop)) if local_targets[l] == target_value}
            for pair in sat:
                local_clauses.append(pair[0])
                for l in pair[1]:
                    lookalikes_for_target[str(l)].append(len(local_clauses) - 1)
            for l in lookalikes_for_target:
                local_lookalikes[str(l)].append(lookalikes_for_target[str(l)])

            target_time = time() - t_target
            targets_done = tv_idx + 1
            targets_left = len(target_values) - targets_done
            elapsed_sat = time() - t_sat_all
            if targets_done > 0:
                avg_per_target = elapsed_sat / targets_done
                eta_targets = avg_per_target * targets_left
            else:
                eta_targets = 0
            self.qprint(f"#     [SAT] target {targets_done}/{len(target_values)} [{target_value}]: {n_fs_start} positives -> {len(sat)} clauses in {target_time:.2f}s — ETA {eta_targets:.2f}s", level=2)

        total_sat_time = time() - t_sat_all
        self.qprint(f"#     [SAT] local SAT complete: {len(local_clauses)} total clauses in {total_sat_time:.2f}s")

        return local_clauses, local_lookalikes

    """
    Constructs a logical layer of lookalikes (bucketed)
    """
    def construct_layer(self):
        t_layer = time()
        n = len(self.population)
        m = len(self.header) - 1
        k = len(self._unique_targets())
        self.qprint(f"#   [layer] Building bucket chain... O(n={n}, m={m}, k={k})")

        _oppose = self._active_oppose if hasattr(self, '_active_oppose') else self.oppose
        chain = build_bucket_chain(
            self.population, self.targets, self.bucket,
            _oppose, self.apply_literal, self.noise,
            log_fn=self.qprint, header=self.header
        )

        self.qprint(f"#   [layer] Bucket chain ready: {len(chain)} buckets. Now constructing SAT per bucket...")
        t_sat_start = time()
        for b_idx, entry in enumerate(chain):
            t_bucket = time()
            n_b = len(entry["members"])
            k_b = len({self._target_key(self.targets[i]) for i in entry["members"]})
            # O(m * n_b^2) per bucket SAT construction
            complexity = m * n_b * n_b
            cond_type = f"IF({len(entry['condition'])} lit)" if entry["condition"] else "ELSE"
            self.qprint(f"#   [layer] BUCKET {b_idx}/{len(chain)} ({cond_type}): {n_b} members, {k_b} classes, complexity O(m*n^2)={complexity:,}")

            entry["clauses"], entry["lookalikes"] = self._construct_local_sat(entry["members"])

            bucket_time = time() - t_bucket
            buckets_done = b_idx + 1
            buckets_left = len(chain) - buckets_done
            if buckets_done > 0:
                avg_bucket = (time() - t_sat_start) / buckets_done
                eta_buckets = avg_bucket * buckets_left
            else:
                eta_buckets = 0
            self.qprint(f"#   [layer] BUCKET {b_idx} DONE: {len(entry['clauses'])} clauses in {bucket_time:.2f}s — ETA remaining buckets: {eta_buckets:.2f}s")

            # Global ETA: remaining buckets in this layer + avg_per_layer * remaining layers
            layers_left = self.n_layers - (self._current_layer + 1)
            global_eta = eta_buckets + self._avg_per_layer * layers_left
            elapsed = time() - self._t0
            self._write_progress(
                self._current_layer + 1, global_eta, elapsed,
                bucket=buckets_done, n_buckets=len(chain), eta_bucket_seconds=eta_buckets,
            )

        layer_time = time() - t_layer
        self.qprint(f"#   [layer] Layer construction total: {layer_time:.2f}s")
        self.layers.append(chain)

    # ------------------------------------------------------------------
    # Prediction pipeline (bucketed)
    # ------------------------------------------------------------------

    """
    Predict the probability vector for a given X
    """
    def _normalize_features(self, X):
        """Coerce an incoming datapoint to the model's trained feature contract.

        Enforces, on every new datapoint, exactly what training enforced:
          - type per column (float for N, str for T) — so str(int) vs str(float)
            and literal types like ND/TUC/TDC/TSC behave identically to training;
          - NaN/None filled by design — 0.0 for numeric fields, "" for text;
          - MISSING columns populated as full-NA — a feature absent from X is
            treated as all-missing and filled with its default, so a row always
            scores against every trained clause instead of silently mismatching.

        The result is a complete dict over self.header[1:], NaN-free, typed."""
        # v5.5.0: grow the same derived columns the model was trained on, from
        # the raw source columns, BEFORE coercion — so the derived "N" columns
        # exist in `out` and score against every trained clause. Reads sources
        # off a shallow copy so the caller's dict is untouched.
        if getattr(self, "expansions", None):
            X = self._expand_row(dict(self._as_single(X)))
        out = {}
        for i in range(1, len(self.header)):
            h = self.header[i]
            numeric = self.datatypes[i] == "N"
            if h not in X or _is_missing(X[h]):
                # Absent column or missing value -> default fill.
                out[h] = 0.0 if numeric else ""
            elif numeric:
                try:
                    out[h] = float(X[h])
                except (ValueError, TypeError):
                    out[h] = X[h]
            else:
                out[h] = X[h]
        return out

    def _normalize_batch(self, Xs):
        """Coerce a batch (list of dicts OR a DataFrame) to a list of clean,
        typed, NaN-free dicts via _normalize_features. The explicit-batch fast
        paths (get_batch_prediction/candles/regression) call the Cython kernel
        directly on raw rows, so they must normalize here to inherit the same
        NaN-by-design + missing-column contract as the single-dict path."""
        batch = self._as_batch(Xs)
        rows = batch if batch is not None else Xs
        return [self._normalize_features(self._as_single(X)) for X in rows]

    # ------------------------------------------------------------------
    # v5.4.8 — population guard + parallel batch inference
    # ------------------------------------------------------------------

    def _require_population(self, method):
        """Raise a clear error when a population-dependent method is called on a
        stripped model (loaded from to_json(stripped=True))."""
        if self._stripped or not self.population:
            raise RuntimeError(
                f"{method}() needs the training population, but this model was "
                "loaded stripped (no population). Stripped models serve the hot "
                "path (prediction / probability / lookalikes / candle / "
                "regression) only. Re-save with to_json(stripped=False) and "
                "load the full model to use audit / augmented."
            )

    def _infer_pool_size(self, n_items):
        """Proportionate worker count: one chunk per worker, never more workers
        than items, capped by available CPU (or self._max_workers)."""
        import os
        cap = self._max_workers or os.cpu_count() or 1
        return max(1, min(cap, n_items))

    def _as_batch(self, X):
        """If X is a batch of datapoints, return it as a list of row-dicts;
        otherwise return None (X is a single datapoint).

        Accepts a list of dicts or a pandas DataFrame. Duck-typed — a DataFrame
        is anything with a callable to_dict() and a `columns` attribute (which
        excludes a pd.Series, that has to_dict() but is a single row). Snake
        never imports pandas, so the zero-dependency guarantee holds: pandas
        objects flow through only because the caller brought them.
        """
        if isinstance(X, list):
            return X
        to_dict = getattr(X, "to_dict", None)
        if callable(to_dict) and hasattr(X, "columns"):
            records = X.to_dict("records")
            if not records:
                raise ValueError("Empty DataFrame")
            return records
        return None

    def _as_single(self, X):
        """Coerce a single datapoint to a plain dict. A pd.Series (one row) has
        a callable to_dict() but no `columns`; convert it so plain dict semantics
        hold in every downstream method. A real dict passes through untouched."""
        if isinstance(X, dict):
            return X
        to_dict = getattr(X, "to_dict", None)
        if callable(to_dict) and not hasattr(X, "columns"):
            return X.to_dict()
        return X

    def _parallel_infer(self, method_name, Xs):
        """Run a single-dict inference method over a list of datapoints, dividing
        the work across a proportionate number of CPU processes.

        Order-preserving and EXACT: each datapoint is scored independently by the
        same single-dict method, so the parallel result is element-for-element
        identical to [getattr(self, method_name)(X) for X in Xs]. Pure-Python
        inference is GIL-bound, so we fork processes (not threads) to use every
        core. Small batches and single-worker cases run inline — IPC isn't worth
        it below self._parallel_threshold.
        """
        if not isinstance(Xs, list):
            raise TypeError("_parallel_infer expects a list of datapoints")
        if len(Xs) == 0:
            return []

        n_workers = self._infer_pool_size(len(Xs))
        if n_workers == 1 or len(Xs) < self._parallel_threshold:
            method = getattr(self, method_name)
            return [method(X) for X in Xs]

        # Balanced contiguous chunks — one per worker, order preserved on merge.
        k, r = divmod(len(Xs), n_workers)
        chunks, start = [], 0
        for w in range(n_workers):
            size = k + (1 if w < r else 0)
            if size:
                chunks.append((start, Xs[start:start + size]))
                start += size

        import multiprocessing
        if "fork" not in multiprocessing.get_all_start_methods():
            # Platform without fork (e.g. Windows): fall back to inline.
            method = getattr(self, method_name)
            return [method(X) for X in Xs]
        ctx = multiprocessing.get_context("fork")

        # Share the model with workers via fork-inherited COW, NOT pickling.
        # Setting the module global before Pool() means each forked child sees
        # this exact model in its address space at no copy cost — the whole
        # point of stripping the population: the model is small and shared.
        global _infer_model
        _infer_model = self
        try:
            results = [None] * len(Xs)
            jobs = [(method_name, start, chunk) for (start, chunk) in chunks]
            with ctx.Pool(n_workers) as pool:
                for start, chunk_out in pool.imap_unordered(_infer_chunk_worker, jobs):
                    results[start:start + len(chunk_out)] = chunk_out
        finally:
            _infer_model = None
        return results

    def get_lookalikes(self, X):
        batch = self._as_batch(X)
        if batch is not None:
            return self._parallel_infer("get_lookalikes", batch)
        X = self._normalize_features(self._as_single(X))
        if _HAS_ACCEL:
            return get_lookalikes_fast(self.layers, X, self.header, self.targets)
        all_lookalikes = []
        for layer in self.layers:
            bucket = traverse_chain(layer, X, self.apply_literal)
            if bucket is None:
                continue
            clause_bool = [self.apply_clause(X, c) for c in bucket["clauses"]]
            negated = {i for i in range(len(clause_bool)) if not clause_bool[i]}
            for l in bucket["lookalikes"]:
                for condition in bucket["lookalikes"][l]:
                    if all(c_idx in negated for c_idx in condition):
                        global_idx = bucket["members"][int(l)]
                        all_lookalikes.append([global_idx, self.targets[global_idx], condition])
        return all_lookalikes

    def get_lookalikes_labeled(self, X):
        """Like get_lookalikes but each entry includes origin: 'c' (core) or 'n' (noise).
        Returns list of [global_idx, target_value, condition, origin]."""
        batch = self._as_batch(X)
        if batch is not None:
            return self._parallel_infer("get_lookalikes_labeled", batch)
        X = self._normalize_features(self._as_single(X))
        all_lookalikes = []
        for layer in self.layers:
            bucket = traverse_chain(layer, X, self.apply_literal)
            if bucket is None:
                continue
            origins = bucket.get("origins")
            clause_bool = [self.apply_clause(X, c) for c in bucket["clauses"]]
            negated = {i for i in range(len(clause_bool)) if not clause_bool[i]}
            for l in bucket["lookalikes"]:
                for condition in bucket["lookalikes"][l]:
                    if all(c_idx in negated for c_idx in condition):
                        global_idx = bucket["members"][int(l)]
                        origin = origins[int(l)] if origins else "c"
                        all_lookalikes.append([global_idx, self.targets[global_idx], condition, origin])
        return all_lookalikes

    def _get_probability_from_lookalikes(self, lookalikes):
        """Compute probability vector from a pre-computed lookalikes list.
        Returns list of (target_value, probability) tuples to support unhashable targets."""
        target_values = self._sorted_unique_targets()
        if len(lookalikes) == 0:
            return [(tv, 1 / len(target_values)) for tv in target_values]
        return [(tv, sum((triple[1] == tv for triple in lookalikes)) / len(lookalikes)) for tv in target_values]

    def _prob_to_dict(self, prob_tuples):
        """Convert probability tuples to a dict. Uses _target_key for unhashable targets."""
        try:
            return {tv: p for tv, p in prob_tuples}
        except TypeError:
            return {self._target_key(tv): p for tv, p in prob_tuples}

    def _prediction_from_prob(self, prob_tuples):
        """Return the target value with highest probability from tuples list."""
        best_tv, best_p = prob_tuples[0]
        for tv, p in prob_tuples[1:]:
            if p > best_p:
                best_tv, best_p = tv, p
        return best_tv

    """
    Gives the probability vector associated
    """
    def get_probability(self, X):
        batch = self._as_batch(X)
        if batch is not None:
            return self._parallel_infer("get_probability", batch)
        lookalikes = self.get_lookalikes(X)
        return self._prob_to_dict(self._get_probability_from_lookalikes(lookalikes))

    """
    Predicts the outcome for a datapoint
    """
    def get_prediction(self, X):
        batch = self._as_batch(X)
        if batch is not None:
            return self._parallel_infer("get_prediction", batch)
        lookalikes = self.get_lookalikes(X)
        prob_tuples = self._get_probability_from_lookalikes(lookalikes)
        return self._prediction_from_prob(prob_tuples)

    """
    Augments a datapoint with every available information
    """
    def get_augmented(self, X):
        batch = self._as_batch(X)
        if batch is not None:
            return self._parallel_infer("get_augmented", batch)
        self._require_population("get_augmented")
        X = self._as_single(X)
        Y = X.copy()                       # keep the caller's original metadata (NaN and all)
        Xn = self._normalize_features(X)   # NaN-free, typed view for scoring + audit rendering
        lookalikes = self.get_lookalikes(Xn)
        prob_tuples = self._get_probability_from_lookalikes(lookalikes)
        probability = self._prob_to_dict(prob_tuples)
        prediction = self._prediction_from_prob(prob_tuples)
        Y["Lookalikes"] = lookalikes
        Y["Probability"] = probability
        Y["Prediction"] = prediction
        Y["Audit"] = self._get_audit_with_precomputed(Xn, lookalikes, probability, prediction)
        return Y

    """
    Distribution candle of the lookalike y values for a single datapoint.
    Returns a Candle (high/q3/median/q1/low/mean/std/n). Intended for
    regression-style continuous targets — falls back gracefully on classes
    that are numerically coercible, returns NaNs + n=0 otherwise.
    """
    def get_candle(self, X):
        batch = self._as_batch(X)
        if batch is not None:
            return self._parallel_infer("get_candle", batch)
        lookalikes = self.get_lookalikes(X)
        return compute_candle([la[1] for la in lookalikes])

    """
    Batch candles for a list of datapoints. Mirrors get_batch_prediction's
    routing path so amortized Cython lookups are reused when available.
    """
    def get_batch_candles(self, Xs):
        Xs = self._normalize_batch(Xs)
        if _HAS_ACCEL:
            all_lookalikes = batch_get_lookalikes_fast(
                self.layers, Xs, self.header, self.targets
            )
            return [compute_candle([la[1] for la in lks]) for lks in all_lookalikes]
        return [self.get_candle(X) for X in Xs]

    """
    Regression prediction for a continuous (float) target. Returns the
    IQR-trimmed mean of the lookalike distribution — the robust point
    estimate from the candle. Use this instead of get_prediction when y
    is continuous: get_prediction picks a single lookalike y (mode-like),
    while get_regression averages the consensus middle of the distribution.
    """
    def get_regression(self, X):
        batch = self._as_batch(X)
        if batch is not None:
            return self._parallel_infer("get_regression", batch)
        return self.get_candle(X).iqr_mean

    """
    Batch regression: list of float predictions, sharing the Cython
    lookalike fast path with get_batch_candles.
    """
    def get_batch_regression(self, Xs):
        return [c.iqr_mean for c in self.get_batch_candles(Xs)]

    """
    Batch prediction for a list of datapoints.
    Returns list of {"prediction": ..., "probability": ..., "confidence": ...} dicts.
    Delegates to Cython batch_predict_fast when available for maximum throughput.
    """
    def get_batch_prediction(self, Xs):
        Xs = self._normalize_batch(Xs)
        if _HAS_ACCEL:
            # Use batch_get_lookalikes_fast for amortized routing, then derive per-query
            all_lookalikes = batch_get_lookalikes_fast(
                self.layers, Xs, self.header, self.targets
            )
            results = []
            for lookalikes in all_lookalikes:
                prob_tuples = self._get_probability_from_lookalikes(lookalikes)
                probability = self._prob_to_dict(prob_tuples)
                prediction = self._prediction_from_prob(prob_tuples)
                confidence = max(p for _, p in prob_tuples)
                results.append({
                    "prediction": prediction,
                    "probability": probability,
                    "confidence": confidence,
                })
            return results
        results = []
        for X in Xs:
            lookalikes = self.get_lookalikes(X)
            prob_tuples = self._get_probability_from_lookalikes(lookalikes)
            probability = self._prob_to_dict(prob_tuples)
            prediction = self._prediction_from_prob(prob_tuples)
            confidence = max(p for _, p in prob_tuples)
            results.append({
                "prediction": prediction,
                "probability": probability,
                "confidence": confidence,
            })
        return results

    # ------------------------------------------------------------------
    # Enhanced Audit
    # ------------------------------------------------------------------

    def _negate_literal(self, literal):
        """Return a literal with flipped negation."""
        index, value, negat, datat = literal
        return (index, value, not negat, datat)

    def _first_failing_literal(self, X, condition):
        """Find the first literal in a condition that evaluates to False for X."""
        for lit in condition:
            if not self.apply_literal(X, lit):
                return lit
        return None

    def _format_literal_text(self, literal):
        """Human-readable description of a single literal."""
        index, value, negat, datat = literal
        h = self.header[index]
        if datat == "T":
            if negat:
                return f'"{h}" does NOT contain "{value}"'
            return f'"{h}" contains "{value}"'
        if datat == "TN":
            if negat:
                return f'len("{h}") >= {value}'
            return f'len("{h}") < {value}'
        if datat == "TLN":
            if negat:
                return f'alphabet("{h}") >= {value}'
            return f'alphabet("{h}") < {value}'
        if datat == "TWS":
            if negat:
                return f'words("{h}") >= {value}'
            return f'words("{h}") < {value}'
        if datat == "TPS":
            if negat:
                return f'parts("{h}") >= {value}'
            return f'parts("{h}") < {value}'
        if datat == "TSS":
            if negat:
                return f'sentences("{h}") >= {value}'
            return f'sentences("{h}") < {value}'
        if datat == "N":
            if negat:
                return f'"{h}" <= {value}'
            return f'"{h}" > {value}'
        # --- New literal types (v5.2.1) ---
        if datat == "ND":
            return f'digits("{h}") >= {value}' if negat else f'digits("{h}") < {value}'
        if datat == "TUC":
            return f'uppercase("{h}") >= {value}' if negat else f'uppercase("{h}") < {value}'
        if datat == "TDC":
            return f'digit_chars("{h}") >= {value}' if negat else f'digit_chars("{h}") < {value}'
        if datat == "TSC":
            return f'special_chars("{h}") >= {value}' if negat else f'special_chars("{h}") < {value}'
        if datat == "LEV":
            ref, threshold = value[0], value[1]
            return f'levenshtein("{h}", "{ref}") <= {threshold:.1f}' if negat else f'levenshtein("{h}", "{ref}") > {threshold:.1f}'
        if datat == "JAC":
            ref, threshold = value[0], value[1]
            return f'jaccard("{h}", "{ref}") >= {threshold:.2f}' if negat else f'jaccard("{h}", "{ref}") < {threshold:.2f}'
        if datat == "PFX":
            ref, threshold = value[0], value[1]
            return f'prefix("{h}", "{ref}") >= {threshold:.1f}' if negat else f'prefix("{h}", "{ref}") < {threshold:.1f}'
        if datat == "SFX":
            ref, threshold = value[0], value[1]
            return f'suffix("{h}", "{ref}") >= {threshold:.1f}' if negat else f'suffix("{h}", "{ref}") < {threshold:.1f}'
        if datat == "ENT":
            return f'entropy("{h}") >= {value:.2f}' if negat else f'entropy("{h}") < {value:.2f}'
        if datat == "HEX":
            return f'hex_ratio("{h}") >= {value:.2f}' if negat else f'hex_ratio("{h}") < {value:.2f}'
        if datat == "REP":
            return f'repeat_score("{h}") >= {value:.2f}' if negat else f'repeat_score("{h}") < {value:.2f}'
        if datat == "CFC":
            _, threshold = value[0], value[1]
            return f'charfreq_chi("{h}") < {threshold:.2f}' if negat else f'charfreq_chi("{h}") >= {threshold:.2f}'
        if datat == "NZ":
            mu, std, threshold = value[0], value[1], value[2]
            return f'zscore("{h}") >= {threshold:.2f}' if negat else f'zscore("{h}") < {threshold:.2f}'
        if datat == "NL":
            return f'log("{h}") >= {value:.2f}' if negat else f'log("{h}") < {value:.2f}'
        if datat == "NMG":
            return f'magnitude("{h}") >= {value}' if negat else f'magnitude("{h}") < {value}'
        if datat == "NZR":
            return f'"{h}" != 0' if negat else f'"{h}" == 0'
        if datat == "NRG":
            lo, hi = value[0], value[1]
            return f'{lo} < "{h}" <= {hi}' if negat else f'NOT ({lo} < "{h}" <= {hi})'
        if datat == "TEQ":
            return f'"{h}" != "{value}"' if negat else f'"{h}" == "{value}"'
        if datat == "TSW":
            return f'"{h}" NOT starts with "{value}"' if negat else f'"{h}" starts with "{value}"'
        if datat == "TEW":
            return f'"{h}" NOT ends with "{value}"' if negat else f'"{h}" ends with "{value}"'
        if datat == "TVR":
            return f'vowel_ratio("{h}") >= {value:.2f}' if negat else f'vowel_ratio("{h}") < {value:.2f}'
        return str(literal)

    def _format_bar(self, pct, width=20):
        filled = int(pct / 100 * width)
        return "\u2588" * filled + "\u2591" * (width - filled)

    def get_plain_text_assertion(self, condition, l, bucket_clauses=None, bucket_members=None):
        """Human-readable AND statement explaining why a lookalike matched.

        Args:
            condition: list of local clause indices
            l: local index within bucket
            bucket_clauses: list of clauses from the matched bucket
            bucket_members: list of global indices for the bucket
        """
        # Resolve global index
        if bucket_members is not None:
            global_idx = bucket_members[int(l)]
        else:
            global_idx = int(l)
        target_val = self.targets[global_idx]

        # Extract a short label from the sample (first text feature, truncated)
        sample = self.population[global_idx]
        label_parts = []
        for h in self.header[1:]:
            v = sample.get(h, "")
            if isinstance(v, str) and v:
                label_parts.append(v)
                break
        sample_label = (label_parts[0][:60] if label_parts else str(global_idx))

        if bucket_clauses is None:
            # Backwards compat: no clause data available
            return f"    Lookalike #{global_idx} [{target_val}]: {sample_label}\n      AND: matched via condition {condition}"

        # Build the AND statement by negating each literal in each clause
        # A lookalike is matched when ALL its clauses evaluate to FALSE on X.
        # A clause (OR of literals) is FALSE iff EVERY literal in it is FALSE.
        # Negating each literal gives us what IS true when the literal is false.
        and_parts = []
        seen = set()
        for c_idx in condition:
            if c_idx < len(bucket_clauses):
                clause = bucket_clauses[c_idx]
                for lit in clause:
                    negated = self._negate_literal(lit)
                    desc = self._format_literal_text(negated)
                    if desc not in seen:
                        seen.add(desc)
                        and_parts.append(desc)

        and_str = " AND ".join(and_parts) if and_parts else "(unconditional)"
        return f"    Lookalike #{global_idx} [{target_val}]: {sample_label}\n      AND: {and_str}"

    """
    Audit for a given datapoint — enhanced per-layer ASCII
    R.A.G.
    """
    def get_audit(self, X):
        batch = self._as_batch(X)
        if batch is not None:
            return self._parallel_infer("get_audit", batch)
        self._require_population("get_audit")
        X = self._normalize_features(self._as_single(X))  # NaN-free, typed — apply_literal needs a clean dict
        lookalikes = self.get_lookalikes(X)
        prob_tuples = self._get_probability_from_lookalikes(lookalikes)
        probability = self._prob_to_dict(prob_tuples)
        prediction = self._prediction_from_prob(prob_tuples)
        return self._get_audit_with_precomputed(X, lookalikes, probability, prediction)

    def _get_audit_with_precomputed(self, X, lookalikes, probability, prediction):
        """Build audit string using pre-computed lookalikes, probability, and prediction."""
        audit = "### BEGIN AUDIT ###\n"
        audit += f"  Prediction: {prediction}\n"
        audit += f"  Layers: {len(self.layers)}, Lookalikes: {len(lookalikes)}\n"

        # --- LOOKALIKE SUMMARY ---
        audit += f"\n  LOOKALIKE SUMMARY\n  {'='*48}\n"
        if lookalikes:
            # Group by target
            counts = {}
            key_to_val = {}
            examples = {}
            for triple in lookalikes:
                global_idx, target_val, _ = triple
                k = self._target_key(target_val)
                counts[k] = counts.get(k, 0) + 1
                key_to_val[k] = target_val
                if k not in examples:
                    examples[k] = []
                if len(examples[k]) < 2:
                    # Extract sample label
                    sample = self.population[global_idx]
                    for h in self.header[1:]:
                        v = sample.get(h, "")
                        if isinstance(v, str) and v:
                            examples[k].append(v[:60])
                            break
                    else:
                        examples[k].append(str(global_idx))
            total = len(lookalikes)
            for k in sorted(counts, key=lambda x: -counts[x]):
                t = key_to_val[k]
                cnt = counts[k]
                pct = 100 * cnt / total
                bar = self._format_bar(pct)
                t_str = str(t)
                padding = max(1, 20 - len(t_str))
                audit += f"  {t_str}{' '*padding}{pct:5.1f}% ({cnt}/{total}) {bar}\n"
                for ex in examples.get(k, []):
                    audit += f"    e.g. {ex}\n"
        else:
            audit += "  (no lookalikes found)\n"

        # --- PROBABILITY ---
        audit += f"\n  PROBABILITY\n  {'='*48}\n"
        for t in sorted(probability, key=lambda k: -probability[k]):
            if probability[t] > 0:
                pct = 100 * probability[t]
                bar = self._format_bar(pct)
                audit += f"  P({t}) = {pct:.1f}% {bar}\n"

        # --- PER-LAYER DETAIL ---
        for layer_idx, chain in enumerate(self.layers):
            audit += f"\n  {'='*48}\n  LAYER {layer_idx}\n  {'='*48}\n"

            # Walk the chain to find the matched bucket, collecting skip reasons
            matched_bucket_idx = None
            routing_and_parts = []

            for b_idx, entry in enumerate(chain):
                cond = entry["condition"]
                if matched_bucket_idx is not None:
                    break
                if cond is None:
                    # ELSE bucket
                    matched_bucket_idx = b_idx
                else:
                    if all(self.apply_literal(X, lit) for lit in cond):
                        # X matches this condition — routed here
                        matched_bucket_idx = b_idx
                        # Add the passing literals to routing AND
                        for lit in cond:
                            routing_and_parts.append(self._format_literal_text(lit))
                    else:
                        # X fails this condition — find first failing literal and negate it
                        fail_lit = self._first_failing_literal(X, cond)
                        if fail_lit is not None:
                            negated = self._negate_literal(fail_lit)
                            routing_and_parts.append(self._format_literal_text(negated))

            # Emit Routing AND
            if matched_bucket_idx is not None:
                bucket = chain[matched_bucket_idx]
                n_members = len(bucket["members"])
                if bucket["condition"] is None and len(chain) > 1:
                    routing_desc = " AND ".join(routing_and_parts) if routing_and_parts else "(all conditions failed)"
                    audit += f"\n  Routing AND (default — ELSE bucket, {n_members} members):\n"
                    audit += f"    {routing_desc}\n"
                elif bucket["condition"] is None:
                    audit += f"\n  Routing AND (single bucket, {n_members} members):\n"
                    audit += f"    (only bucket — no routing needed)\n"
                else:
                    routing_desc = " AND ".join(routing_and_parts) if routing_and_parts else "(direct match)"
                    audit += f"\n  Routing AND (bucket {matched_bucket_idx + 1}/{len(chain)}, {n_members} members):\n"
                    audit += f"    {routing_desc}\n"

                # Compute local lookalikes with their conditions
                clause_bool = [self.apply_clause(X, c) for c in bucket["clauses"]]
                negated_set = {i for i in range(len(clause_bool)) if not clause_bool[i]}
                local_matches = []
                for l in bucket["lookalikes"]:
                    for condition in bucket["lookalikes"][l]:
                        if all(c_idx in negated_set for c_idx in condition):
                            local_matches.append((l, condition))

                audit += f"\n  Lookalike AND ({len(local_matches)} matches):\n"
                shown = 0
                max_show = 5
                for l, condition in local_matches:
                    if shown >= max_show:
                        remaining = len(local_matches) - max_show
                        audit += f"    ... and {remaining} more\n"
                        break
                    audit += self.get_plain_text_assertion(
                        condition, l,
                        bucket_clauses=bucket["clauses"],
                        bucket_members=bucket["members"]
                    ) + "\n"
                    shown += 1

        audit += f"\n  >> PREDICTION: {prediction}\n"
        audit += "### END AUDIT ###\n"
        return audit

    # ------------------------------------------------------------------
    # Synthetic audit (v5.4.7) — local-compute scan + optional cloud voice
    # ------------------------------------------------------------------

    def _synthetic_auroc(self, scores, labels):
        """Rank-based (Mann-Whitney) AUROC for binary labels in {0,1}. Pure
        Python, tie-averaged. Returns None if a class is absent."""
        pos = sum(labels)
        neg = len(labels) - pos
        if pos == 0 or neg == 0:
            return None
        order = sorted(range(len(scores)), key=lambda k: scores[k])
        rsum = 0.0
        j = 0
        rank = 1
        while j < len(order):
            k = j
            while k < len(order) and scores[order[k]] == scores[order[j]]:
                k += 1
            avg = (rank + rank + (k - j) - 1) / 2.0
            for t in range(j, k):
                if labels[order[t]] == 1:
                    rsum += avg
            rank += (k - j)
            j = k
        return (rsum - pos * (pos + 1) / 2.0) / (pos * neg)

    def _synthetic_summary(self, Xs):
        """Deterministic, 0-token scan over a batch of datapoints. Returns the
        summary dict that get_synthetic interprets. 100% local compute.
        Accepts a list of dicts, a single dict, or a pandas DataFrame."""
        if isinstance(Xs, dict):
            Xs = [Xs]
        else:
            df_rows = self._as_batch(Xs)        # DataFrame -> list of row-dicts
            if df_rows is not None:
                Xs = df_rows
        n = len(Xs)
        target_key = self.header[0]
        uniq = self._sorted_unique_targets()
        k = max(len(uniq), 1)
        uniform = 1.0 / k

        # --- batch scan: predictions + confidence + coverage (fast path) ---
        feats = [{kk: vv for kk, vv in X.items() if kk != target_key} for X in Xs]
        batch = self.get_batch_prediction(feats) if feats else []
        pred_counts = {}
        conf_sum = 0.0
        low_conf = 0
        covered = 0
        low_thresh = (uniform + 1.0) / 2.0
        for r in batch:
            pk = self._target_key(r["prediction"])
            pred_counts[pk] = pred_counts.get(pk, 0) + 1
            c = r["confidence"]
            conf_sum += c
            if c < low_thresh:
                low_conf += 1
            if c > uniform + 1e-9:        # not the flat prior => a bucket matched
                covered += 1

        # --- true base rate from training targets ---
        base = {}
        for t in self.targets:
            tk = self._target_key(t)
            base[tk] = base.get(tk, 0) + 1
        ntr = max(len(self.targets), 1)
        base_rate = {kk: round(100 * v / ntr, 1) for kk, v in base.items()}
        pred_rate = {kk: round(100 * v / max(n, 1), 1) for kk, v in pred_counts.items()}
        # calibration gap on the training-majority class
        maj = max(base, key=base.get) if base else None
        calib_gap = None
        if maj is not None:
            calib_gap = round(pred_rate.get(maj, 0.0) - base_rate.get(maj, 0.0), 1)

        # --- feature MI (already computed at train time) + noise detector ---
        if not self._feature_mi:
            try:
                self._precompute_feature_mi()
            except Exception:
                self._feature_mi = {}
        # target entropy H(target) in bits — to normalise MI (base-rate robust)
        h_target = 0.0
        for v in base.values():
            p = v / ntr
            if p > 0:
                h_target -= p * math.log2(p)
        # A feature with NO real link to the target still shows MI > 0 on finite
        # data, purely because empty (feature, class) cells never appear. We
        # separate real signal from this finite-sample illusion two ways:
        #   - effect size:   debiased MI (Miller-Madow) / H(target), n-robust
        #   - significance:  z-score of raw MI vs its analytic null. Under
        #                    independence 2n.MI.ln2 ~ chi2 with df=(bins-1)(k-1),
        #                    so null mean = df/(2n.ln2), null var = 2.df/(2n.ln2)^2.
        # is_noise fires when NO feature clears ~3 sigma (significant by chance,
        # max taken over all features). This is the honest is_noise flag.
        k_t = max(len(base), 1)
        ln2 = math.log(2)
        debiased = {}
        max_z = 0.0
        for c, v in self._feature_mi.items():
            bins = self._feature_bins.get(c, 2)
            df = max((bins - 1) * (k_t - 1), 1)
            if ntr > 0:
                null_mean = df / (2 * ntr * ln2)
                null_std = math.sqrt(2 * df) / (2 * ntr * ln2)
            else:
                null_mean = null_std = 0.0
            debiased[c] = max(v - null_mean, 0.0)
            if null_std > 1e-12:
                max_z = max(max_z, (v - null_mean) / null_std)
        mi_named = sorted(
            ((self.header[c], debiased.get(c, 0.0)) for c in self._feature_mi),
            key=lambda x: -x[1],
        )
        max_mi = mi_named[0][1] if mi_named else 0.0
        norm_mi = (max_mi / h_target) if h_target > 1e-9 else 0.0
        if max_z < 3.0:                       # not significant -> looks like chance
            strength, noise = "none", True
        elif norm_mi < 0.08:
            strength, noise = "weak", False
        elif norm_mi < 0.20:
            strength, noise = "moderate", False
        else:
            strength, noise = "strong", False

        # --- held-out metrics IF the caller passed labels (no retrain) ---
        labeled = [(X, X[target_key]) for X in Xs if target_key in X]
        acc = auroc = None
        n_labeled = len(labeled)
        if n_labeled:
            correct = 0
            scores = []
            bin_labels = []
            pos_class = uniq[-1] if uniq else None     # 1 > 0, True > False
            pos_key = self._target_key(pos_class)
            for X, y in labeled:
                Xf = {kk: vv for kk, vv in X.items() if kk != target_key}
                prob = self.get_probability(Xf)
                pred = self.get_prediction(Xf)
                if self._target_key(pred) == self._target_key(y):
                    correct += 1
                if k == 2:
                    scores.append(prob.get(pos_class, prob.get(pos_key, 0.0)))
                    bin_labels.append(1 if self._target_key(y) == pos_key else 0)
            acc = round(100 * correct / n_labeled, 1)
            if k == 2:
                a = self._synthetic_auroc(scores, bin_labels)
                auroc = round(a, 3) if a is not None else None

        return {
            "n_points": n,
            "n_classes": k,
            "task": "binary" if k == 2 else ("multiclass" if k > 2 else "single"),
            "n_layers": len(self.layers),
            "prediction_distribution_pct": pred_rate,
            "true_base_rate_pct": base_rate,
            "calibration_gap_pts": calib_gap,
            "mean_confidence": round(conf_sum / n, 3) if n else None,
            "low_confidence_rate_pct": round(100 * low_conf / n, 1) if n else None,
            "coverage_rate_pct": round(100 * covered / n, 1) if n else None,
            "top_features_mi": [(nm, round(v, 4)) for nm, v in mi_named[:6]],
            "target_entropy_bits": round(h_target, 3),
            "signal_strength": strength,
            "is_noise": noise,
            "n_labeled": n_labeled,
            "holdout_accuracy_pct": acc,
            "holdout_auroc": auroc,
        }

    def get_synthetic(self, Xs, interpret=True):
        """Synthetic audit at dataset scale (v5.4.7).

        Snake scans a *batch* of datapoints locally (0 tokens, sub-ms each),
        aggregates deterministic diagnostics — prediction spread, calibration
        vs the true base rate, feature mutual-information, a label-free noise
        detector, and held-out accuracy/AUROC when labels are supplied — then
        (optionally) makes ONE cloud call via the monceai SDK to narrate the
        finding as a tiny scientific experiment: hypothesis / experiment /
        result, explained plainly.

        Local compute is always available (interpret=False returns just the
        deterministic summary). The cloud voice is the only part that needs
        monceai — imported lazily so algorithmeai stays zero-dependency.

        Returns a dict: {"summary": <deterministic dict>, and when
        interpret=True also "hypothese"/"experience"/"resultat"}.
        """
        summary = self._synthetic_summary(Xs)
        result = {"summary": summary}
        if not interpret:
            return result

        try:
            from monceai import Json
        except ImportError:
            raise RuntimeError(
                "get_synthetic(interpret=True) needs the monceai SDK for the "
                "cloud narration step:\n"
                "    pip install git+https://github.com/Monce-AI/monceai-sdk.git\n"
                "For pure-local compute (0 tokens, no cloud), call "
                "get_synthetic(Xs, interpret=False) and read result['summary']."
            )

        prompt = (
            "You are explaining a machine-learning result to a smart 13-year-old, "
            "as if it were a small science experiment. Be warm and concrete, no jargon. "
            "Below is the deterministic output of a SAT-based classifier (Snake) that "
            "scanned a batch of data points. Return STRICT JSON with exactly three keys:\n"
            '  "hypothese"  — one sentence: what we bet the data could tell us.\n'
            '  "experience" — one sentence: what Snake actually did to test it '
            "(mention how many points, and that it costs no AI tokens).\n"
            '  "resultat"   — one sentence: what we found, using the real numbers, '
            "and whether the bet held up.\n"
            "Ground every claim in the numbers. If is_noise is true, say honestly the "
            "data looks like random luck. If calibration_gap_pts is large, mention the "
            "model is a bit over-optimistic. Never invent a number that is not below.\n\n"
            "DATA:\n" + json.dumps(summary, ensure_ascii=False)
        )
        narration = dict(Json(prompt))
        for key in ("hypothese", "experience", "resultat"):
            result[key] = narration.get(key, "")
        return result

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_json(self, fout="snakeclassifier.json", stripped=False):
        """Serialize the model to JSON.

        stripped=True drops the training `population` (typically ~95% of the
        bytes) and writes only what inference needs: layers, targets, header,
        datatypes, config. A stripped model serves the hot path
        (prediction / probability / lookalikes / candle / regression) at a
        fraction of the RAM, so many workers can each hold a full copy — the
        v5.4.8 worker-pool lever. Population-dependent methods (get_audit,
        get_augmented, get_lookalikes_labeled's sample labels) raise a clear
        error on a stripped model. Save a full model too if you need those.
        """
        snake_classifier = {
            "version": "5.5.1",
            "stripped": bool(stripped),
            "population": [] if stripped else self.population,
            "header": self.header,
            "target": self.target,
            "targets": self.targets,
            "datatypes": self.datatypes,
            "config": {
                "n_layers": self.n_layers,
                "bucket": self.bucket,
                "noise": self.noise,
                "vocal": self.vocal,
                "workers": self.workers,
                "oppose_profile": getattr(self, 'oppose_profile', 'auto'),
                "lookahead": getattr(self, 'lookahead', 5),
                # v5.5.0 domain extension rides INSIDE config — the nine
                # top-level keys stay byte-identical to v5.4.8. Old loaders
                # ignore unknown config entries; the derived columns themselves
                # already serialize as ordinary header/datatypes/population.
                "expand": getattr(self, 'expand', 'auto'),
                "expansions": self._expansions_to_json(),
            },
            "layers": self.layers,
            "log": self.log
        }
        with open(fout, "w") as f:
            f.write(json.dumps(snake_classifier, indent=2))
        kind = "stripped" if stripped else "full"
        self.qprint(f"Safely saved to {fout} ({kind})")

    def from_json(self, filepath="snakeclassifier.json"):
        with open(filepath, "r") as f:
            loaded_module = json.load(f)
        self.population = loaded_module.get("population", [])
        # v5.4.8: a stripped model carries no population. Flag it so the
        # population-dependent methods can raise a clear error rather than
        # an opaque IndexError, and so the parallel pool knows it's cheap to fork.
        self._stripped = bool(loaded_module.get("stripped", False)) or not self.population
        self.header = loaded_module["header"]
        self.target = loaded_module["target"]
        self.targets = loaded_module["targets"]
        self.datatypes = loaded_module["datatypes"]

        # Backwards compat: flat v0.1 format
        if "clauses" in loaded_module and "layers" not in loaded_module:
            self._load_flat(loaded_module)
        elif "layers" in loaded_module:
            self._load_bucketed(loaded_module)

        if "config" in loaded_module:
            cfg = loaded_module["config"]
            self.n_layers = cfg.get("n_layers", self.n_layers)
            self.bucket = cfg.get("bucket", self.bucket)
            self.noise = cfg.get("noise", self.noise)
            self.vocal = cfg.get("vocal", self.vocal)
            self.workers = cfg.get("workers", 1)
            self.oppose_profile = cfg.get("oppose_profile", "auto")
            self.lookahead = cfg.get("lookahead", 5)
            self.expand = cfg.get("expand", "auto")
            # v5.5.0: restore fitted expansion records. The derived columns are
            # already in header/datatypes (serialized as ordinary columns); these
            # records let _normalize_features regrow them for new inference rows.
            self.expansions = cfg.get("expansions", []) or []
            self._rebuild_derived_names()
        elif "n_layers" in loaded_module:
            self.n_layers = loaded_module["n_layers"]
            self.vocal = loaded_module.get("vocal", self.vocal)

        # Initialize oppose profile for loaded models
        if hasattr(self, 'oppose_profile') and self.oppose_profile in _VALID_PROFILES and self.oppose_profile != "auto":
            self._active_oppose = getattr(self, f"_oppose_{self.oppose_profile}")
            if self.oppose_profile == "scientific":
                self._precompute_col_stats()
        else:
            # Old models or auto: default to original oppose
            self._active_oppose = self.oppose

        self.log = loaded_module.get("log", self.log)
        self.qprint(f"# Algorithme.ai : Successful load from {filepath}")

    def _load_flat(self, loaded_module):
        """Load v0.1 flat format (clauses + lookalikes at top level)."""
        self.clauses = loaded_module["clauses"]
        self.lookalikes = loaded_module["lookalikes"]
        # Wrap the flat model into a single ELSE bucket in a single layer
        members = list(range(len(self.population)))
        self.layers = [[{
            "condition": None,
            "members": members,
            "clauses": self.clauses,
            "lookalikes": self.lookalikes
        }]]

    def _load_bucketed(self, loaded_module):
        """Load v4.3.3 bucketed format."""
        self.layers = loaded_module["layers"]
        self.clauses = []
        self.lookalikes = {str(l): [] for l in range(len(self.population))}

    # ------------------------------------------------------------------
    # Validation (adapted for bucketed layers)
    # ------------------------------------------------------------------

    """
    Validation process of the lookalikes table on the premise of a targeted sample
    """
    def make_validation(self, Xs, pruning_coef=0.5):
        new_n_layers = max(1, int(len(self.layers) * pruning_coef))
        self.qprint(f"#")
        self.qprint(f"# ============================================================")
        self.qprint(f"#   VALIDATION START")
        self.qprint(f"# ============================================================")
        self.qprint(f"#   Validation samples: {len(Xs)}")
        self.qprint(f"#   Pruning:            {len(self.layers)} layers -> {new_n_layers} (coef={pruning_coef})")
        self.qprint(f"#   Complexity:         O({len(self.layers)} layers * {len(Xs)} samples)")
        self.qprint(f"# ============================================================")

        # Score each layer by accuracy on validation set
        layer_scores = []
        t_val_start = time()
        for layer_idx, chain in enumerate(self.layers):
            t_layer = time()
            correct = 0
            total = 0
            n_buckets = len(chain)
            n_clauses = sum(len(e["clauses"]) for e in chain)
            for X in Xs:
                if self.target not in X:
                    continue
                target = X[self.target]
                bucket = traverse_chain(chain, X, self.apply_literal)
                if bucket is None:
                    continue
                clause_bool = [self.apply_clause(X, c) for c in bucket["clauses"]]
                negated = {i for i in range(len(clause_bool)) if not clause_bool[i]}
                votes = []
                for l in bucket["lookalikes"]:
                    for condition in bucket["lookalikes"][l]:
                        if all(c_idx in negated for c_idx in condition):
                            global_idx = bucket["members"][int(l)]
                            votes.append(self.targets[global_idx])
                if votes:
                    counts = {}
                    key_to_vote = {}
                    for v in votes:
                        k = self._target_key(v)
                        counts[k] = counts.get(k, 0) + 1
                        key_to_vote[k] = v
                    best_key = max(counts, key=counts.get)
                    pred = key_to_vote[best_key]
                    if pred == target:
                        correct += 1
                total += 1
            accuracy = correct / total if total > 0 else 0
            layer_scores.append((layer_idx, accuracy))

            layer_time = time() - t_layer
            layers_done = layer_idx + 1
            layers_left = len(self.layers) - layers_done
            elapsed_val = time() - t_val_start
            avg_per = elapsed_val / layers_done
            eta_val = avg_per * layers_left
            self.qprint(f"#   [val] Layer {layer_idx}: accuracy={accuracy:.3f} ({correct}/{total}), {n_buckets} buckets, {n_clauses} clauses, {layer_time:.2f}s — ETA {eta_val:.2f}s")

        # Keep top layers
        layer_scores.sort(key=lambda x: -x[1])
        kept_indices = sorted([ls[0] for ls in layer_scores[:new_n_layers]])
        dropped_indices = [ls[0] for ls in layer_scores[new_n_layers:]]

        total_val_time = time() - t_val_start
        self.qprint(f"#")
        self.qprint(f"#   Scoring complete in {total_val_time:.2f}s")
        self.qprint(f"#   Best layers:  {[(idx, f'{acc:.3f}') for idx, acc in layer_scores[:new_n_layers]]}")
        self.qprint(f"#   Dropped:      {dropped_indices}")
        self.qprint(f"#   Kept:         {kept_indices}")

        self.layers = [self.layers[i] for i in kept_indices]
        self.n_layers = len(self.layers)
        self.qprint(f"# ============================================================")
        self.qprint(f"#   VALIDATION COMPLETE — {self.n_layers} layers retained")
        self.qprint(f"# ============================================================")


# ---------------------------------------------------------------------------
# Module-level worker functions (must be top-level for multiprocessing)
# ---------------------------------------------------------------------------

# Per-worker shared data (set once via Pool initializer, reused across jobs)
_worker_data = {}

# v5.4.8 — model shared with inference workers via fork-inherited COW.
# _parallel_infer sets this global before forking the pool; each forked child
# reads it directly from its inherited address space (no pickling of the model).
_infer_model = None


def _infer_chunk_worker(job):
    """Score a contiguous chunk of datapoints with a single-dict Snake method.
    Returns (start_index, [results...]) so the parent can splice in order."""
    method_name, start, chunk = job
    method = getattr(_infer_model, method_name)
    return start, [method(X) for X in chunk]


def _init_worker(population, targets, header, datatypes, oppose_profile=None, col_stats=None, feature_mi=None, lookahead=5):
    """Pool initializer — sends large data once per worker process."""
    global _worker_data
    _worker_data = {
        "population": population,
        "targets": targets,
        "header": header,
        "datatypes": datatypes,
        "oppose_profile": oppose_profile,
        "col_stats": col_stats or {},
        "feature_mi": feature_mi or {},
        "lookahead": lookahead,
    }


def _setup_worker_logger(s, seed):
    """Attach a silent buffer-only logger to a worker Snake instance."""
    global _snake_instance_counter
    _snake_instance_counter += 1
    s._logger = logging.getLogger(f"snake.worker.{_snake_instance_counter}.{seed}")
    s._logger.setLevel(logging.DEBUG)
    s._logger.propagate = False
    s._logger.handlers.clear()
    s._buffer_handler = _StringBufferHandler()
    s._buffer_handler.setLevel(logging.DEBUG)
    s._buffer_handler.setFormatter(logging.Formatter("%(message)s"))
    s._logger.addHandler(s._buffer_handler)


def _build_layer_worker(args):
    """Build one Snake layer in a separate process."""
    bucket, noise, seed, layer_idx, n_layers = args
    import random as _rng
    _rng.seed(seed)

    # Lightweight Snake instance — just enough for construct_layer() methods
    s = Snake.__new__(Snake)
    s.population = _worker_data["population"]
    s.targets = _worker_data["targets"]
    s.header = _worker_data["header"]
    s.datatypes = _worker_data["datatypes"]
    s.bucket = bucket
    s.noise = noise
    s.n_layers = n_layers
    s.layers = []
    s.clauses = []
    s.lookalikes = {}
    s.vocal = False
    s.progress_file = None
    s.workers = 1
    s.oppose_profile = _worker_data.get("oppose_profile", "auto")
    s._col_stats = _worker_data.get("col_stats", {})
    s._feature_mi = _worker_data.get("feature_mi", {})
    s.lookahead = _worker_data.get("lookahead", 5)
    s._t0 = time()
    s._avg_per_layer = 0
    s._current_layer = layer_idx

    _setup_worker_logger(s, seed)

    # Initialize _active_oppose for the worker
    if s.oppose_profile in _VALID_PROFILES and s.oppose_profile != "auto":
        s._active_oppose = getattr(s, f"_oppose_{s.oppose_profile}")
    else:
        s._active_oppose = s.oppose

    s.construct_layer()
    return s.layers[0]
