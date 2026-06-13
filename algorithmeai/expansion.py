"""
Snake v5.5.0 — Automatic Domain Extension (expansion layer).

The constructor auto-recognizes columns with expansion potential and grows
DERIVED NUMERIC columns from them; Shannon MI (already in snake.py) then filters
the wide candidate set down to the informative few. Every derived column:

  * is an ordinary "N" column appended to self.header / self.datatypes /
    self.population — so the EXISTING SAT / oppose / MI / audit machinery runs
    on it unchanged (zero new literal code);
  * carries a human-readable name "{source}::{op}({arg})" that flows verbatim
    through _format_literal_text into get_audit() (explainability for free);
  * is fitted on train and deterministically re-appliable to one raw datapoint
    at inference (so get_prediction/regression/audit/candle see it transparently);
  * round-trips through to_json/from_json inside an additive "expansions" block —
    the nine v5.4.8 top-level keys are UNCHANGED, old models default to [].

v1 scope: the TOKENSET family only (dense NLP / token-set columns like a
gene-symbol blob). has(token) binary membership + token_count. This is the
headline that fixes the LEV/JAC-on-blobs cost and lifts R² to TF-IDF-Snake
levels natively. Other families (numeric crypto/stat, categorical, regex, ...)
plug into the same spine in later rounds.

Pure Python. Zero dependencies. The whole file is a mixin on the Snake class.
"""

import math
import re
from random import sample


def _is_missing(v):
    """Local mirror of snake._is_missing (kept byte-identical in behavior to avoid
    a circular import): None or any NaN/NA/NaT sentinel (the only value != itself).
    An empty string is NOT missing — it is a legitimate categorical value."""
    if v is None:
        return True
    try:
        return bool(v != v)
    except Exception:
        return False


# --- module-level tokenizer (shared train + inference, must be deterministic) ---

_DELIMS = (" ", ",", ";", "|", "/")


def _pick_delim(values):
    """Choose the delimiter giving the highest mean token count over a sample.
    Gene blobs are space-separated; categorical pipes use '|', etc."""
    best, best_score = " ", -1.0
    for d in _DELIMS:
        total = 0
        for v in values:
            total += len(v.split(d))
        score = total / max(len(values), 1)
        if score > best_score:
            best, best_score = d, score
    return best


def _tokenize(value, delim):
    """Deterministic tokenization: split on delim, strip, drop empties.
    Returns a list (order irrelevant downstream; membership is what matters)."""
    if value is None:
        return []
    s = value if isinstance(value, str) else str(value)
    out = []
    for tok in s.split(delim):
        tok = tok.strip()
        if tok:
            out.append(tok)
    return out


def _safe_token(tok):
    """A token may contain characters that would break the ::name grammar.
    The grammar splits source on the FIRST '::' and reads arg between the
    outermost parens, so the only hard constraint is no ')' that closes early
    and no '::'. Tokens that violate it are skipped at fit time (rare for
    biology symbols / SKUs). Returns True if the token is safe to name."""
    return "::" not in tok and ")" not in tok and "(" not in tok


class ExpansionMixin:
    """Domain-extension methods mixed into Snake. Assumes the host provides:
    self.header, self.datatypes, self.population, self.targets, self.target,
    self._target_key, self.qprint, and (post-fit) self._feature_mi."""

    # ------------------------------------------------------------------
    # Defaults / knobs (overridable via constructor kwargs stashed on self)
    # ------------------------------------------------------------------
    # Defaults mirror snake_showcase.py's TfidfVectorizer(token_pattern=r'\S+',
    # max_features=20, min_df=2) EXACTLY — so native expansion reproduces the
    # article's winning recipe cell-for-cell (no gaslighting a worse TF-IDF).
    EXPAND_MIN_AVG_TOKENS = 4.0      # dense-column trigger: it's a multi-token field
    EXPAND_MIN_RECURRENT = 8         # >= this many df>=2 tokens (a recurrent core exists)
    EXPAND_TOP_K = 100               # FLOOD width: candidates fitted per source (= sklearn
                                     #   max_features at fit). We over-provide on purpose.
    EXPAND_MI_KEEP = 10              # AGGRESSIVE CUT: keep this many top-MI derived cols per
                                     #   source after _apply_mi_gate. Won the bench (0.217 vs
                                     #   tfidf20 0.203). Less is more: top10 > top20 > top40.
    EXPAND_DF_MIN = 2                # = sklearn min_df (token in >= this many documents)
    EXPAND_DF_MAX_FRAC = 1.0         # = sklearn max_df=1.0 (no upper doc-freq filter)
    EXPAND_EMIT_COUNT = False        # token_count column — OFF by default to match showcase
    EXPAND_SAMPLE = 500              # rows sampled for delim pick
    EXPAND_MIN_DISTINCT = 8          # numeric col is "continuous" (Gaussian family) if it has
                                     #   >= this many distinct values (skips binary/low-card flags)
    EXPAND_MAX_CARD = 100            # v5.5.1 CATEGORICAL trigger: a column (numeric OR text) with
                                     #   2 <= n_unique <= this is categorical -> flat one-hot col==v
                                     #   booleans, one per value, MI-filtered. Above this a numeric
                                     #   col is continuous (GAUSSIAN); a text col with a recurrent
                                     #   token core is TOKENSET. The two families are exclusive.
    EXPAND_GAUSS_MIN_FIT = 0.95      # v5.5.1: GAUSSIAN only fires on a high-card numeric column
                                     #   whose empirical CDF agrees with the fitted normal to within
                                     #   this (1 - max|F_emp - Phi(z)|, a KS-style agreement). A
                                     #   non-normal column (uniform n, skewed log_n) scores < this
                                     #   and gets NO Gaussian KPIs — they'd be position-noise. Raw
                                     #   column always stays. Calibrated: N(0,1)->0.985, U(0,1)->0.940.
    EXPAND_FAMILIES = ("TOKENSET", "GAUSSIAN", "CATEGORICAL")
                                     # which families may fire. Override per-instance to ablate
                                     #   (e.g. ("TOKENSET",) reproduces the v5.5.0 family set exactly).
    EXPAND_CAT_MIN_ROWS = 30         # v5.5.1: CATEGORICAL needs at least this many rows before
                                     #   "recurrence" and MI are statistically meaningful. Below it,
                                     #   one-hotting a tiny toy/fixture column manufactures features
                                     #   from coincidence (incl. tuple-padding artifacts). TOKENSET
                                     #   self-gates similarly via its 8-recurrent-token floor.

    # ------------------------------------------------------------------
    # Detection + fit (train side)
    # ------------------------------------------------------------------
    def _detect_expansions(self):
        """Scan T columns; fit a TOKENSET family on dense token-set columns.
        Populates self.expansions (list of fit records) and materializes the
        derived columns into header/datatypes/population. No-op when expand is
        off or no column qualifies. Must run AFTER datatype detection and
        BEFORE _init_oppose_profile so MI sees the derived columns."""
        if not getattr(self, "expansions", None):
            self.expansions = []
        mode = getattr(self, "expand", "auto")
        if mode is False or mode == "off":
            return

        forced = mode if isinstance(mode, dict) else None
        n = len(self.population)
        if n < 2:
            return

        # v5.5.1 CATEGORICAL: any column (numeric OR text) with low cardinality is
        # categorical, not continuous/token. Detected first so a low-card numeric is
        # claimed here and never reaches the GAUSSIAN branch, and a low-card text
        # column is claimed here and never reaches TOKENSET. Cardinality is the router.
        claimed = set()  # source header names already handled by CATEGORICAL
        if "CATEGORICAL" in self.EXPAND_FAMILIES and n >= self.EXPAND_CAT_MIN_ROWS:
            for col in range(1, len(self.header)):
                if self.datatypes[col] not in ("T", "N"):
                    continue
                h = self.header[col]
                if forced is not None and h not in forced:
                    continue
                # Value frequencies over non-missing cells (NaN/None never become a
                # candidate value — they are filled, not encoded). Key by a stable
                # normalized form so 1, 1.0 and "1" count as the same value.
                freq = {}
                for row in self.population:
                    v = row.get(h)
                    if _is_missing(v):
                        continue
                    k = self._cat_key(v)
                    if k not in freq:
                        freq[k] = [0, v]   # count, representative raw value
                    freq[k][0] += 1
                n_present = sum(c for c, _ in freq.values())
                if n_present < 2:
                    continue
                nu = len(freq)
                # FLOOD-FILTER PARITY with TOKENSET (Charles): a value is a candidate
                # only if it RECURS (count >= EXPAND_DF_MIN, == sklearn min_df). A
                # column whose values are all singletons (e.g. a continuous numeric
                # read as low-card on a tiny sample, or a free-id field) yields ZERO
                # recurrent candidates -> the column is NOT claimed and falls through
                # to GAUSSIAN/nothing. This is the categorical analogue of df>=2:
                # one-hotting singletons memorizes row ids, never generalizes.
                recurrent = [(c, rep) for (c, rep) in freq.values()
                             if c >= self.EXPAND_DF_MIN]
                qualifies = (2 <= nu <= self.EXPAND_MAX_CARD) and len(recurrent) >= 1
                if forced is not None:
                    # Power-user override still respects the recurrence floor — a
                    # forced one-hot on pure singletons would be guaranteed noise.
                    qualifies = len(recurrent) >= 1
                if not qualifies:
                    continue
                # Candidate values: recurrent ones, ordered deterministically by
                # (descending count, then stable value string) for reproducibility.
                cand_vals = [rep for _, rep in sorted(
                    recurrent, key=lambda cr: (-cr[0], str(cr[1])))]
                self._fit_categorical(col, h, cand_vals)
                claimed.add(h)

        text_cols = ([i for i in range(1, len(self.header))
                      if self.datatypes[i] == "T" and self.header[i] not in claimed]
                     if "TOKENSET" in self.EXPAND_FAMILIES else [])
        for col in text_cols:
            h = self.header[col]
            if forced is not None and h not in forced:
                continue
            values = [str(row.get(h, "")) for row in self.population]
            samp = values if n <= self.EXPAND_SAMPLE else sample(values, self.EXPAND_SAMPLE)
            delim = _pick_delim(samp)

            # detection stats. Genomic blobs are long-tailed: many tokens/row,
            # a few recurrent drivers (TP53, KRAS, ...) + a long private tail.
            # We detect "multi-token field WITH a recurrent core" — the df>=2
            # filter at fit time discards the private tail, so it never matters
            # how big the singleton tail is.
            vocab_df = {}            # token -> document frequency (rows containing it)
            vocab_tf = {}            # token -> total term frequency (sklearn ranking key)
            total_tokens = 0
            for v in values:
                toks = _tokenize(v, delim)
                seen = set()
                for t in toks:
                    vocab_tf[t] = vocab_tf.get(t, 0) + 1
                    if t not in seen:
                        seen.add(t)
                        vocab_df[t] = vocab_df.get(t, 0) + 1
                total_tokens += len(seen)
            vocab_size = len(vocab_df)
            if vocab_size == 0:
                continue
            avg_tokens = total_tokens / n
            recurrent = sum(1 for t, df in vocab_df.items() if df >= self.EXPAND_DF_MIN)

            qualifies = (
                avg_tokens >= self.EXPAND_MIN_AVG_TOKENS
                and recurrent >= self.EXPAND_MIN_RECURRENT
            )
            if forced is not None:
                qualifies = True  # power-user override forces the family

            if not qualifies:
                continue

            self._fit_tokenset(col, h, delim, vocab_df, vocab_tf, n)

        # Numeric family — Gaussian-position KPIs on RAW continuous N columns.
        # Scanned BEFORE materialization, so derived TF-IDF cols (also "N") are
        # not yet in the header and can never be recursively expanded.
        numeric_cols = ([i for i in range(1, len(self.header))
                         if self.datatypes[i] == "N" and self.header[i] not in claimed]
                        if "GAUSSIAN" in self.EXPAND_FAMILIES else [])
        for col in numeric_cols:
            h = self.header[col]
            if forced is not None and h not in forced:
                continue
            vals = []
            for row in self.population:
                v = row.get(h)
                if isinstance(v, (int, float)) and v == v:
                    vals.append(float(v))
            if len(vals) < 2:
                continue
            distinct = len(set(vals))
            # High-card by construction (low-card was claimed by CATEGORICAL above),
            # but keep the floor as a guard. GAUSSIAN only earns its place if the
            # column is actually ~normal — else its position KPIs are pure noise.
            fit = self._gaussianity(vals)
            qualifies = (distinct >= self.EXPAND_MIN_DISTINCT
                         and fit >= self.EXPAND_GAUSS_MIN_FIT)
            if forced is not None:
                qualifies = True
            if not qualifies:
                self.qprint(f"# Expansion[GAUSSIAN] {h}: SKIP "
                            f"(gaussianity {fit:.3f} < {self.EXPAND_GAUSS_MIN_FIT}, "
                            f"distinct={distinct}) — raw column kept", level=2)
                continue
            self._fit_gaussian(col, h, vals)

        # Materialize all fitted derived columns into the population once.
        if self.expansions:
            self._materialize_population()

    def _fit_tokenset(self, col, h, delim, vocab_df, vocab_tf, n):
        """Register continuous TF-IDF derived columns, mirroring sklearn's
        TfidfVectorizer(token_pattern=r'\\S+', max_features=K, min_df=2) EXACTLY
        so native expansion reproduces snake_showcase.py cell-for-cell.

        sklearn semantics replicated:
          * vocabulary = tokens with document-frequency >= min_df (and <= max_df)
          * max_features = keep the top-K by TOTAL TERM FREQUENCY across the
            corpus (sklearn's tie-break is term then alphabetical)
          * idf(t) = ln((1+n)/(1+df)) + 1   (smooth_idf=True default)
          * weight = tf_in_doc * idf, then L2-normalize the row vector (norm='l2')

        Selection is TARGET-BLIND by construction (term frequency, not target
        correlation) — which also dodges the 'train MI lies' overfit trap, since
        Snake is perfect-fit on train via lookalikes."""
        df_max = self.EXPAND_DF_MAX_FRAC * n
        cand = [t for t, df in vocab_df.items()
                if self.EXPAND_DF_MIN <= df <= df_max and _safe_token(t)]
        if not cand:
            return
        # sklearn max_features: rank by total term frequency, desc; alpha tie-break.
        cand.sort(key=lambda t: (-vocab_tf[t], t))
        keep = cand[: self.EXPAND_TOP_K]

        derived = []
        if self.EXPAND_EMIT_COUNT:
            derived.append({"name": f"{h}::token_count", "op": "CNT", "arg": None, "dt": "N"})
        for t in keep:
            df = vocab_df[t]
            idf = math.log((1.0 + n) / (1.0 + df)) + 1.0  # sklearn smooth idf
            derived.append({"name": f"{h}::tfidf({t})", "op": "TFIDF",
                            "arg": t, "dt": "N", "idf": idf})

        self.expansions.append({
            "source": h,
            "family": "TOKENSET",
            "delim": delim,
            "norm": "l2",
            "derived": derived,
        })
        self.qprint(f"# Expansion[TOKENSET] {h}: vocab={len(vocab_df)} -> "
                    f"{len(keep)} tfidf() (max_features={self.EXPAND_TOP_K}, "
                    f"min_df={self.EXPAND_DF_MIN}, delim={delim!r}, l2-norm)")

    @staticmethod
    def _gaussianity(vals):
        """KS-style agreement between the column's empirical CDF and the fitted
        normal: 1 - max_i |F_emp(x_i) - Phi((x_i-mu)/sigma)|, in [0, 1]. 1.0 is a
        perfect normal fit; ~0.94 is uniform, lower is heavier-tailed/skewed.
        Pure math, O(n log n) for the sort. Constant column -> 0.0 (no shape)."""
        n = len(vals)
        if n < 2:
            return 0.0
        mu = sum(vals) / n
        var = sum((v - mu) ** 2 for v in vals) / n
        sigma = math.sqrt(var)
        if sigma <= 0.0:
            return 0.0
        sv = sorted(vals)
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        d = 0.0
        for i, v in enumerate(sv):
            f_emp = (i + 1) / n
            f_th = 0.5 * (1.0 + math.erf((v - mu) / sigma * inv_sqrt2))
            diff = f_emp - f_th
            if diff < 0:
                diff = -diff
            if diff > d:
                d = diff
        return 1.0 - d

    def _fit_categorical(self, col, h, cand_vals):
        """Register flat one-hot derived columns for a low-cardinality source
        (numeric OR text). For each candidate value v, emit a 0/1 column "{h}==v"
        that is 1.0 iff the row's value equals v, else 0.0.

        This is the v5.5.1 family. The point (Charles): a low-card column like a
        residue, a status code, or a small count is CATEGORICAL, not continuous —
        routing it through GAUSSIAN buries its signal in position KPIs that can't
        express membership. A flat one-hot turns each value into an exact-match
        bit; Snake's numeric-midpoint literal splits a 0/1 column at 0.5 — an EXACT
        discriminator with no stochastic threshold to miss. MI then keeps only the
        few values that carry signal (the gate is value-agnostic: mod2==0 survives,
        last_digit==7 doesn't). The RAW source column is always kept underneath.

        `cand_vals` arrives already filtered to RECURRENT values (count >=
        EXPAND_DF_MIN, the flood-filter parity with TOKENSET's min_df) and ordered
        deterministically by (count desc, value). We flood them as candidates
        (capped at EXPAND_TOP_K) and let _apply_mi_gate cut to the global top
        EXPAND_MI_KEEP per family by MI, exactly like the other families."""
        derived = []
        for v in cand_vals[: self.EXPAND_TOP_K]:
            label = self._categorical_label(v)
            if label is None:
                continue  # value can't be safely named -> skip (rare)
            derived.append({"name": f"{h}=={label}", "op": "EQ",
                            "arg": v, "dt": "N"})
        if not derived:
            return
        self.expansions.append({
            "source": h,
            "family": "CATEGORICAL",
            "derived": derived,
        })
        self.qprint(f"# Expansion[CATEGORICAL] {h}: {len(cand_vals)} recurrent "
                    f"values -> {len(derived)} one-hot col==v (MI-filtered to "
                    f"global top-{self.EXPAND_MI_KEEP}/family, raw col kept)")

    @staticmethod
    def _categorical_label(v):
        """Human-readable, name-grammar-safe label for a categorical value. The
        derived-name grammar reserves '::' and parentheses; a value containing
        them can't be named so we skip it. Integer-valued floats render without
        the trailing '.0' (population stores numerics as float, so mod2==0 reads
        cleaner than mod2==0.0); other numbers and strings render as-is — the
        '==' in the name disambiguates the derived column from its source."""
        if isinstance(v, bool):
            s = str(v)
        elif isinstance(v, float):
            s = str(int(v)) if v.is_integer() else repr(v)
        elif isinstance(v, int):
            s = str(v)
        else:
            s = str(v)
        if "::" in s or "(" in s or ")" in s:
            return None
        return s

    def _fit_gaussian(self, col, h, vals):
        """Register Gaussian-position KPI columns for a continuous numeric source.

        Charles's idea: tell Snake not just a number's value but its POSITION in
        the training distribution, three ways — all pure `math`, zero deps:

          * {h}::gauss_z       z = (x-mu)/sigma            (standardized deviation)
          * {h}::gauss_density exp(-0.5 * z^2)             (similarity-to-typical:
                                                            1.0 at the mean, ->0 in
                                                            the tails. A Gaussian
                                                            kernel to the centre.)
          * {h}::gauss_cdf     Phi(z) = 0.5*(1+erf(z/√2))  (percentile position in
                                                            the fitted normal: where
                                                            this sample sits, 0..1)

        mu and sigma are frozen at fit (train) and ride in the JSON exactly like
        TF-IDF idf does. Aggressive MI then keeps whichever of the three earns its
        place; the RAW numeric column is always kept underneath (the invariant)."""
        n = len(vals)
        mu = sum(vals) / n
        var = sum((v - mu) ** 2 for v in vals) / n
        sigma = math.sqrt(var)
        if sigma <= 0.0:
            return  # constant column — no position information to add
        derived = [
            {"name": f"{h}::gauss_z", "op": "GZ", "arg": None, "dt": "N"},
            {"name": f"{h}::gauss_density", "op": "GD", "arg": None, "dt": "N"},
            {"name": f"{h}::gauss_cdf", "op": "GC", "arg": None, "dt": "N"},
        ]
        self.expansions.append({
            "source": h,
            "family": "GAUSSIAN",
            "mu": mu,
            "sigma": sigma,
            "derived": derived,
        })
        self.qprint(f"# Expansion[GAUSSIAN] {h}: mu={mu:.4g} sigma={sigma:.4g} "
                    f"-> gauss_z / gauss_density / gauss_cdf")

    def _target_bins_for_expansion(self, n_bins=8):
        """Bin the target into <= n_bins strata for the chi-square pre-rank.
        Classification: target_key itself. Regression: quantile bins."""
        # Numeric target -> quantile bins; else categorical key
        if self.datatypes and self.datatypes[0] in ("N", "I"):
            vals = []
            for t in self.targets:
                try:
                    vals.append(float(t))
                except (ValueError, TypeError):
                    vals.append(0.0)
            su = sorted(set(vals))
            nb = min(n_bins, len(su))
            if nb < 2:
                return [0] * len(vals)
            bounds = []
            for b in range(1, nb):
                idx = min(int(b * len(su) / nb), len(su) - 1)
                bounds.append(su[idx])
            out = []
            for v in vals:
                assigned = len(bounds)
                for bi, bnd in enumerate(bounds):
                    if v <= bnd:
                        assigned = bi
                        break
                out.append(assigned)
            return out
        return [self._target_key(t) for t in self.targets]

    # ------------------------------------------------------------------
    # Materialization (train) + reapply (inference)
    # ------------------------------------------------------------------
    def _materialize_population(self):
        """Append derived names/datatypes to header/datatypes (once) and write
        derived values into every population row."""
        existing = set(self.header)
        for rec in self.expansions:
            for d in rec["derived"]:
                if d["name"] in existing:
                    continue
                self.header.append(d["name"])
                self.datatypes.append("N")
                existing.add(d["name"])
        self._rebuild_derived_names()
        for row in self.population:
            self._apply_expansions_inplace(row)

    def _apply_mi_gate(self):
        """AGGRESSIVE MI cut: flood wide, keep only the top-MI few DERIVED columns.

        The v5.5.0 thesis, proven on the honest seed-42 20/80 drug split (all 6
        drugs, 64 layers): fit a WIDE candidate set (EXPAND_TOP_K=100/source), then
        keep the EXPAND_MI_KEEP (=10) highest-MI derived columns per source. That
        config scored R²=0.217 vs the paper's fixed-20 recipe 0.203 and raw 0.146 —
        and rescued the two drugs TF-IDF had HURT (elephantin 0.059→0.136,
        plx4720 0.140→0.200). Less is more: top10 > top20 > top40.

        THE INVARIANT (Charles's design call): this gate prunes ONLY derived
        columns. Every RAW source training column is ALWAYS kept — `survivors`
        starts as nothing, drop set is computed against `_derived_names` only in
        `_prune_derived_to`, so raw columns are untouchable by construction. This
        guarantees baseline + improvement, never regression: expansion can only
        ADD signal on top of the original information, never erase it. The only
        sub-raw scores in the bench were where noise derived cols crowded out raw
        signal — protecting raw makes that failure mode structurally impossible.

        MI is Snake's own native binned-histogram MI (_precompute_feature_mi,
        pure Python). Must run AFTER it; followed by a re-materialization so the
        TF-IDF L2-norm basis matches the surviving token set at train==inference."""
        if not getattr(self, "expansions", None):
            return
        name_to_idx = {self.header[i]: i for i in range(1, len(self.header))}

        # PER-FAMILY GLOBAL cut (v5.5.1): pool ALL derived candidates of a family
        # across every source, rank by MI ONCE, keep the global top EXPAND_MI_KEEP.
        # No per-source quota — a single rich source (e.g. subtype's 95 one-hots)
        # competes head-to-head with every other source's candidates and takes as
        # many of the family's 10 slots as MI says it earns. MI is the only judge;
        # it isn't applied twice. Each family gets its own budget so a wide
        # CATEGORICAL flood can't crowd out the tuned TOKENSET set, or vice versa.
        family_scored = {}   # family -> [(mi, name), ...] pooled across sources
        for rec in self.expansions:
            fam = rec.get("family", "TOKENSET")
            bucket = family_scored.setdefault(fam, [])
            for d in rec["derived"]:
                idx = name_to_idx.get(d["name"])
                if idx is not None:
                    bucket.append((self._feature_mi.get(idx, 0.0), d["name"]))

        survivors = set()
        report = []
        for fam, scored in family_scored.items():
            ranked = sorted(scored, key=lambda x: -x[0])
            keep = ranked[: self.EXPAND_MI_KEEP]
            survivors.update(nm for _, nm in keep)
            report.append((fam, len(scored), len(keep)))

        self._prune_derived_to(survivors)
        for fam, before, after in report:
            self.qprint(f"# MI-gate [{fam}]: pooled {before} candidates -> "
                        f"global top-{after} by MI (raw cols always kept)")

    def _prune_derived_to(self, survivors):
        """Drop derived columns not in `survivors` from header / datatypes /
        population / expansion records, remap the MI tables, and re-materialize
        the surviving derived values (so the TF-IDF L2-norm basis is the kept
        token set, identically at train and inference)."""
        drop = self._derived_names - survivors
        if not drop:
            return
        keep_pairs = [(i, h) for i, h in enumerate(self.header) if h not in drop]
        new_mi, new_bins = {}, {}
        for new_i, (old_i, _) in enumerate(keep_pairs):
            if old_i in self._feature_mi:
                new_mi[new_i] = self._feature_mi[old_i]
            if old_i in self._feature_bins:
                new_bins[new_i] = self._feature_bins[old_i]
        self.header = [h for _, h in keep_pairs]
        self.datatypes = [self.datatypes[i] for i, _ in keep_pairs]
        self._feature_mi = new_mi
        self._feature_bins = new_bins
        for row in self.population:
            for d in drop:
                row.pop(d, None)
        # prune the fit records to survivors, drop now-empty records
        for rec in self.expansions:
            rec["derived"] = [d for d in rec["derived"] if d["name"] in survivors]
        self.expansions = [rec for rec in self.expansions if rec["derived"]]
        self._rebuild_derived_names()
        # re-materialize: recompute derived values with the pruned token set so
        # the L2 norm is over survivors only — matches what inference will do.
        for row in self.population:
            self._apply_expansions_inplace(row)

    def _rebuild_derived_names(self):
        """Set of derived column names — the machine marker (the '::' is the
        human marker). Used to keep str()-based numeric literals (ND digit-count)
        off synthetic numeric columns, where 'digits in a token count' is
        meaningless and only muddies the audit."""
        names = set()
        for rec in getattr(self, "expansions", []) or []:
            for d in rec["derived"]:
                names.add(d["name"])
        self._derived_names = names

    def _is_derived(self, index):
        """True if header[index] is a derived (expansion) column."""
        dn = getattr(self, "_derived_names", None)
        if not dn:
            return False
        return self.header[index] in dn

    def _apply_expansions_inplace(self, row):
        """Compute and write every derived value into `row` (a population or
        normalized inference dict). Dispatches per fit-record family.

        TOKENSET (TF-IDF): tf = raw term count in the document, weight = tf*idf
        (idf frozen at fit), then L2-normalize across that source's tfidf columns
        (sklearn default). A token unseen at fit has no column.

        GAUSSIAN: z = (x-mu)/sigma with mu,sigma frozen at fit; density =
        exp(-0.5 z^2); cdf = Phi(z) = 0.5*(1+erf(z/√2)). A non-numeric / missing
        value is treated as 0.0 — Snake's universal numeric default — so an
        absent column stays identical to an explicit 0.0 column (the contract
        test_missing_column_treated_as_full_na pins)."""
        for rec in self.expansions:
            family = rec.get("family", "TOKENSET")
            if family == "GAUSSIAN":
                self._apply_gaussian_inplace(row, rec)
                continue
            if family == "CATEGORICAL":
                self._apply_categorical_inplace(row, rec)
                continue
            src = rec["source"]
            delim = rec.get("delim", " ")
            toks = _tokenize(row.get(src, ""), delim)
            n_tok = len(toks)
            counts = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            # First pass: raw tf*idf weights for the kept tokens.
            weights = {}
            for d in rec["derived"]:
                if d["op"] == "TFIDF":
                    tf = counts.get(d["arg"], 0)
                    weights[d["name"]] = tf * d["idf"] if tf else 0.0
            # L2 normalize across this source's tfidf vector (sklearn default).
            if rec.get("norm") == "l2":
                norm = math.sqrt(sum(w * w for w in weights.values()))
                if norm > 0:
                    for k in weights:
                        weights[k] /= norm
            # Write everything.
            for d in rec["derived"]:
                op = d["op"]
                if op == "TFIDF":
                    row[d["name"]] = weights.get(d["name"], 0.0)
                elif op == "CNT":
                    row[d["name"]] = float(n_tok)
                elif op == "HAS":  # retained for forward-compat / forced configs
                    row[d["name"]] = 1.0 if d["arg"] in counts else 0.0
        return row

    def _apply_gaussian_inplace(self, row, rec):
        """Write the Gaussian-position KPIs for one source numeric column."""
        src = rec["source"]
        mu = rec["mu"]
        sigma = rec["sigma"] or 1.0
        v = row.get(src)
        if isinstance(v, (int, float)) and v == v:
            x = float(v)
        else:
            x = 0.0  # Snake's universal numeric default: missing == explicit 0.0
        z = (x - mu) / sigma
        for d in rec["derived"]:
            op = d["op"]
            if op == "GZ":
                row[d["name"]] = z
            elif op == "GD":
                row[d["name"]] = math.exp(-0.5 * z * z)
            elif op == "GC":
                row[d["name"]] = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    def _apply_categorical_inplace(self, row, rec):
        """Write the one-hot bits for one low-card source column. Each derived
        column is 1.0 iff the row's source value equals the fitted value, else 0.0.

        Equality is type-tolerant on the numeric side: the population stores numeric
        features as float (e.g. 1.0) while a raw inference dict may carry int (1) or
        a numeric string ("1"). We compare on a normalized key so 1 == 1.0 == "1"
        all light the same bit — the same str(int)!=str(float) class of mismatch
        that _normalize_features fixes for native literals (see CLAUDE.md v5.4.3)."""
        src = rec["source"]
        raw = row.get(src)
        key = self._cat_key(raw)
        for d in rec["derived"]:
            row[d["name"]] = 1.0 if self._cat_key(d["arg"]) == key else 0.0

    @staticmethod
    def _cat_key(v):
        """Normalized equality key: numbers (and numeric strings) collapse to a
        canonical float key so 1, 1.0 and '1' match; everything else compares as
        its string. Missing -> a sentinel that never equals a fitted value."""
        if _is_missing(v):
            return ("\x00missing",)
        if isinstance(v, bool):
            return ("b", v)
        if isinstance(v, (int, float)):
            return ("n", float(v))
        s = str(v).strip()
        try:
            return ("n", float(s))
        except (ValueError, TypeError):
            return ("s", s)

    def _expand_row(self, X):
        """Inference-side: given a normalized single datapoint dict, add the
        derived keys. Returns the same dict (mutated) for the caller to score."""
        if getattr(self, "expansions", None):
            self._apply_expansions_inplace(X)
        return X

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _expansions_to_json(self):
        """Return the JSON-safe expansions block (already pure lists/dicts)."""
        return getattr(self, "expansions", []) or []

    def _expansions_from_json(self, loaded):
        """Restore expansions from a loaded model. Old models lack the key ->
        []. Header/datatypes already carry the derived columns (they were
        serialized as ordinary columns), so we only need the fit records to
        reapply at inference."""
        self.expansions = loaded.get("expansions", []) or []
