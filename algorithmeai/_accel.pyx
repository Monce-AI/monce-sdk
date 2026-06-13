# cython: language_level=3
"""
Cython-accelerated hot paths for Snake classifier.

Build:  python setup.py build_ext --inplace
Install: pip install -e ".[fast]" && python setup.py build_ext --inplace

These are standalone functions (not methods) that mirror the pure-Python
logic in snake.py. They are conditionally imported at module level.
"""
from libc.math cimport log2, log, fabs, floor, copysign
from libc.stdlib cimport abs as c_abs

# ---------------------------------------------------------------------------
# String helper functions (C-speed equivalents of module-level helpers)
# ---------------------------------------------------------------------------

cdef int _levenshtein_c(str a, str b):
    """String distance: exact DP for short, O(n) bag-of-chars for long."""
    cdef int la = len(a)
    cdef int lb = len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    # Short strings: exact DP
    if la <= 32 and lb <= 32:
        prev = list(range(lb + 1))
        for i in range(la):
            curr = [i + 1] + [0] * lb
            for j in range(lb):
                cost = 0 if a[i] == b[j] else 1
                curr[j + 1] = min(prev[j + 1] + 1, curr[j] + 1, prev[j] + cost)
            prev = curr
        return prev[lb]
    # Long strings: O(n) char-frequency distance
    cdef dict fa = {}
    cdef dict fb = {}
    cdef str c
    cdef int shared = 0
    for c in a:
        fa[c] = fa.get(c, 0) + 1
    for c in b:
        fb[c] = fb.get(c, 0) + 1
    for c in set(fa) | set(fb):
        shared += min(fa.get(c, 0), fb.get(c, 0))
    return (la - shared) + (lb - shared)


cdef double _jaccard_bigrams_c(str a, str b):
    """Jaccard similarity on char bigrams."""
    if len(a) < 2 and len(b) < 2:
        return 1.0 if a == b else 0.0
    cdef set sa = {a[i:i+2] for i in range(max(0, len(a)-1))}
    cdef set sb = {b[i:i+2] for i in range(max(0, len(b)-1))}
    cdef set union = sa | sb
    if not union:
        return 1.0
    return <double>len(sa & sb) / <double>len(union)


cdef int _common_prefix_len_c(str a, str b):
    cdef int n = min(len(a), len(b))
    cdef int i
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


cdef int _common_suffix_len_c(str a, str b):
    return _common_prefix_len_c(a[::-1], b[::-1])


cdef double _entropy_c(str s):
    """Shannon entropy of character distribution."""
    if not s:
        return 0.0
    cdef dict freq = {}
    cdef str c
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    cdef int n = len(s)
    cdef double result = 0.0
    cdef double p
    for cnt in freq.values():
        p = <double>cnt / <double>n
        result -= p * log2(p)
    return result


cdef double _hex_ratio_c(str s):
    if not s:
        return 0.0
    cdef set hex_chars = set("0123456789abcdefABCDEF")
    cdef int count = 0
    cdef str c
    for c in s:
        if c in hex_chars:
            count += 1
    return <double>count / <double>len(s)


cdef double _repeat_period_score_c(str s):
    if len(s) > 64:
        s = s[:64]
    cdef int slen = len(s)
    if slen < 4:
        return 0.0
    cdef double best = 0.0
    cdef double score
    cdef int period, matches, i
    for period in range(1, slen // 2 + 1):
        matches = 0
        for i in range(period, slen):
            if s[i] == s[i % period]:
                matches += 1
        score = <double>matches / <double>(slen - period)
        if score > best:
            best = score
    return best


cdef int _count_upper_c(str s):
    cdef int count = 0
    cdef str c
    for c in s:
        if c.isupper():
            count += 1
    return count


cdef int _count_digits_c(str s):
    cdef int count = 0
    cdef str c
    for c in s:
        if c.isdigit():
            count += 1
    return count


cdef int _count_special_c(str s):
    cdef int count = 0
    cdef str c
    for c in s:
        if not c.isalnum() and not c.isspace():
            count += 1
    return count


cdef double _signed_log_c(double x):
    if x == 0:
        return 0.0
    return copysign(log(fabs(x) + 1), x)


cdef double _mag_c(double x):
    if x == 0:
        return 0.0
    return floor(log(fabs(x) + 1e-300) / log(10.0))


# ---------------------------------------------------------------------------
# Core inference functions
# ---------------------------------------------------------------------------

def apply_literal_fast(dict X, list literal, list header):
    """Cython version of Snake.apply_literal — returns True/False."""
    cdef int index = literal[0]
    cdef object value = literal[1]
    cdef bint negat = literal[2]
    cdef str datat = literal[3]
    cdef str key = header[index]

    if key not in X:
        return False

    cdef object field = X[key]
    cdef str sfield
    cdef double dfield
    cdef int ival
    cdef double dval
    cdef list vlist
    cdef dict ref_freq
    cdef double threshold, chi_sq, observed, expected, mu, std

    # --- Original literal types ---
    if datat == "TWS":
        sfield = str(field)
        if negat:
            return value <= len(sfield.split(" "))
        return value > len(sfield.split(" "))
    elif datat == "TPS":
        sfield = str(field)
        if negat:
            return value <= len(sfield.split(","))
        return value > len(sfield.split(","))
    elif datat == "TSS":
        sfield = str(field)
        if negat:
            return value <= len(sfield.split("."))
        return value > len(sfield.split("."))
    elif datat == "TLN":
        sfield = str(field)
        if negat:
            return value <= len(set(sfield))
        return value > len(set(sfield))
    elif datat == "TN":
        sfield = str(field)
        if negat:
            return value <= len(sfield)
        return value > len(sfield)
    elif datat == "T":
        sfield = str(field)
        if negat:
            return value not in sfield
        return value in sfield
    elif datat == "N":
        if negat:
            return value <= field
        return value > field

    # --- New literal types (v5.2.0) ---
    elif datat == "ND":
        ival = _count_digits_c(str(field))
        return value <= ival if negat else value > ival
    elif datat == "TUC":
        ival = _count_upper_c(str(field))
        return value <= ival if negat else value > ival
    elif datat == "TDC":
        ival = _count_digits_c(str(field))
        return value <= ival if negat else value > ival
    elif datat == "TSC":
        ival = _count_special_c(str(field))
        return value <= ival if negat else value > ival
    elif datat == "LEV":
        vlist = value
        ival = _levenshtein_c(str(field), <str>vlist[0])
        threshold = vlist[1]
        return ival <= threshold if negat else ival > threshold
    elif datat == "JAC":
        vlist = value
        dval = _jaccard_bigrams_c(str(field), <str>vlist[0])
        threshold = vlist[1]
        return dval >= threshold if negat else dval < threshold
    elif datat == "PFX":
        vlist = value
        ival = _common_prefix_len_c(str(field), <str>vlist[0])
        threshold = vlist[1]
        return ival >= threshold if negat else ival < threshold
    elif datat == "SFX":
        vlist = value
        ival = _common_suffix_len_c(str(field), <str>vlist[0])
        threshold = vlist[1]
        return ival >= threshold if negat else ival < threshold
    elif datat == "ENT":
        dval = _entropy_c(str(field))
        return value <= dval if negat else value > dval
    elif datat == "HEX":
        dval = _hex_ratio_c(str(field))
        return value <= dval if negat else value > dval
    elif datat == "REP":
        dval = _repeat_period_score_c(str(field))
        return value <= dval if negat else value > dval
    elif datat == "CFC":
        vlist = value
        ref_freq = vlist[0]
        threshold = vlist[1]
        sfield = str(field)
        nf = max(len(sfield), 1)
        f_freq = {}
        for c in sfield:
            f_freq[c] = f_freq.get(c, 0) + 1
        chi_sq = 0.0
        all_chars = set(ref_freq) | set(f_freq)
        for c in all_chars:
            observed = <double>f_freq.get(c, 0) / <double>nf
            expected = ref_freq.get(c, 0.001)
            if expected < 0.001:
                expected = 0.001
            chi_sq += (observed - expected) ** 2 / expected
        return chi_sq < threshold if negat else chi_sq >= threshold
    elif datat == "NZ":
        vlist = value
        mu = vlist[0]
        std = vlist[1]
        threshold = vlist[2]
        dval = (<double>field - mu) / std
        return dval >= threshold if negat else dval < threshold
    elif datat == "NL":
        dval = _signed_log_c(<double>field)
        return value <= dval if negat else value > dval
    elif datat == "NMG":
        dval = _mag_c(<double>field)
        return value <= dval if negat else value > dval
    elif datat == "NZR":
        return (field != 0) if negat else (field == 0)
    elif datat == "NRG":
        vlist = value
        inside = vlist[0] < field <= vlist[1]
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
        ival = 0
        for c in sfield:
            if c in "aeiouAEIOU":
                ival += 1
        dval = <double>ival / <double>max(len(sfield), 1)
        return value <= dval if negat else value > dval
    return False


def apply_clause_fast(dict X, list clause, list header):
    """Cython version of Snake.apply_clause — OR over literals."""
    cdef list literal
    for literal in clause:
        if apply_literal_fast(X, literal, header):
            return True
    return False


def traverse_chain_fast(list chain, dict X, list header):
    """Cython version of traverse_chain — walk IF/ELIF/ELSE chain."""
    cdef dict entry
    cdef list condition
    cdef list lit
    cdef bint all_match
    for entry in chain:
        condition = entry["condition"]
        if condition is None:
            return entry
        all_match = True
        for lit in condition:
            if not apply_literal_fast(X, lit, header):
                all_match = False
                break
        if all_match:
            return entry
    if chain:
        return chain[-1]
    return None


def get_lookalikes_fast(list layers, dict X, list header, list targets):
    """Cython version of Snake.get_lookalikes — full inference in C."""
    cdef list all_lookalikes = []
    cdef list layer, clause_bool
    cdef dict bucket
    cdef set negated
    cdef int i, global_idx, c_idx
    cdef list condition
    cdef bint all_negated

    for layer in layers:
        bucket = traverse_chain_fast(layer, X, header)
        if bucket is None:
            continue
        clause_bool = [apply_clause_fast(X, c, header) for c in bucket["clauses"]]
        negated = {i for i in range(len(clause_bool)) if not clause_bool[i]}
        for l in bucket["lookalikes"]:
            for condition in bucket["lookalikes"][l]:
                all_negated = True
                for c_idx in condition:
                    if c_idx not in negated:
                        all_negated = False
                        break
                if all_negated:
                    global_idx = bucket["members"][int(l)]
                    all_lookalikes.append([global_idx, targets[global_idx], condition])
    return all_lookalikes


# ---------------------------------------------------------------------------
# Training acceleration functions
# ---------------------------------------------------------------------------


def filter_ts_remainder_fast(list Ts, list literal, list header):
    """Filter Ts to keep only those where apply_literal is False (remainder).
    Used in construct_clause to find Ts not yet covered by the last literal."""
    cdef list result = []
    cdef dict T
    for T in Ts:
        if not apply_literal_fast(T, literal, header):
            result.append(T)
    return result


def minimize_clause_fast(list clause, list Ts, list header):
    """Minimize a clause by removing redundant literals.
    A literal is redundant if removing it still leaves the clause True on all Ts."""
    cdef int i = 0
    cdef int j, n
    cdef list sub_clause
    cdef dict T
    cdef bint some_fail
    while i < len(clause):
        n = len(clause)
        sub_clause = [clause[j] for j in range(n) if j != i]
        some_fail = False
        for T in Ts:
            if not apply_clause_fast(T, sub_clause, header):
                some_fail = True
                break
        if some_fail:
            i += 1
        else:
            clause = sub_clause
    return clause


def filter_indices_by_literal_fast(list indices, list population, list literal, list header):
    """Filter population indices where apply_literal is True.
    Used in build_condition to filter matching indices by a literal."""
    cdef list result = []
    cdef int idx
    for idx in indices:
        if apply_literal_fast(population[idx], literal, header):
            result.append(idx)
    return result


def check_clause_covers_all_fast(list Ts, list clause, list header):
    """Check that a clause (OR of literals) is True on all samples in Ts."""
    cdef dict T
    for T in Ts:
        if not apply_clause_fast(T, clause, header):
            return False
    return True


def filter_consequence_fast(list local_pop, list local_targets, object target_value, list clause, list header):
    """Compute consequence indices and remaining Fs for a clause and target value.
    Returns (consequence_indices, remaining_Fs) where:
    - consequence_indices: indices where target matches AND clause is False (NOT eliminated)
    - remaining_Fs: list of Fs where clause is True (eliminated, need further clauses)
    """
    cdef list consequence = []
    cdef list remaining_fs = []
    cdef int i
    cdef int n = len(local_pop)
    for i in range(n):
        if local_targets[i] == target_value:
            if not apply_clause_fast(local_pop[i], clause, header):
                consequence.append(i)
            else:
                remaining_fs.append(local_pop[i])
    return consequence, remaining_fs


def batch_get_lookalikes_fast(list layers, list Xs, list header, list targets):
    """Batch lookalike computation: route all queries per layer, group by bucket.
    Returns list of lookalike-lists, one per query.
    Amortizes chain traversal and improves cache locality for clause evaluation."""
    cdef int n_queries = len(Xs)
    cdef int q_idx, i, global_idx, c_idx
    cdef list result = [[] for _ in range(n_queries)]
    cdef dict X, bucket, grouped
    cdef list layer, clause_bool, condition
    cdef set negated
    cdef bint all_negated

    for layer in layers:
        # Group all queries by their routed bucket (using id for same-object grouping)
        grouped = {}  # id(bucket) -> list of (q_idx, X)
        for q_idx in range(n_queries):
            X = Xs[q_idx]
            bucket = traverse_chain_fast(layer, X, header)
            if bucket is None:
                continue
            bucket_id = id(bucket)
            if bucket_id not in grouped:
                grouped[bucket_id] = (bucket, [])
            grouped[bucket_id][1].append((q_idx, X))

        # Process each bucket group
        for bucket_id in grouped:
            bucket, queries = grouped[bucket_id]
            clauses = bucket["clauses"]
            members = bucket["members"]
            lookalikes_map = bucket["lookalikes"]

            for q_idx, X in queries:
                clause_bool = [apply_clause_fast(X, c, header) for c in clauses]
                negated = {i for i in range(len(clause_bool)) if not clause_bool[i]}
                for l in lookalikes_map:
                    for condition in lookalikes_map[l]:
                        all_negated = True
                        for c_idx in condition:
                            if c_idx not in negated:
                                all_negated = False
                                break
                        if all_negated:
                            global_idx = members[int(l)]
                            result[q_idx].append([global_idx, targets[global_idx], condition])

    return result


def batch_predict_fast(list layers, list Xs, list header, list targets, list unique_targets):
    """Cython-accelerated batch prediction. Returns list of (prediction, confidence, prob_dict)."""
    cdef list results = []
    cdef int n_classes = len(unique_targets)
    cdef double uniform = 1.0 / n_classes if n_classes > 0 else 0.0
    cdef dict X
    cdef list lookalikes
    cdef dict prob
    cdef double best_p, p
    cdef object best_tv, tv
    cdef int n_lk

    for X in Xs:
        lookalikes = get_lookalikes_fast(layers, X, header, targets)
        n_lk = len(lookalikes)

        if n_lk == 0:
            prob = {}
            for tv in unique_targets:
                try:
                    prob[tv] = uniform
                except TypeError:
                    import json
                    prob[json.dumps(tv, sort_keys=True)] = uniform
            best_tv = unique_targets[0]
            best_p = uniform
        else:
            prob = {}
            best_tv = unique_targets[0]
            best_p = -1.0
            for tv in unique_targets:
                count = 0
                for triple in lookalikes:
                    if triple[1] == tv:
                        count += 1
                p = <double>count / <double>n_lk
                try:
                    prob[tv] = p
                except TypeError:
                    import json
                    prob[json.dumps(tv, sort_keys=True)] = p
                if p > best_p:
                    best_p = p
                    best_tv = tv

        results.append({
            "prediction": best_tv,
            "probability": prob,
            "confidence": best_p,
        })
    return results
