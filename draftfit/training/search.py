"""Optional, bounded Optuna ask/report/tell controller; never runs a trainer."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import math
import re
import sqlite3
import sys
import time
from contextlib import contextmanager
from pathlib import Path

OPTUNA_VERSION = "4.5.0"
FAMILIES = {"dspark", "dflash", "dflash2"}
TERMINAL = {"complete", "pruned", "rejected", "interrupted", "failed"}
# Objective weights, data, seeds, architecture, schedules and batch geometry stay
# in the immutable base config. These knobs preserve the validation loss formula.
PARAMETERS = {
    "training.learning_rate": ("float", 0, 1, False),
    "training.warmup_ratio": ("float", 0, 1, True),
    "training.max_grad_norm": ("float", 0, 100, False),
    "training.lr_scheduler": ("categorical", ("cosine", "constant")),
    "training.optimizer_cpu_offload": ("categorical", (False, True)),
    "training.objective_chunk_blocks": ("int", 1, 65536, True),
}


class SearchError(ValueError):
    """Invalid input, exhausted budget, or incompatible persistent state."""


def _object(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise SearchError(
            f"{label} requires exactly these keys: {', '.join(sorted(keys))}"
        )
    return value


def _integer(value, label, minimum=0, maximum=10**12):
    if type(value) is not int or not minimum <= value <= maximum:
        raise SearchError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def _number(value, label, minimum=0, maximum=10**12):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid or not minimum <= value <= maximum:
        raise SearchError(f"{label} must be a finite number in [{minimum}, {maximum}]")
    return value


def _text(value, label, maximum=1000):
    if type(value) is not str or not value.strip() or len(value) > maximum:
        raise SearchError(
            f"{label} must be a nonempty string of at most {maximum} characters"
        )
    return value


def _sha(value, label):
    if type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise SearchError(f"{label} must be a lowercase SHA256")
    return value


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise SearchError("Only finite JSON values are supported") from exc


def digest(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SearchError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path):
    """Read strict JSON, including duplicate-key and NaN/Infinity rejection."""

    def invalid_constant(value):
        raise SearchError(f"Non-finite JSON constant: {value}")

    try:
        return json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=invalid_constant,
        )
    except json.JSONDecodeError as exc:
        raise SearchError(f"Invalid JSON: {exc}") from exc


def validate_contract(value):
    """Validate and copy an exact, single-family screening contract."""
    c = _object(
        value,
        {
            "schema_version",
            "study_name",
            "identity",
            "space",
            "budget",
            "memory",
            "sampler",
            "pruner",
            "report_updates",
        },
        "contract",
    )
    if type(c["schema_version"]) is not int or c["schema_version"] != 1:
        raise SearchError("schema_version must be 1")
    if type(c["study_name"]) is not str or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", c["study_name"]
    ):
        raise SearchError(
            "study_name must contain 1-80 letters, digits, dots, underscores or hyphens"
        )
    identity = _object(
        c["identity"],
        {
            "target",
            "draft",
            "data",
            "objective",
            "base_config_sha256",
            "code_revision",
        },
        "identity",
    )
    target = _object(
        identity["target"], {"model_id", "revision", "weights_sha256"}, "target"
    )
    _text(target["model_id"], "target.model_id")
    _text(target["revision"], "target.revision")
    _sha(target["weights_sha256"], "target.weights_sha256")
    draft = _object(identity["draft"], {"config_sha256", "checkpoint_sha256"}, "draft")
    data = _object(
        identity["data"], {"train_sha256", "validation_sha256", "split_sha256"}, "data"
    )
    objective = _object(
        identity["objective"],
        {
            "family",
            "name",
            "config_sha256",
            "validation_protocol_sha256",
        },
        "objective",
    )
    family = objective["family"]
    if type(family) is not str or family not in FAMILIES:
        raise SearchError("objective.family must be dspark, dflash or dflash2")
    _text(objective["name"], "objective.name")
    for section, fields in (
        (draft, draft),
        (data, data),
        (objective, ("config_sha256", "validation_protocol_sha256")),
    ):
        for field in fields:
            _sha(section[field], field)
    if data["train_sha256"] == data["validation_sha256"]:
        raise SearchError("Training and validation data digests must differ")
    _sha(identity["base_config_sha256"], "base_config_sha256")
    if type(identity["code_revision"]) is not str or not re.fullmatch(
        r"(?:[0-9a-f]{40}|[0-9a-f]{64})", identity["code_revision"]
    ):
        raise SearchError("code_revision must be a full lowercase revision hash")

    if type(c["space"]) is not dict or not c["space"]:
        raise SearchError("space must be a nonempty object")
    active = 0
    for name, spec in c["space"].items():
        if name not in PARAMETERS:
            raise SearchError(f"Parameter is not allowed: {name}")
        if type(spec) is not dict:
            raise SearchError(f"Invalid distribution: {name}")
        rule = PARAMETERS[name]
        keys = (
            {"kind", "choices"} if rule[0] == "categorical" else {"kind", "low", "high"}
        )
        if rule[0] == "float":
            keys.add("log")
        if "strategies" in spec:
            keys.add("strategies")
            strategies = spec["strategies"]
            if (
                type(strategies) is not list
                or not strategies
                or any(type(s) is not str or s not in FAMILIES for s in strategies)
                or len(set(strategies)) != len(strategies)
            ):
                raise SearchError(f"Invalid strategy condition: {name}")
        _object(spec, keys, name)
        if spec["kind"] != rule[0]:
            raise SearchError(f"Wrong distribution kind: {name}")
        active += family in spec.get("strategies", [family])
        if rule[0] == "categorical":
            choices = spec["choices"]
            if type(choices) is not list or not choices:
                raise SearchError(f"choices must be nonempty: {name}")
            allowed = {(type(v), v) for v in rule[1]}
            if any(
                type(v) not in (bool, str) or (type(v), v) not in allowed
                for v in choices
            ):
                raise SearchError(f"Unsupported categorical value: {name}")
            if len({_json(v) for v in choices}) != len(choices):
                raise SearchError(f"Duplicate choices: {name}")
        else:
            check = _integer if rule[0] == "int" else _number
            low = check(spec["low"], f"{name}.low", rule[1], rule[2])
            high = check(spec["high"], f"{name}.high", rule[1], rule[2])
            if low > high or (not rule[3] and low == rule[1]):
                raise SearchError(f"Invalid range: {name}")
            if rule[0] == "float" and (
                type(spec["log"]) is not bool or (spec["log"] and low <= 0)
            ):
                raise SearchError(f"Invalid logarithmic range: {name}")
    if not active:
        raise SearchError("space has no parameters active for this family")

    b = _object(
        c["budget"],
        {
            "max_trials",
            "max_total_updates",
            "max_total_seconds",
            "trial_max_updates",
            "trial_max_seconds",
        },
        "budget",
    )
    _integer(b["max_trials"], "max_trials", 1, 10**6)
    _integer(b["trial_max_updates"], "trial_max_updates", 1)
    _number(b["trial_max_seconds"], "trial_max_seconds", 0.001)
    if b["max_total_updates"] is None and b["max_total_seconds"] is None:
        raise SearchError("An aggregate update or elapsed-seconds budget is required")
    if b["max_total_updates"] is not None:
        _integer(b["max_total_updates"], "max_total_updates", b["trial_max_updates"])
    if b["max_total_seconds"] is not None:
        _number(b["max_total_seconds"], "max_total_seconds", b["trial_max_seconds"])
    rungs = c["report_updates"]
    if type(rungs) is not list or not rungs:
        raise SearchError("report_updates must be a nonempty increasing list")
    for rung in rungs:
        _integer(rung, "report update", 1, b["trial_max_updates"])
    if rungs != sorted(set(rungs)) or rungs[-1] != b["trial_max_updates"]:
        raise SearchError("report_updates must increase to trial_max_updates")
    memory = _object(c["memory"], {"max_trial_mib", "reserve_mib"}, "memory")
    _number(memory["max_trial_mib"], "max_trial_mib", 0.001)
    _number(memory["reserve_mib"], "reserve_mib", 0.001)
    sampler = _object(c["sampler"], {"seed", "startup_trials"}, "sampler")
    _integer(sampler["seed"], "seed", 0, 2**32 - 1)
    _integer(sampler["startup_trials"], "sampler.startup_trials", 1, 10**6)
    pruner = _object(
        c["pruner"],
        {"startup_trials", "warmup_updates", "interval_updates", "min_trials"},
        "pruner",
    )
    for key in pruner:
        _integer(
            pruner[key], f"pruner.{key}", 0 if key == "warmup_updates" else 1, 10**6
        )
    return json.loads(_json(c))


class SearchStudy:
    """Single-host controller with a conservative, irrevocable reservation ledger.

    All calls serialize on a local advisory lock. The adapter's SQLite tables
    record intentions before Optuna writes; reopening reconciles those writes.
    A crash while allocating is an interrupted, charged attempt, never a retry.
    """

    def __init__(self, storage, contract):
        self._contract = validate_contract(contract)
        self.contract_digest = digest(self._contract)
        self.path = Path(storage).expanduser().resolve()
        if not self.path.parent.is_dir():
            raise SearchError("The local SQLite parent directory must already exist")
        if self.path.suffix not in {".sqlite", ".sqlite3", ".db"}:
            raise SearchError("storage must be a local .sqlite, .sqlite3 or .db path")
        if any(ch in str(self.path) for ch in "?#{}\n\r"):
            raise SearchError(
                "storage path cannot contain URL/template delimiters or newlines"
            )

    def _optuna(self):
        try:
            optuna = importlib.import_module("optuna")
        except ImportError as exc:
            raise SearchError(
                'Optional dependency missing; install "draftfit[search]" in the controller environment'
            ) from exc
        if optuna.__version__ != OPTUNA_VERSION:
            raise SearchError(f"This adapter requires optuna=={OPTUNA_VERSION}")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        return optuna

    def _rows(self, db):
        return [
            json.loads(row[0])
            for row in db.execute(
                "SELECT record FROM draftfit_search_trials WHERE study = ? ORDER BY attempt",
                (self._contract["study_name"],),
            )
        ]

    def _save(self, db, record):
        db.execute(
            "INSERT OR REPLACE INTO draftfit_search_trials(study, attempt, record) VALUES(?, ?, ?)",
            (self._contract["study_name"], record["trial"], _json(record)),
        )
        db.commit()

    @contextmanager
    def _session(self):
        optuna = self._optuna()
        with Path(str(self.path) + ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            storage = optuna.storages.RDBStorage(f"sqlite:///{self.path}")
            db = sqlite3.connect(self.path)
            try:
                db.execute(
                    "CREATE TABLE IF NOT EXISTS draftfit_search_contracts (study TEXT PRIMARY KEY, digest TEXT NOT NULL, contract TEXT NOT NULL, created REAL NOT NULL)"
                )
                db.execute(
                    "CREATE TABLE IF NOT EXISTS draftfit_search_trials (study TEXT NOT NULL, attempt INTEGER NOT NULL, record TEXT NOT NULL, PRIMARY KEY(study, attempt))"
                )
                row = db.execute(
                    "SELECT digest, contract, created FROM draftfit_search_contracts WHERE study = ?",
                    (self._contract["study_name"],),
                ).fetchone()
                if row is None:
                    if any(
                        s.study_name == self._contract["study_name"]
                        for s in optuna.get_all_study_summaries(storage)
                    ):
                        raise SearchError("Refusing to adopt an unmanaged Optuna study")
                    created = time.time()
                    db.execute(
                        "INSERT INTO draftfit_search_contracts VALUES(?, ?, ?, ?)",
                        (
                            self._contract["study_name"],
                            self.contract_digest,
                            _json(self._contract),
                            created,
                        ),
                    )
                    db.commit()
                else:
                    if row[0] != self.contract_digest or row[1] != _json(
                        self._contract
                    ):
                        raise SearchError(
                            "Study contract digest mismatch; budgets and lineage cannot be changed on reuse"
                        )
                    created = row[2]
                p = self._contract["pruner"]
                study = optuna.create_study(
                    storage=storage,
                    study_name=self._contract["study_name"],
                    load_if_exists=True,
                    direction="minimize",
                    sampler=self._sampler(optuna, 0),
                    pruner=optuna.pruners.MedianPruner(
                        n_startup_trials=p["startup_trials"],
                        n_warmup_steps=p["warmup_updates"],
                        interval_steps=p["interval_updates"],
                        n_min_trials=p["min_trials"],
                    ),
                )
                stored = study.user_attrs.get("draftfit_contract_digest")
                if stored is not None and stored != self.contract_digest:
                    raise SearchError("Optuna study contract digest mismatch")
                study.set_user_attr("draftfit_contract_digest", self.contract_digest)
                records = self._rows(db)
                mapped = {
                    r["optuna_trial_number"]
                    for r in records
                    if r["optuna_trial_number"] is not None
                }
                orphans = [t for t in study.trials if t.number not in mapped]
                allocating = [r for r in records if r["state"] == "allocating"]
                if orphans and (
                    len(orphans) != 1
                    or len(allocating) != 1
                    or allocating[0]["optuna_trial_number"] is not None
                ):
                    raise SearchError(
                        "Unmanaged Optuna trials detected; refusing to reset accounting"
                    )
                for record in allocating:
                    if orphans:
                        record["optuna_trial_number"] = orphans.pop().number
                    record.update(
                        state="interrupted",
                        reason="Controller interrupted while allocating; reservation retained",
                    )
                    self._save(db, record)
                for record in records:
                    self._sync(db, optuna, storage, study, record)
                yield db, optuna, storage, study, created
            finally:
                db.close()
                storage.remove_session()
                storage.engine.dispose()

    def _sampler(self, optuna, attempt):
        s = self._contract["sampler"]
        # RDB storage does not persist RNG state. Derive a distinct reproducible
        # seed for every reserved attempt instead of restarting one RNG stream.
        seed = int(
            hashlib.sha256(f"{s['seed']}:{attempt}".encode()).hexdigest()[:8], 16
        )
        return optuna.samplers.TPESampler(
            seed=seed, n_startup_trials=s["startup_trials"]
        )

    def _live(self, optuna, storage, study, number):
        study_id = storage.get_study_id_from_name(study.study_name)
        trial_id = storage.get_trial_id_from_study_id_trial_number(study_id, number)
        return optuna.trial.Trial(study, trial_id)

    def _sync(self, db, optuna, storage, study, record):
        number = record["optuna_trial_number"]
        if number is None:
            return
        frozen = next((t for t in study.trials if t.number == number), None)
        if frozen is None:
            raise SearchError(
                "An accounted Optuna trial is missing; refusing to reset accounting"
            )
        if "config_digest" in record and frozen.params != record["params"]:
            raise SearchError("Optuna parameters changed outside the controller")
        expected = {"complete": "COMPLETE", "pruned": "PRUNED"}.get(
            record["state"], "FAIL"
        )
        if frozen.state.is_finished():
            if record["state"] not in TERMINAL or frozen.state.name != expected:
                raise SearchError("Optuna state changed outside the controller")
            if (
                record["state"] == "complete"
                and frozen.value != record["reports"][-1]["validation_loss"]
            ):
                raise SearchError("Optuna value changed outside the controller")
            return
        if frozen.state.name == "WAITING" and record["state"] == "interrupted":
            study_id = storage.get_study_id_from_name(study.study_name)
            trial_id = storage.get_trial_id_from_study_id_trial_number(study_id, number)
            storage.set_trial_state_values(trial_id, optuna.trial.TrialState.FAIL)
            return
        trial = self._live(optuna, storage, study, number)
        for report in record["reports"]:
            step = report["updates"] - 1
            if step in frozen.intermediate_values:
                if frozen.intermediate_values[step] != report["validation_loss"]:
                    raise SearchError("Conflicting Optuna intermediate value")
            else:
                trial.report(report["validation_loss"], step)
        if record.pop("decision_pending", False):
            if (
                record["reports"][-1]["updates"]
                < self._contract["budget"]["trial_max_updates"]
                and trial.should_prune()
            ):
                record.update(state="pruned", reason="MedianPruner screening decision")
            self._save(db, record)
        if record["state"] in TERMINAL:
            state = getattr(
                optuna.trial.TrialState,
                {"complete": "COMPLETE", "pruned": "PRUNED"}.get(
                    record["state"], "FAIL"
                ),
            )
            value = (
                record["reports"][-1]["validation_loss"]
                if record["state"] == "complete"
                else None
            )
            study.tell(number, values=value, state=state)

    def _record(self, db, trial):
        _integer(trial, "trial")
        for row in self._rows(db):
            if row["trial"] == trial:
                return row
        raise SearchError("Unknown trial")

    def _result(self, record):
        return dict(
            record, contract_digest=self.contract_digest, metric_role="screening_only"
        )

    def _charged(self, records):
        b = self._contract["budget"]
        totals = {"trials": len(records), "updates": 0, "seconds": 0}
        for record in records:
            receipts = record["reports"] + (
                [record["rejected_receipt"]] if "rejected_receipt" in record else []
            )
            for total, field, cap in (
                ("updates", "updates", "trial_max_updates"),
                ("seconds", "elapsed_seconds", "trial_max_seconds"),
            ):
                totals[total] += max([b[cap]] + [r[field] for r in receipts])
        return totals

    def ask(self, *, free_memory_mib, estimated_memory_mib, params=None):
        """Reserve one attempt and return overrides; rejected attempts stay spent."""
        _number(free_memory_mib, "free_memory_mib")
        _number(estimated_memory_mib, "estimated_memory_mib", 0.001)
        family = self._contract["identity"]["objective"]["family"]
        active_space = {
            name: spec
            for name, spec in self._contract["space"].items()
            if family in spec.get("strategies", [family])
        }
        if params is not None:
            _object(params, active_space, "params")
            for name, spec in active_space.items():
                value = params[name]
                if spec["kind"] == "categorical":
                    if not any(
                        type(value) is type(v) and value == v for v in spec["choices"]
                    ):
                        raise SearchError(
                            f"Fixed parameter is outside the search space: {name}"
                        )
                else:
                    check = _integer if spec["kind"] == "int" else _number
                    check(value, name, spec["low"], spec["high"])
            params = json.loads(_json(params))
        with self._session() as (db, optuna, storage, study, created):
            records = self._rows(db)
            if any(r["state"] not in TERMINAL for r in records):
                raise SearchError(
                    "A trial is still running; report/tell it explicitly before asking again"
                )
            b = self._contract["budget"]
            count = len(records)
            if count >= b["max_trials"]:
                raise SearchError("Trial count budget exhausted")
            charged = self._charged(records)
            for total, cap, field in (
                ("max_total_updates", "trial_max_updates", "updates"),
                ("max_total_seconds", "trial_max_seconds", "seconds"),
            ):
                if b[total] is not None and charged[field] + b[cap] > b[total]:
                    raise SearchError(f"Reservation exceeds {total}")
            if (
                b["max_total_seconds"] is not None
                and time.time() - created >= b["max_total_seconds"]
            ):
                raise SearchError("Study elapsed deadline exhausted")
            record = dict(
                trial=count,
                optuna_trial_number=None,
                state="allocating",
                reason=None,
                params={},
                reports=[],
                started_at=time.time(),
                free_memory_mib=free_memory_mib,
                estimated_memory_mib=estimated_memory_mib,
            )
            deadline = record["started_at"] + b["trial_max_seconds"]
            if b["max_total_seconds"] is not None:
                deadline = min(deadline, created + b["max_total_seconds"])
            record["limits"] = {
                "max_updates": b["trial_max_updates"],
                "deadline": deadline,
                "max_memory_mib": estimated_memory_mib,
            }
            self._save(db, record)
            study.sampler = self._sampler(optuna, count)
            if params is not None:
                study.enqueue_trial(params)
            trial = study.ask()
            record["optuna_trial_number"] = trial.number
            self._save(db, record)
            trial.set_user_attr("draftfit_contract_digest", self.contract_digest)
            trial.set_user_attr("draftfit_attempt", count)
            for name, spec in sorted(active_space.items()):
                if spec["kind"] == "categorical":
                    value = trial.suggest_categorical(name, spec["choices"])
                elif spec["kind"] == "int":
                    value = trial.suggest_int(name, spec["low"], spec["high"])
                else:
                    value = trial.suggest_float(
                        name, spec["low"], spec["high"], log=spec["log"]
                    )
                record["params"][name] = value
            record["config_digest"] = digest(
                {
                    "contract_digest": self.contract_digest,
                    "trial": count,
                    "params": record["params"],
                }
            )
            m = self._contract["memory"]
            rejected = (
                estimated_memory_mib > m["max_trial_mib"]
                or estimated_memory_mib + m["reserve_mib"] > free_memory_mib
            )
            record.update(
                state="rejected" if rejected else "running",
                reason="Memory admission gate rejected" if rejected else None,
            )
            self._save(db, record)
            self._sync(db, optuna, storage, study, record)
            return self._result(record)

    def report(self, trial, receipt):
        """Record one checkpoint's heldout objective loss and pruning decision."""
        _object(
            receipt,
            {
                "contract_digest",
                "config_digest",
                "checkpoint_sha256",
                "parent_checkpoint_sha256",
                "updates",
                "elapsed_seconds",
                "validation_loss",
                "peak_memory_mib",
                "free_memory_mib",
            },
            "receipt",
        )
        for key in (
            "contract_digest",
            "config_digest",
            "checkpoint_sha256",
            "parent_checkpoint_sha256",
        ):
            _sha(receipt[key], key)
        _integer(receipt["updates"], "updates", 1)
        for key in ("elapsed_seconds", "peak_memory_mib", "free_memory_mib"):
            _number(receipt[key], key)
        _number(receipt["validation_loss"], "validation_loss", -(10**12))
        receipt = json.loads(_json(receipt))
        with self._session() as (db, optuna, storage, study, created):
            record = self._record(db, trial)
            if receipt["contract_digest"] != self.contract_digest or receipt[
                "config_digest"
            ] != record.get("config_digest"):
                raise SearchError("Receipt contract/config digest mismatch")
            previous_receipts = record["reports"]
            if "rejected_receipt" in record:
                previous_receipts = previous_receipts + [record["rejected_receipt"]]
            for previous in previous_receipts:
                if previous["updates"] == receipt["updates"]:
                    if previous == receipt:
                        return self._result(record)
                    raise SearchError("Conflicting duplicate report")
            if record["state"] != "running":
                raise SearchError("Reports require a running trial")
            last = record["reports"][-1] if record["reports"] else None
            parent = (
                last["checkpoint_sha256"]
                if last
                else self._contract["identity"]["draft"]["checkpoint_sha256"]
            )
            ancestors = {self._contract["identity"]["draft"]["checkpoint_sha256"]}
            ancestors.update(r["checkpoint_sha256"] for r in record["reports"])
            if (
                receipt["parent_checkpoint_sha256"] != parent
                or receipt["checkpoint_sha256"] in ancestors
            ):
                raise SearchError("Checkpoint lineage mismatch")
            if last and (
                receipt["elapsed_seconds"] < last["elapsed_seconds"]
                or receipt["peak_memory_mib"] < last["peak_memory_mib"]
            ):
                raise SearchError(
                    "Cumulative elapsed time and peak memory cannot decrease"
                )
            b, m = self._contract["budget"], self._contract["memory"]
            reason = None
            if (
                receipt["updates"] > b["trial_max_updates"]
                or receipt["elapsed_seconds"] > b["trial_max_seconds"]
            ):
                reason = "Reported trial budget exceeded"
            elif time.time() - record["started_at"] > b["trial_max_seconds"]:
                reason = "Trial elapsed deadline exceeded"
            elif (
                b["max_total_seconds"] is not None
                and time.time() - created > b["max_total_seconds"]
            ):
                reason = "Study elapsed deadline exceeded"
            elif (
                receipt["peak_memory_mib"]
                > min(record["estimated_memory_mib"], m["max_trial_mib"])
                or receipt["free_memory_mib"] < m["reserve_mib"]
            ):
                reason = "Reported memory gate exceeded"
            if reason:
                record.update(state="rejected", reason=reason, rejected_receipt=receipt)
            else:
                if len(record["reports"]) >= len(self._contract["report_updates"]):
                    raise SearchError(
                        "Every checkpoint has been reported; tell the trial explicitly"
                    )
                next_update = self._contract["report_updates"][len(record["reports"])]
                if receipt["updates"] != next_update:
                    raise SearchError(
                        f"Expected checkpoint report at update {next_update}"
                    )
                record["reports"].append(receipt)
                record["decision_pending"] = True
            self._save(db, record)
            self._sync(db, optuna, storage, study, record)
            return self._result(record)

    def tell(self, trial, *, state, reason=None):
        """Finalize from the last report; no independent or invented score input."""
        if type(state) is not str or state not in TERMINAL:
            raise SearchError(
                "state must be complete, pruned, rejected, interrupted or failed"
            )
        if state != "complete":
            _text(reason, "reason")
        elif reason is not None:
            raise SearchError(
                "A complete trial does not accept a reason or score override"
            )
        with self._session() as (db, optuna, storage, study, created):
            record = self._record(db, trial)
            if record["state"] in TERMINAL:
                if record["state"] != state:
                    raise SearchError("A terminal trial cannot be reopened or changed")
                return self._result(record)
            if state == "complete":
                b = self._contract["budget"]
                if len(record["reports"]) != len(self._contract["report_updates"]):
                    raise SearchError(
                        "Completion requires every checkpoint report through trial_max_updates"
                    )
                if time.time() - record["started_at"] > b["trial_max_seconds"] or (
                    b["max_total_seconds"] is not None
                    and time.time() - created > b["max_total_seconds"]
                ):
                    raise SearchError(
                        "Elapsed deadline exceeded; explicitly tell interrupted or rejected"
                    )
            if state == "pruned" and not record["reports"]:
                raise SearchError(
                    "Pruning requires a validation report; use rejected/interrupted/failed"
                )
            record.update(state=state, reason=reason)
            self._save(db, record)
            self._sync(db, optuna, storage, study, record)
            return self._result(record)

    def status(self):
        with self._session() as (db, _optuna, _storage, _study, created):
            records = self._rows(db)
            b = self._contract["budget"]
            complete = [r for r in records if r["state"] == "complete"]
            best = (
                min(complete, key=lambda r: r["reports"][-1]["validation_loss"])
                if complete
                else None
            )
            return {
                "contract_digest": self.contract_digest,
                "metric_role": "screening_only",
                "budget": b,
                "created_at": created,
                "reserved": {
                    "trials": len(records),
                    "updates": len(records) * b["trial_max_updates"],
                    "seconds": len(records) * b["trial_max_seconds"],
                },
                "charged": self._charged(records),
                "best_screening_trial": best["trial"] if best else None,
                "trials": records,
            }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "ask", "report", "tell", "status"):
        p = commands.add_parser(name)
        p.add_argument("--contract", type=Path, required=True)
        if name != "validate":
            p.add_argument(
                "--storage",
                type=Path,
                required=True,
                help="Local SQLite file; one host only",
            )
        if name == "ask":
            p.add_argument("--free-memory-mib", type=float, required=True)
            p.add_argument("--estimated-memory-mib", type=float, required=True)
            p.add_argument(
                "--params",
                type=Path,
                help="Optional strict JSON of every active parameter for a seed observation",
            )
        if name in {"report", "tell"}:
            p.add_argument(
                "--trial",
                type=int,
                required=True,
                help="Adapter attempt number returned by ask",
            )
        if name == "report":
            p.add_argument("--receipt", type=Path, required=True)
        if name == "tell":
            p.add_argument("--state", choices=sorted(TERMINAL), required=True)
            p.add_argument("--reason")
    args = parser.parse_args(argv)
    try:
        contract = validate_contract(load_json(args.contract))
        if args.command == "validate":
            result = {
                "contract_digest": digest(contract),
                "metric_role": "screening_only",
            }
        else:
            controller = SearchStudy(args.storage, contract)
            if args.command == "ask":
                result = controller.ask(
                    free_memory_mib=args.free_memory_mib,
                    estimated_memory_mib=args.estimated_memory_mib,
                    params=load_json(args.params) if args.params is not None else None,
                )
            elif args.command == "report":
                result = controller.report(args.trial, load_json(args.receipt))
            elif args.command == "tell":
                result = controller.tell(
                    args.trial, state=args.state, reason=args.reason
                )
            else:
                result = controller.status()
        print(_json(result))
        return 0
    except Exception as exc:
        # A CLI/storage failure must not discard a reservation or print a
        # non-JSON traceback. API callers still receive the original exception.
        print(_json({"error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
