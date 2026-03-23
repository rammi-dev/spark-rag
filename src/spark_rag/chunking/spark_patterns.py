"""Detect Spark API usage and problem patterns in code.

Used in both ingestion (enrich chunks with metadata) and query pipeline
(detect patterns in user-submitted code).

Pattern detection uses regex on source text — works on any language without
needing a full AST parse. tree-sitter is used separately for chunking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class RiskLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class DetectedPattern:
    name: str
    risk: RiskLevel
    description: str
    line: int | None = None  # line number where detected (1-based)
    match_text: str = ""  # the matched code fragment


@dataclass
class SparkAPIUsage:
    api: str
    line: int | None = None


@dataclass
class PatternResult:
    """Result of analyzing a code snippet."""
    apis: list[SparkAPIUsage] = field(default_factory=list)
    patterns: list[DetectedPattern] = field(default_factory=list)

    @property
    def api_names(self) -> list[str]:
        return [a.api for a in self.apis]

    @property
    def has_problems(self) -> bool:
        return len(self.patterns) > 0

    def to_metadata(self) -> dict:
        """Convert to JSON-serializable metadata for Milvus."""
        return {
            "apis": self.api_names,
            "patterns": [
                {"name": p.name, "risk": p.risk.value, "description": p.description}
                for p in self.patterns
            ],
        }


# ── Spark API detection ──────────────────────────────────────────────

# Common Spark APIs to detect (method calls on known types)
_SPARK_API_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("SparkSession.builder", re.compile(r"SparkSession\.builder")),
    ("SparkContext", re.compile(r"SparkContext\s*\(")),
    ("DataFrame.select", re.compile(r"\.select\s*\(")),
    ("DataFrame.filter", re.compile(r"\.filter\s*\(")),
    ("DataFrame.where", re.compile(r"\.where\s*\(")),
    ("DataFrame.groupBy", re.compile(r"\.groupBy\s*\(")),
    ("DataFrame.agg", re.compile(r"\.agg\s*\(")),
    ("DataFrame.join", re.compile(r"\.join\s*\(")),
    ("DataFrame.union", re.compile(r"\.union\s*\(")),
    ("DataFrame.write", re.compile(r"\.write\b")),
    ("DataFrame.read", re.compile(r"\.read\b")),
    ("DataFrame.cache", re.compile(r"\.cache\s*\(")),
    ("DataFrame.persist", re.compile(r"\.persist\s*\(")),
    ("DataFrame.unpersist", re.compile(r"\.unpersist\s*\(")),
    ("DataFrame.collect", re.compile(r"\.collect\s*\(")),
    ("DataFrame.show", re.compile(r"\.show\s*\(")),
    ("DataFrame.toPandas", re.compile(r"\.toPandas\s*\(")),
    ("DataFrame.coalesce", re.compile(r"\.coalesce\s*\(")),
    ("DataFrame.repartition", re.compile(r"\.repartition\s*\(")),
    ("RDD.map", re.compile(r"\.map\s*\(")),
    ("RDD.flatMap", re.compile(r"\.flatMap\s*\(")),
    ("RDD.reduceByKey", re.compile(r"\.reduceByKey\s*\(")),
    ("RDD.groupByKey", re.compile(r"\.groupByKey\s*\(")),
    ("RDD.collect", re.compile(r"\.collect\s*\(")),
    ("broadcast", re.compile(r"\bbroadcast\s*\(")),
    ("udf", re.compile(r"\budf\s*\(")),
    ("window", re.compile(r"Window\.")),
    ("SparkSession.sql", re.compile(r"\.sql\s*\(")),
    ("collect_list", re.compile(r"\bcollect_list\s*\(")),
    ("collect_set", re.compile(r"\bcollect_set\s*\(")),
    ("explode", re.compile(r"\bexplode\s*\(")),
    ("pandas_udf", re.compile(r"\bpandas_udf\b")),
]


def detect_apis(code: str) -> list[SparkAPIUsage]:
    """Find Spark API calls in code."""
    found: list[SparkAPIUsage] = []
    seen: set[str] = set()
    lines = code.split("\n")

    for line_num, line in enumerate(lines, 1):
        for api_name, pattern in _SPARK_API_PATTERNS:
            if api_name not in seen and pattern.search(line):
                found.append(SparkAPIUsage(api=api_name, line=line_num))
                seen.add(api_name)

    return found


# ── Problem pattern detection ────────────────────────────────────────

@dataclass
class _PatternRule:
    name: str
    risk: RiskLevel
    description: str
    pattern: re.Pattern
    # Optional: a second pattern that must NOT be present to trigger
    # (e.g. .cache without .unpersist)
    negation: re.Pattern | None = None


_PROBLEM_RULES: list[_PatternRule] = [
    _PatternRule(
        name="collect_on_large",
        risk=RiskLevel.HIGH,
        description="collect() pulls entire dataset to driver memory",
        pattern=re.compile(r"\.collect\s*\("),
    ),
    _PatternRule(
        name="collect_list_unbounded",
        risk=RiskLevel.HIGH,
        description="collect_list() can exceed driver memory if group is large",
        pattern=re.compile(r"\bcollect_list\s*\("),
    ),
    _PatternRule(
        name="groupByKey",
        risk=RiskLevel.MEDIUM,
        description="groupByKey shuffles all values per key into memory; prefer reduceByKey/aggregateByKey",
        pattern=re.compile(r"\.groupByKey\s*\("),
    ),
    _PatternRule(
        name="broadcast_large",
        risk=RiskLevel.HIGH,
        description="broadcast() copies full dataset to every executor",
        pattern=re.compile(r"\bbroadcast\s*\("),
    ),
    _PatternRule(
        name="coalesce_one",
        risk=RiskLevel.MEDIUM,
        description="coalesce(1) forces all data into a single partition",
        pattern=re.compile(r"\.coalesce\s*\(\s*1\s*\)"),
    ),
    _PatternRule(
        name="toPandas_large",
        risk=RiskLevel.HIGH,
        description="toPandas() materializes entire DataFrame in driver memory",
        pattern=re.compile(r"\.toPandas\s*\("),
    ),
    _PatternRule(
        name="cache_no_unpersist",
        risk=RiskLevel.LOW,
        description="cache()/persist() without unpersist() leaks memory over long jobs",
        pattern=re.compile(r"\.(?:cache|persist)\s*\("),
        negation=re.compile(r"\.unpersist\s*\("),
    ),
    _PatternRule(
        name="repartition_to_one",
        risk=RiskLevel.MEDIUM,
        description="repartition(1) forces expensive full shuffle into single partition",
        pattern=re.compile(r"\.repartition\s*\(\s*1\s*\)"),
    ),
    _PatternRule(
        name="udf_over_builtin",
        risk=RiskLevel.LOW,
        description="UDFs prevent Catalyst optimization; prefer built-in functions when possible",
        pattern=re.compile(r"\budf\s*\("),
    ),
    _PatternRule(
        name="collect_in_loop",
        risk=RiskLevel.HIGH,
        description="collect() inside a loop creates repeated driver-side materialization",
        pattern=re.compile(r"for\s.*\n.*\.collect\s*\(", re.MULTILINE),
    ),
    _PatternRule(
        name="crossjoin",
        risk=RiskLevel.HIGH,
        description="Cross join produces cartesian product — O(n*m) rows",
        pattern=re.compile(r"\.crossJoin\s*\("),
    ),
]


def detect_patterns(code: str) -> list[DetectedPattern]:
    """Find problem patterns in code."""
    found: list[DetectedPattern] = []
    lines = code.split("\n")

    for rule in _PROBLEM_RULES:
        # Check negation first (whole-file check)
        if rule.negation and rule.negation.search(code):
            continue

        # Search line by line for single-line patterns
        if not (rule.pattern.flags & re.MULTILINE):
            for line_num, line in enumerate(lines, 1):
                m = rule.pattern.search(line)
                if m:
                    found.append(DetectedPattern(
                        name=rule.name,
                        risk=rule.risk,
                        description=rule.description,
                        line=line_num,
                        match_text=line.strip(),
                    ))
                    break  # one match per rule
        else:
            # Multiline pattern: search whole code
            m = rule.pattern.search(code)
            if m:
                # Find line number of match start
                line_num = code[:m.start()].count("\n") + 1
                found.append(DetectedPattern(
                    name=rule.name,
                    risk=rule.risk,
                    description=rule.description,
                    line=line_num,
                    match_text=m.group().strip(),
                ))

    return found


def analyze(code: str) -> PatternResult:
    """Full analysis: detect APIs and problem patterns."""
    return PatternResult(
        apis=detect_apis(code),
        patterns=detect_patterns(code),
    )


# ── Log/error analysis ───────────────────────────────────────────────

_ERROR_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("OutOfMemoryError", re.compile(r"java\.lang\.OutOfMemoryError")),
    ("StackOverflowError", re.compile(r"java\.lang\.StackOverflowError")),
    ("SparkException", re.compile(r"org\.apache\.spark\.SparkException")),
    ("AnalysisException", re.compile(r"org\.apache\.spark\.sql\.AnalysisException")),
    ("IllegalArgumentException", re.compile(r"java\.lang\.IllegalArgumentException")),
    ("ClassNotFoundException", re.compile(r"java\.lang\.ClassNotFoundException")),
    ("NullPointerException", re.compile(r"java\.lang\.NullPointerException")),
    ("FileNotFoundException", re.compile(r"java\.io\.FileNotFoundException")),
    ("TimeoutException", re.compile(r"TimeoutException")),
    ("FetchFailedException", re.compile(r"FetchFailedException")),
    ("ShuffleMapTask", re.compile(r"ShuffleMapTask")),
    ("TaskKilled", re.compile(r"TaskKilled")),
    ("ExecutorLostFailure", re.compile(r"ExecutorLostFailure")),
    ("MetadataFetchFailedException", re.compile(r"MetadataFetchFailedException")),
]


def extract_error_types(text: str) -> list[str]:
    """Extract Spark/Java exception types from log text or stacktrace."""
    found: list[str] = []
    for name, pattern in _ERROR_PATTERNS:
        if pattern.search(text):
            found.append(name)
    return found


def is_stacktrace(text: str) -> bool:
    """Heuristic: does this text look like a Java/Spark stacktrace?"""
    indicators = [
        re.compile(r"^\s+at\s+\w+", re.MULTILINE),  # "at org.apache..."
        re.compile(r"Caused by:"),
        re.compile(r"Exception in thread"),
        re.compile(r"Traceback \(most recent call last\)"),  # Python
    ]
    matches = sum(1 for p in indicators if p.search(text))
    return matches >= 1
