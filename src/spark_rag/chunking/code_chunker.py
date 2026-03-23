"""AST-aware code chunking using tree-sitter.

Splits source files into semantic chunks:
- method/function bodies
- class summary (name + method signatures)
- import blocks

Supports Scala, Java, Python. Each chunk includes metadata for Milvus.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import tree_sitter
import tree_sitter_java
import tree_sitter_python
import tree_sitter_scala

from spark_rag.chunking.spark_patterns import analyze as analyze_patterns

logger = logging.getLogger(__name__)

Language = Literal["scala", "java", "python"]
ChunkType = Literal["method", "class_summary", "imports"]

# Max tokens (rough: 1 token ≈ 4 chars) before splitting a chunk
MAX_CHUNK_CHARS = 2048  # ~512 tokens

# tree-sitter language objects
_LANGUAGES: dict[Language, tree_sitter.Language] = {
    "scala": tree_sitter.Language(tree_sitter_scala.language()),
    "java": tree_sitter.Language(tree_sitter_java.language()),
    "python": tree_sitter.Language(tree_sitter_python.language()),
}

# Node types that represent functions/methods per language
_METHOD_TYPES: dict[Language, set[str]] = {
    "scala": {"function_definition", "function_declaration"},
    "java": {"method_declaration", "constructor_declaration"},
    "python": {"function_definition"},
}

# Node types that represent classes/objects
_CLASS_TYPES: dict[Language, set[str]] = {
    "scala": {"class_definition", "object_definition", "trait_definition"},
    "java": {"class_declaration", "interface_declaration", "enum_declaration"},
    "python": {"class_definition"},
}

# Node types for imports
_IMPORT_TYPES: dict[Language, set[str]] = {
    "scala": {"import_declaration", "package_clause"},
    "java": {"import_declaration", "package_declaration"},
    "python": {"import_statement", "import_from_statement"},
}


@dataclass
class CodeChunk:
    content: str
    chunk_type: ChunkType
    language: Language
    file_path: str
    qualified_name: str = ""
    signature: str = ""
    spark_apis: dict = field(default_factory=lambda: {"apis": []})
    problem_indicators: dict = field(default_factory=lambda: {"patterns": []})
    start_line: int = 0
    end_line: int = 0

    def to_milvus_data(self, spark_version: str, embedding: list[float]) -> dict:
        """Convert to dict ready for Milvus insert."""
        return {
            "embedding": embedding,
            "content": self.content[:65535],  # VARCHAR limit
            "spark_version": spark_version,
            "file_path": self.file_path,
            "language": self.language,
            "chunk_type": self.chunk_type,
            "qualified_name": self.qualified_name[:512],
            "signature": self.signature[:1024],
            "spark_apis": self.spark_apis,
            "problem_indicators": self.problem_indicators,
        }


def _get_parser(language: Language) -> tree_sitter.Parser:
    return tree_sitter.Parser(_LANGUAGES[language])


def _node_text(node: tree_sitter.Node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _find_name(node: tree_sitter.Node, source: bytes) -> str:
    """Extract the identifier name from a class/method node."""
    for child in node.children:
        if child.type == "identifier" or child.type == "name":
            return _node_text(child, source)
    return ""


def _find_signature(node: tree_sitter.Node, source: bytes, language: Language) -> str:
    """Extract method/function signature (first line or up to body start)."""
    text = _node_text(node, source)
    # Take everything up to the first { or : (body start)
    for delim in ["{", ":", "=>"]:
        idx = text.find(delim)
        if idx != -1:
            sig = text[:idx].strip()
            if sig:
                return sig
    # Fallback: first line
    return text.split("\n")[0].strip()


def _split_long_chunk(chunk: CodeChunk) -> list[CodeChunk]:
    """Split a chunk that exceeds MAX_CHUNK_CHARS into smaller pieces."""
    if len(chunk.content) <= MAX_CHUNK_CHARS:
        return [chunk]

    lines = chunk.content.split("\n")
    parts: list[CodeChunk] = []
    current_lines: list[str] = []
    current_len = 0

    for i, line in enumerate(lines):
        if current_len + len(line) > MAX_CHUNK_CHARS and current_lines:
            parts.append(CodeChunk(
                content="\n".join(current_lines),
                chunk_type=chunk.chunk_type,
                language=chunk.language,
                file_path=chunk.file_path,
                qualified_name=f"{chunk.qualified_name}_part{len(parts)}",
                signature=chunk.signature if len(parts) == 0 else "",
                spark_apis=chunk.spark_apis if len(parts) == 0 else {"apis": []},
                problem_indicators=chunk.problem_indicators if len(parts) == 0 else {"patterns": []},
                start_line=chunk.start_line + (sum(len(p.content.split("\n")) for p in parts)),
            ))
            current_lines = []
            current_len = 0
        current_lines.append(line)
        current_len += len(line) + 1

    if current_lines:
        parts.append(CodeChunk(
            content="\n".join(current_lines),
            chunk_type=chunk.chunk_type,
            language=chunk.language,
            file_path=chunk.file_path,
            qualified_name=f"{chunk.qualified_name}_part{len(parts)}" if len(parts) > 0 else chunk.qualified_name,
            signature="",
            spark_apis={"apis": []},
            problem_indicators={"patterns": []},
            start_line=chunk.start_line + (sum(len(p.content.split("\n")) for p in parts)),
        ))

    return parts


def _extract_methods_from_class(
    class_node: tree_sitter.Node,
    source: bytes,
    language: Language,
    file_path: str,
    class_name: str,
) -> list[CodeChunk]:
    """Extract method chunks from a class body."""
    chunks: list[CodeChunk] = []
    method_types = _METHOD_TYPES[language]

    def walk(node: tree_sitter.Node):
        if node.type in method_types:
            method_name = _find_name(node, source)
            content = _node_text(node, source)
            qualified = f"{class_name}.{method_name}" if class_name else method_name
            analysis = analyze_patterns(content)

            chunk = CodeChunk(
                content=content,
                chunk_type="method",
                language=language,
                file_path=file_path,
                qualified_name=qualified,
                signature=_find_signature(node, source, language),
                spark_apis={"apis": analysis.api_names},
                problem_indicators={"patterns": [
                    {"name": p.name, "risk": p.risk.value}
                    for p in analysis.patterns
                ]},
                start_line=node.start_point.row + 1,
                end_line=node.end_point.row + 1,
            )
            chunks.extend(_split_long_chunk(chunk))
        else:
            for child in node.children:
                walk(child)

    walk(class_node)
    return chunks


def _build_class_summary(
    class_node: tree_sitter.Node,
    source: bytes,
    language: Language,
    file_path: str,
) -> CodeChunk | None:
    """Build a class summary chunk: class name + all method signatures."""
    class_name = _find_name(class_node, source)
    if not class_name:
        return None

    method_types = _METHOD_TYPES[language]
    signatures: list[str] = []

    def collect_sigs(node: tree_sitter.Node):
        if node.type in method_types:
            sig = _find_signature(node, source, language)
            if sig:
                signatures.append(sig)
        else:
            for child in node.children:
                collect_sigs(child)

    collect_sigs(class_node)

    if not signatures:
        return None

    # Build summary content
    class_keyword = class_node.type.split("_")[0]  # "class", "object", "trait", etc.
    content = f"{class_keyword} {class_name}\n\nMethods:\n" + "\n".join(f"  {s}" for s in signatures)

    return CodeChunk(
        content=content,
        chunk_type="class_summary",
        language=language,
        file_path=file_path,
        qualified_name=class_name,
        signature=f"{class_keyword} {class_name}",
        start_line=class_node.start_point.row + 1,
        end_line=class_node.end_point.row + 1,
    )


def chunk_file(
    source_code: str,
    file_path: str,
    language: Language,
) -> list[CodeChunk]:
    """Parse a source file and return semantic chunks.

    Returns:
        List of CodeChunks: imports block, class summaries, method bodies.
    """
    parser = _get_parser(language)
    source = source_code.encode("utf-8")
    tree = parser.parse(source)
    root = tree.root_node

    chunks: list[CodeChunk] = []
    import_types = _IMPORT_TYPES[language]
    class_types = _CLASS_TYPES[language]
    method_types = _METHOD_TYPES[language]

    # 1. Collect imports
    import_lines: list[str] = []
    for child in root.children:
        if child.type in import_types:
            import_lines.append(_node_text(child, source))

    if import_lines:
        chunks.append(CodeChunk(
            content="\n".join(import_lines),
            chunk_type="imports",
            language=language,
            file_path=file_path,
            qualified_name=f"{file_path}:imports",
            start_line=1,
        ))

    # 2. Process classes and top-level methods
    for child in root.children:
        if child.type in class_types:
            class_name = _find_name(child, source)

            # Class summary
            summary = _build_class_summary(child, source, language, file_path)
            if summary:
                chunks.append(summary)

            # Methods within class
            method_chunks = _extract_methods_from_class(
                child, source, language, file_path, class_name,
            )
            chunks.extend(method_chunks)

        elif child.type in method_types:
            # Top-level function (Python, Scala)
            method_name = _find_name(child, source)
            content = _node_text(child, source)
            analysis = analyze_patterns(content)

            chunk = CodeChunk(
                content=content,
                chunk_type="method",
                language=language,
                file_path=file_path,
                qualified_name=method_name,
                signature=_find_signature(child, source, language),
                spark_apis={"apis": analysis.api_names},
                problem_indicators={"patterns": [
                    {"name": p.name, "risk": p.risk.value}
                    for p in analysis.patterns
                ]},
                start_line=child.start_point.row + 1,
                end_line=child.end_point.row + 1,
            )
            chunks.extend(_split_long_chunk(chunk))

    return chunks


def detect_language(file_path: str) -> Language | None:
    """Detect language from file extension."""
    if file_path.endswith(".scala"):
        return "scala"
    elif file_path.endswith(".java"):
        return "java"
    elif file_path.endswith(".py"):
        return "python"
    return None
