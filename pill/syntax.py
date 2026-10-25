from typing import Dict
JAVA_PAIR_STUB = '\npackage org.javatuples;\n\nimport java.util.Objects;\n\npublic final class Pair<A, B> {\n    private final A value0;\n    private final B value1;\n\n    private Pair(A value0, B value1) {\n        this.value0 = value0;\n        this.value1 = value1;\n    }\n\n    public static <A, B> Pair<A, B> with(A value0, B value1) {\n        return new Pair<A, B>(value0, value1);\n    }\n\n    public A getValue0() {\n        return value0;\n    }\n\n    public B getValue1() {\n        return value1;\n    }\n\n    @Override\n    public boolean equals(Object other) {\n        if (!(other instanceof Pair<?, ?>)) {\n            return false;\n        }\n        Pair<?, ?> pair = (Pair<?, ?>) other;\n        return Objects.equals(value0, pair.value0) && Objects.equals(value1, pair.value1);\n    }\n\n    @Override\n    public int hashCode() {\n        return Objects.hash(value0, value1);\n    }\n\n    @Override\n    public String toString() {\n        return "(" + value0 + ", " + value1 + ")";\n    }\n}\n'.strip()

def problem_language(problem: Dict) -> str:
    metadata = problem.get("metadata")
    if isinstance(metadata, dict):
        language = metadata.get("language")
        if isinstance(language, str) and language.strip():
            return language.strip().lower()
    return str(problem.get("language") or "python").strip().lower()


def build_external_program(problem: Dict, completion: str) -> str:
    program = problem["prompt"] + completion + problem["suffix"]
    test = problem.get("test", "")
    if isinstance(test, str) and test.strip() and test not in program:
        program += "\n" + test
    return program
