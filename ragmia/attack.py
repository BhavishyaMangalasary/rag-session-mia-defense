"""Step 2: the MEntA attack (Nguyen et al., USENIX Security 2026).

Part 1 (this piece): write the attacker's queries.
  For each target document, an offline LLM (QUERY_MODEL, not the RAG's generator):
    - writes a short topic phrase (the "summary"), and
    - writes NUM_QUERIES specific questions spread across the document (paper's Figure 13 prompt).
  The final query sent to the RAG is: "I am inquiring about <summary>. <question>"
  (summary-prepending, borrowed from IA, helps the retriever find the target document).
"""
import re

from ragmia import config as C
from ragmia.rag import ollama_chat

# Paper, Appendix A, Figure 13 (query generation prompt template).
QUERY_PROMPT = """Given the document below, generate {n} highly specific questions that can be answered by this document.

Document: {doc}

Requirements:
- Generate exactly {n} different questions
- Critical: The set of questions must cover all different aspects/sections of the document.
- Distribution: Do not focus all questions on a single fact. If the text has a beginning, middle, and end, or multiple distinct points, ensure the {n} questions are distributed across these different parts.
- Each question should require specific information from the document
- Focus on unique details, specific facts, or specific combinations of information
- Make each query different by using different phrasing, focusing on different aspects, varying question structure
- Do not mention "the document", "the text", "this passage", or similar references
- Do not add meta-preambles like "Here is a question:", "Question:", or "Query:"
- If the text uses any abbreviations or acronyms, use the same forms in your questions
- Avoid mentioning 'the study' or any references to the passage itself

Output format:
QUERY_1: [first question here]
QUERY_2: [second question here]
QUERY_3: [third question here]
...

Generate {n} queries now:"""

# The paper does not print its summary prompt; this one produces the short topic phrase
# shown in its Table 2 example ("I am inquiring about <topic>. <question>").
SUMMARY_PROMPT = """Describe the main topic of the document below as one short noun phrase (at most 15 words).
It must complete the sentence "I am inquiring about ...". Output only the phrase, with no final period.

Document: {doc}"""


def doc_text(doc):
    return f"{doc['title']}. {doc['text']}".strip(". ")


def parse_queries(raw):
    """Pull the questions out of 'QUERY_1: ... QUERY_2: ...' output (tolerates markdown and numbering)."""
    questions = []
    for part in re.split(r"QUERY_\d+\s*:", raw)[1:]:
        q = re.sub(r"\s+", " ", part).strip()
        q = re.sub(r"(\s*\*+|\s+\d+[.)])+$", "", q).strip(" *")   # trailing '**' or next item's '2.'
        if len(q) > 1 and q[0] + q[-1] in ('[]', '""'):
            q = q[1:-1].strip()
        if q:
            questions.append(q)
    return questions


def clean_summary(raw):
    s = raw.strip().splitlines()[0].strip().strip('"').rstrip(".")
    return re.sub(r"^(i am inquiring about)\s+", "", s, flags=re.I)


def write_queries(doc, n=C.NUM_QUERIES):
    """Return (summary, [n questions]). Retries with some randomness if the model returns too few."""
    text = doc_text(doc)
    summary = clean_summary(ollama_chat(
        [{"role": "user", "content": SUMMARY_PROMPT.format(doc=text)}],
        model=C.QUERY_MODEL, temperature=0.0, max_tokens=40))
    questions = []
    for temperature in (0.0, 0.7, 0.7):
        raw = ollama_chat([{"role": "user", "content": QUERY_PROMPT.format(n=n, doc=text)}],
                          model=C.QUERY_MODEL, temperature=temperature, max_tokens=600)
        questions = parse_queries(raw)
        if len(questions) >= n:
            break
    if len(questions) < n:
        raise RuntimeError(f"Only got {len(questions)} questions for this document")
    return summary, questions[:n]


def attack_query(summary, question):
    return f"I am inquiring about {summary}. {question}"