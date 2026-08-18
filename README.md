# kgcontext

Knowledge Graph middleware for LLM document analysis.

Drop-in Python library that transforms documents into a typed knowledge graph
before sending context to an LLM — preventing context bottlenecks at scale.

## Usage

```python
from kgcontext import KnowledgeGraph

kg = KnowledgeGraph()
kg.ingest(["doc.pdf", "doc.docx"])
answer = kg.query("What are the key requirements?")
print(answer.content)
kg.save("session.json")
```

MIT License.