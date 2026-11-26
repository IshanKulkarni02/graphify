# WhatsApp connector adapter (fork branch `local-ollama-whatsapp`)

Runs graphify's analysis on the people graph of a local WhatsApp assistant, fully offline (Ollama).

```
graphify whatsapp analyze data/graph.json --out-dir data/graphify [--llm ollama|none] [--model qwen3:14b] [--profiles data/profiles]
graphify whatsapp path data/graph.json "Alice" "Bob" [--dfs] [--via-owner]
```

Input: connector `graph.json` (nodes `{id,type,label,jid?,notes,ai?}`, edges `{from,to,label,ai?,why?}`). The phone number (jid) is the identity key: two nodes with the same number become one.

Output (`--out-dir`):
- `analysis.json`: `communities` (Leiden circles; plain-language names from the local model, hub names with `--llm none`), per-node `community`/`communityName`/`degree`, per-edge `confidence`, `god_nodes`, `surprises` (links that bridge two circles, ranked by edge betweenness), confidence counts.
- `GRAPH_REPORT.md`: the same as a readable report, including ambiguous links to check.

Confidence tags: an explicit `confidence` wins. Otherwise vague labels ("relative?", "likely ...") → AMBIGUOUS; AI-made links → INFERRED, except WhatsApp group memberships → EXTRACTED; hand-made links → EXTRACTED. With `--profiles`, the local model proposes extra links from per-contact notes, each tagged EXTRACTED/INFERRED/AMBIGUOUS.

The owner node (`ishan`) is left out of clustering (it links to everyone and would merge every circle) and out of surprise scoring; path queries avoid going through the owner when another route exists.

Privacy: the only network call is to the local Ollama server (`OLLAMA_HOST`, default http://127.0.0.1:11434). Set `GRAPHIFY_NO_AUTO_REFRESH=1` so running the CLI does not rewrite installed agent skills.
