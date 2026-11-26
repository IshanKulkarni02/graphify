"""WhatsApp connector adapter (fork: local-ollama-whatsapp).

Turns the WhatsApp connector's people graph (``data/graph.json``: nodes
``{id,type,label,jid?,notes,ai?}``, edges ``{from,to,label,ai?,why?}``) into a
graphify graph and runs graphify's analysis on it, fully locally:

* Leiden community clustering (``graphify.cluster``) -> "circles" of people,
  named in plain language by a local Ollama model (or by their hub, no LLM);
* the EXTRACTED / INFERRED / AMBIGUOUS confidence tag on every link;
* god nodes (most connected people/things), surprising cross-circle links,
  a GRAPH_REPORT.md;
* BFS / DFS path queries: "how is X connected to Y".

Optionally (``--profiles``) it reads the connector's per-contact notes
(``data/profiles/<phone>.md``) and asks the local model for extra links, each
tagged with a confidence level. Nothing is sent anywhere except the local
Ollama server (default http://127.0.0.1:11434).

Identity: a person's phone number (jid digits) is the key; two nodes with the
same number are treated as one.

CLI::

    python -m graphify.whatsapp analyze data/graph.json --out-dir data/graphify \\
        [--llm ollama|none] [--model qwen3:14b] [--profiles data/profiles]
    python -m graphify.whatsapp path data/graph.json "Vijaykumar" "Bhavesh" [--dfs]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx

from graphify.cluster import cluster, label_communities_by_hub, score_all

CONFIDENCE = ("EXTRACTED", "INFERRED", "AMBIGUOUS")
_RANK = {"EXTRACTED": 3, "INFERRED": 2, "AMBIGUOUS": 1}
_WEIGHT = {"EXTRACTED": 1.0, "INFERRED": 0.8, "AMBIGUOUS": 0.4}
_VAGUE = re.compile(r"\?|\b(likely|maybe|possibly|probably|might|perhaps|seems?)\b", re.I)
_SECRET = re.compile(r"\b(otp|password|passcode|pin|cvv|ifsc|account\s*(no|number)|card\s*number|upi\s*id)\b", re.I)
OWNER_ID = "ishan"


def _digits(jid: str | None) -> str:
    return re.sub(r"\D", "", (jid or "").split("@")[0])


def edge_confidence(e: dict) -> str:
    """Map a connector edge to a graphify confidence tag.

    An explicit ``confidence`` wins. Otherwise: vague labels ("relative?",
    "likely friend") -> AMBIGUOUS; AI-made links -> INFERRED, except WhatsApp
    group memberships (read from WhatsApp itself) -> EXTRACTED; hand-made
    links -> EXTRACTED.
    """
    c = str(e.get("confidence") or "").upper()
    if c in CONFIDENCE:
        return c
    label = str(e.get("label") or "")
    if _VAGUE.search(label) or _VAGUE.search(str(e.get("why") or "")[:60]):
        return "AMBIGUOUS"
    if e.get("ai"):
        return "EXTRACTED" if label.strip().lower() == "member" else "INFERRED"
    return "EXTRACTED"


def load_connector(path: str | Path) -> tuple[nx.Graph, dict, dict[str, str]]:
    """Load the connector graph. Returns (G, raw json, alias map id->canonical id)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    G = nx.Graph()
    by_phone: dict[str, str] = {}
    alias: dict[str, str] = {}
    for n in raw.get("nodes", []):
        nid = str(n["id"])
        phone = _digits(n.get("jid"))
        if phone and phone in by_phone:  # same person twice -> one node
            alias[nid] = by_phone[phone]
            continue
        if phone:
            by_phone[phone] = nid
        G.add_node(nid, label=n.get("label") or nid, type=n.get("type") or "thing",
                   jid=n.get("jid"), notes=n.get("notes") or "", ai=bool(n.get("ai")),
                   source_file=f"whatsapp/{n.get('type') or 'thing'}")
    for e in raw.get("edges", []):
        a, b = alias.get(e["from"], e["from"]), alias.get(e["to"], e["to"])
        if a == b or a not in G or b not in G:
            continue
        conf = edge_confidence(e)
        if G.has_edge(a, b) and _RANK[G[a][b]["confidence"]] >= _RANK[conf]:
            continue
        G.add_edge(a, b, label=e.get("label") or "", confidence=conf, weight=_WEIGHT[conf],
                   why=e.get("why") or "", direction=(e["from"], e["to"]))
    return G, raw, alias


# ---------- local LLM (Ollama) ----------
def ollama_json(prompt: str, *, model: str, system: str = "", host: str | None = None,
                timeout: float = 600, max_tokens: int = 800) -> dict | None:
    host = (host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
    if not host.startswith("http"):
        host = "http://" + host
    msgs = ([{"role": "system", "content": system + ("\n/no_think" if "qwen3" in model else "")}] if system else [])
    msgs.append({"role": "user", "content": prompt})
    body = json.dumps({"model": model, "messages": msgs, "stream": False, "think": False, "format": "json",
                       "keep_alive": "30m", "options": {"temperature": 0.1, "num_predict": max_tokens, "num_ctx": 8192}}).encode()
    req = urllib.request.Request(host + "/api/chat", data=body, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        txt = json.loads(r.read().decode("utf-8")).get("message", {}).get("content", "")
    txt = re.sub(r"<think>[\s\S]*?</think>", "", txt)
    m = re.search(r"\{[\s\S]*\}", txt)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def name_communities_llm(G: nx.Graph, communities: dict[int, list[str]], model: str,
                         fallback: dict[int, str]) -> dict[int, str]:
    """One local-LLM call: a 2-4 word plain name per circle ("College friends")."""
    lines = []
    for cid, members in communities.items():
        ms = sorted(members, key=lambda n: -G.degree(n))[:12]
        desc = "; ".join(f"{G.nodes[n]['label']} ({G.nodes[n]['type']}: {str(G.nodes[n].get('notes',''))[:70]})" for n in ms)
        lines.append(f"{cid}: {desc}")
    out = ollama_json(
        "These are circles (clusters) in one person's social graph from his WhatsApp. "
        "Give each circle a short plain-English name of 2-4 words describing what ties it together "
        "(e.g. \"Family\", \"College friends\", \"Urban Web Host team\", \"Business Manager project\").\n\n"
        + "\n".join(lines) + '\n\nOutput JSON: {"names": {"<id>": "<name>", ...}}',
        model=model, system="You name clusters of a personal knowledge graph. Facts only.", max_tokens=600)
    names = dict(fallback)
    for k, v in ((out or {}).get("names") or {}).items():
        try:
            cid = int(k)
        except (TypeError, ValueError):
            continue
        v = re.sub(r"\s+", " ", str(v)).strip()[:40]
        if cid in names and v and not _SECRET.search(v):
            names[cid] = v
    return names


def profile_links(G: nx.Graph, profiles_dir: str | Path, model: str) -> int:
    """Read data/profiles/<phone>.md and let the local model propose tagged links."""
    by_phone = {_digits(d.get("jid")): n for n, d in G.nodes(data=True) if d.get("jid")}
    known = "\n".join(f"{n} | {d['type']} | {d['label']}" for n, d in G.nodes(data=True))
    added = 0
    for f in sorted(Path(profiles_dir).glob("*.md")):
        me = by_phone.get(re.sub(r"\D", "", f.stem))
        if not me:
            continue
        text = f.read_text(encoding="utf-8")[:3500]
        out = ollama_json(
            f"Node of this contact: {me}\nExisting nodes (id | type | label):\n{known}\n\nNotes about {G.nodes[me]['label']}:\n{text}\n\n"
            'List relationships between EXISTING nodes that the notes support. Tag each: "EXTRACTED" = stated in the notes, '
            '"INFERRED" = reasonable conclusion from the notes, "AMBIGUOUS" = uncertain. Output JSON: '
            '{"edges": [{"from": "id", "to": "id", "label": "1-4 words", "confidence": "EXTRACTED|INFERRED|AMBIGUOUS", "why": "short"}]} (max 8).',
            model=model, system="You extract relationships from private notes. Never output secrets (passwords, OTPs, bank numbers).")
        for e in (out or {}).get("edges") or []:
            a, b, lab = str(e.get("from", "")), str(e.get("to", "")), str(e.get("label", ""))[:40]
            conf = str(e.get("confidence", "")).upper()
            if a not in G or b not in G or a == b or not lab or conf not in CONFIDENCE or _SECRET.search(lab):
                continue
            if G.has_edge(a, b):
                continue
            G.add_edge(a, b, label=lab, confidence=conf, weight=_WEIGHT[conf], why=str(e.get("why", ""))[:200],
                       direction=(a, b), new=True)
            added += 1
    return added


# ---------- analysis ----------
def cross_circle_surprises(G: nx.Graph, node_comm: dict[str, int], top_n: int = 8) -> list[dict]:
    """Links that bridge two different circles, ranked by edge betweenness
    (how much of the graph's shortest paths run through them), weakest
    confidence first among equals (AMBIGUOUS bridges deserve a look)."""
    H = G.copy()
    if OWNER_ID in H:  # every path runs through the owner; ignore him for "surprise"
        H.remove_node(OWNER_ID)
    if H.number_of_edges() == 0:
        return []
    eb = nx.edge_betweenness_centrality(H, normalized=True)
    out = []
    for (u, v), score in eb.items():
        cu, cv = node_comm.get(u), node_comm.get(v)
        if cu is None or cv is None or cu == cv:
            continue
        d = G[u][v]
        out.append({"from": u, "to": v, "label": d.get("label", ""), "confidence": d["confidence"],
                    "why": d.get("why", ""), "score": round(score, 4), "communities": [cu, cv]})
    out.sort(key=lambda x: (-x["score"], _RANK[x["confidence"]]))
    return out[:top_n]


def god_people(G: nx.Graph, top_n: int = 10) -> list[dict]:
    deg = sorted(((n, G.degree(n)) for n in G if n != OWNER_ID), key=lambda x: (-x[1], x[0]))
    return [{"id": n, "label": G.nodes[n]["label"], "type": G.nodes[n]["type"], "degree": d} for n, d in deg[:top_n]]


def analyze(graph_path: str | Path, out_dir: str | Path, *, llm: str = "ollama", model: str = "qwen3:14b",
            profiles: str | Path | None = None, resolution: float = 1.0) -> dict:
    G, raw, alias = load_connector(graph_path)
    added = profile_links(G, profiles, model) if (profiles and llm == "ollama") else 0
    # cluster without the owner (he is linked to almost everyone and would glue all circles together)
    H = G.copy()
    if OWNER_ID in H:
        H.remove_node(OWNER_ID)
    H.remove_nodes_from([n for n in list(H) if H.degree(n) == 0])
    communities = cluster(H, resolution=resolution) if H.number_of_nodes() else {}
    labels = label_communities_by_hub(H, communities)
    if llm == "ollama" and communities:
        try:
            labels = name_communities_llm(H, communities, model, labels)
        except Exception as e:  # naming is cosmetic; keep hub names
            print(f"[graphify whatsapp] community naming failed: {e}", file=sys.stderr)
    cohesion = score_all(H, communities) if communities else {}
    node_comm = {n: cid for cid, ms in communities.items() for n in ms}
    # isolated nodes (only linked to the owner) form a "Just you" circle
    loose = [n for n in G if n not in node_comm and n != OWNER_ID]
    if loose:
        lid = max(communities, default=-1) + 1
        communities[lid] = loose
        labels[lid] = "Only linked to you"
        cohesion[lid] = 0.0
        node_comm.update({n: lid for n in loose})
    if OWNER_ID in G:
        node_comm[OWNER_ID] = -1
        labels[-1] = "You"
    surprises = cross_circle_surprises(G, node_comm)
    gods = god_people(G)
    conf_counts = {c: 0 for c in CONFIDENCE}
    for _, _, d in G.edges(data=True):
        conf_counts[d["confidence"]] += 1
    result = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": "graphify (fork local-ollama-whatsapp)",
        "model": model if llm == "ollama" else None,
        "communities": [{"id": cid, "name": labels.get(cid, f"Circle {cid}"), "size": len(ms),
                         "cohesion": round(float(cohesion.get(cid, 0.0)), 3),
                         "members": sorted(ms, key=lambda n: -G.degree(n))}
                        for cid, ms in sorted(communities.items(), key=lambda kv: -len(kv[1]))],
        "nodes": {n: {"community": node_comm.get(n), "communityName": labels.get(node_comm.get(n), ""),
                      "degree": G.degree(n)} for n in G},
        "aliases": alias,
        "edges": [{"from": d["direction"][0], "to": d["direction"][1], "label": d.get("label", ""),
                   "confidence": d["confidence"], **({"why": d["why"]} if d.get("why") else {}),
                   **({"new": True} if d.get("new") else {})} for _, _, d in G.edges(data=True)],
        "confidence": conf_counts,
        "god_nodes": gods,
        "surprises": surprises,
        "profile_links_added": added,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "GRAPH_REPORT.md").write_text(report(G, result), encoding="utf-8")
    return result


def report(G: nx.Graph, r: dict) -> str:
    lab = lambda n: G.nodes[n]["label"] if n in G else n  # noqa: E731
    total = sum(r["confidence"].values()) or 1
    lines = [f"# Graph report: {G.number_of_nodes()} nodes, {G.number_of_edges()} links",
             f"_Generated {r['generated']} by {r['tool']}, locally{(' with ' + r['model']) if r['model'] else ''}._", "",
             "## Confidence of links",
             *(f"- {c}: {k} ({round(k / total * 100)}%)" for c, k in r["confidence"].items()), "",
             "## Circles (communities)"]
    for c in r["communities"]:
        lines.append(f"- **{c['name']}** ({c['size']}, cohesion {c['cohesion']}): " + ", ".join(lab(n) for n in c["members"][:12]) + (" …" if c["size"] > 12 else ""))
    lines += ["", "## Most connected (god nodes)", *(f"- {g['label']} ({g['type']}): {g['degree']} links" for g in r["god_nodes"])]
    names = {c["id"]: c["name"] for c in r["communities"]}
    lines += ["", "## Surprising links between circles"]
    for s in r["surprises"]:
        lines.append(f"- {lab(s['from'])} → {s['label']} → {lab(s['to'])} [{s['confidence']}] bridges "
                     f"*{names.get(s['communities'][0], '?')}* and *{names.get(s['communities'][1], '?')}*" + (f" ({s['why']})" if s.get("why") else ""))
    if not r["surprises"]:
        lines.append("- none yet")
    amb = [(u, v, d) for u, v, d in G.edges(data=True) if d["confidence"] == "AMBIGUOUS"]
    lines += ["", "## Ambiguous links to check", *(f"- {lab(u)} — {d.get('label','')} — {lab(v)}" for u, v, d in amb[:15])]
    if not amb:
        lines.append("- none")
    return "\n".join(lines) + "\n"


# ---------- path queries ----------
def resolve(G: nx.Graph, q: str) -> str | None:
    q = q.strip()
    if q in G:
        return q
    d = _digits(q)
    if len(d) >= 6:
        for n, a in G.nodes(data=True):
            if a.get("jid") and _digits(a["jid"]).endswith(d[-10:]):
                return n
    ql = q.lower()
    exact = [n for n, a in G.nodes(data=True) if str(a.get("label", "")).lower() == ql]
    if exact:
        return max(exact, key=G.degree)
    part = [n for n, a in G.nodes(data=True) if ql in str(a.get("label", "")).lower()]
    if part:
        return max(part, key=G.degree)
    try:
        from rapidfuzz import process, fuzz
        labels = {n: str(a.get("label", "")) for n, a in G.nodes(data=True)}
        best = process.extractOne(q, labels, scorer=fuzz.WRatio)
        return best[2] if best and best[1] >= 80 else None
    except ImportError:
        return None


def path(graph_path: str | Path, a: str, b: str, *, dfs: bool = False, avoid_owner: bool = True,
         max_len: int = 5, limit: int = 3) -> dict:
    """How is a connected to b? BFS = shortest path(s); DFS = all simple paths up to
    max_len, best confidence first. Paths avoid the owner when another route exists."""
    G, _, _ = load_connector(graph_path)
    s, t = resolve(G, a), resolve(G, b)
    if not s or not t:
        return {"found": False, "error": f"not found: {a if not s else b}"}
    if s == t:
        return {"found": True, "paths": [], "from": G.nodes[s]["label"], "to": G.nodes[t]["label"]}

    def describe(p: list[str]) -> dict:
        hops = [{"from": G.nodes[u]["label"], "to": G.nodes[v]["label"], "label": G[u][v].get("label", ""),
                 "confidence": G[u][v]["confidence"]} for u, v in zip(p, p[1:])]
        weakest = min((h["confidence"] for h in hops), key=lambda c: _RANK[c])
        return {"length": len(hops), "weakest": weakest, "hops": hops,
                "text": " → ".join([G.nodes[p[0]]["label"]] + [f"({h['label']}) {h['to']}" for h in hops])}

    graphs = []
    if avoid_owner and OWNER_ID in G and OWNER_ID not in (s, t):
        H = G.copy(); H.remove_node(OWNER_ID); graphs.append(H)
    graphs.append(G)
    for H in graphs:
        if not nx.has_path(H, s, t):
            continue
        if dfs:
            ps = list(nx.all_simple_paths(H, s, t, cutoff=max_len))
            ps.sort(key=lambda p: (-min(_RANK[H[u][v]["confidence"]] for u, v in zip(p, p[1:])), len(p)))
        else:
            ps = list(nx.all_shortest_paths(H, s, t))
            ps.sort(key=lambda p: -min(_RANK[H[u][v]["confidence"]] for u, v in zip(p, p[1:])))
        return {"found": True, "from": G.nodes[s]["label"], "to": G.nodes[t]["label"],
                "via_owner": H is G and OWNER_ID in G and OWNER_ID not in (s, t),
                "paths": [describe(p) for p in ps[:limit]]}
    return {"found": False, "from": G.nodes[s]["label"], "to": G.nodes[t]["label"], "paths": []}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="graphify whatsapp", description="Analyze the WhatsApp connector's people graph locally.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("graph")
    a.add_argument("--out-dir", default=None)
    a.add_argument("--llm", choices=["ollama", "none"], default=os.environ.get("GRAPHIFY_WA_LLM", "ollama"))
    a.add_argument("--model", default=os.environ.get("GRAPHIFY_WA_MODEL", "qwen3:14b"))
    a.add_argument("--profiles", default=None)
    a.add_argument("--resolution", type=float, default=1.0)
    p = sub.add_parser("path")
    p.add_argument("graph"); p.add_argument("a"); p.add_argument("b")
    p.add_argument("--dfs", action="store_true")
    p.add_argument("--via-owner", action="store_true", help="allow paths through the owner even if another exists")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.cmd == "analyze":
        out = args.out_dir or str(Path(args.graph).parent / "graphify")
        r = analyze(args.graph, out, llm=args.llm, model=args.model, profiles=args.profiles, resolution=args.resolution)
        print(json.dumps({"ok": True, "out_dir": out, "communities": [{"name": c["name"], "size": c["size"]} for c in r["communities"]],
                          "confidence": r["confidence"], "surprises": len(r["surprises"]), "profile_links_added": r["profile_links_added"]}, ensure_ascii=False))
    else:
        print(json.dumps(path(args.graph, args.a, args.b, dfs=args.dfs, avoid_owner=not args.via_owner), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
