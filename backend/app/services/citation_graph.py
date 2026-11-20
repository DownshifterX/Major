import math
from typing import Dict, Any, List, Optional
import networkx as nx
from sqlalchemy.orm import Session
from .. import models

class LegalCitationGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._in_degrees: Dict[str, int] = {}
        self._max_in_degree: int = 0
        self._authority_cache_log: Dict[str, float] = {}
        self._authority_cache_linear: Dict[str, float] = {}

    def _rebuild_cache(self):
        if len(self.graph) == 0:
            self._in_degrees, self._max_in_degree = {}, 0
            self._authority_cache_log, self._authority_cache_linear = {}, {}
            return
        self._in_degrees = dict(self.graph.in_degree())
        self._max_in_degree = max(self._in_degrees.values()) if self._in_degrees else 0
        n_nodes = max(1, len(self.graph))
        log_denom = math.log(1.0 + self._max_in_degree) if self._max_in_degree > 0 else 1.0

        self._authority_cache_linear = {nid: round(deg / n_nodes, 4) for nid, deg in self._in_degrees.items()}
        self._authority_cache_log = {
            nid: (round(math.log(1.0 + deg) / log_denom, 4) if self._max_in_degree > 0 else 0.0)
            for nid, deg in self._in_degrees.items()
        }

    def build_from_db(self, db: Session):
        self.graph.clear()
        cases = db.query(
            models.HistoricalCase.id, models.HistoricalCase.case_name, models.HistoricalCase.court,
            models.HistoricalCase.year, models.HistoricalCase.citation,
            models.HistoricalCase.cites_count, models.HistoricalCase.cited_by_count
        ).all()

        for c in cases:
            self.graph.add_node(
                c.id, name=c.case_name, court=c.court, year=c.year, citation=c.citation,
                cites=c.cites_count or 0, cited_by=c.cited_by_count or 0
            )

        by_court: Dict[str, list] = {}
        for c in cases:
            by_court.setdefault(c.court or "Other", []).append(c)

        edges = []
        for court_cases in by_court.values():
            sorted_cases = sorted(court_cases, key=lambda x: (x.year or 2020), reverse=True)
            for i, c_a in enumerate(sorted_cases):
                link_count = 0
                for c_b in sorted_cases[i + 1:]:
                    if (c_b.year or 2010) < (c_a.year or 2020):
                        if (c_b.cited_by_count or 0) > 5 or link_count < 2:
                            edges.append((c_a.id, c_b.id))
                            link_count += 1
                            if link_count >= 5: break

        self.graph.add_edges_from(edges, type="cites")
        self._rebuild_cache()

    def get_case_citation_subgraph(self, case_id: str, depth: int = 1) -> Dict[str, Any]:
        if case_id not in self.graph:
            return {"nodes": [], "links": [], "metrics": {}}

        nodes_set = {case_id}
        frontier = {case_id}
        for _ in range(depth):
            next_frontier = set()
            for n in frontier:
                next_frontier |= set(self.graph.successors(n)) | set(self.graph.predecessors(n))
            nodes_set |= next_frontier
            frontier = next_frontier

        subgraph = self.graph.subgraph(nodes_set)
        nodes = [{
            "id": n, "label": d.get("name", n), "court": d.get("court"), "year": d.get("year"),
            "in_degree": self.graph.in_degree(n), "out_degree": self.graph.out_degree(n),
            "is_focus": (n == case_id),
        } for n, d in subgraph.nodes(data=True)]

        links = [{"source": u, "target": v, "type": d.get("type", "cites")} for u, v, d in subgraph.edges(data=True)]
        metrics = {
            "node_count": len(nodes), "edge_count": len(links),
            "focus_citations_count": len(list(self.graph.successors(case_id))),
            "focus_cited_by_count": len(list(self.graph.predecessors(case_id))),
        }
        return {"nodes": nodes, "links": links, "metrics": metrics}

    def get_authority_score(
        self, case_id: str, mode: str = "log", alpha: float = 1.0, anchor_score: Optional[float] = None
    ) -> float:
        if case_id not in self.graph or len(self.graph) == 0:
            return 0.0
        if not self._authority_cache_log:
            self._rebuild_cache()

        s_deg = self._authority_cache_linear.get(case_id, 0.0) if mode == "linear" else self._authority_cache_log.get(case_id, 0.0)
        if anchor_score is not None and 0.0 <= alpha < 1.0:
            clamped = max(0.0, min(1.0, float(anchor_score)))
            return round(max(0.0, min(1.0, alpha * s_deg + (1.0 - alpha) * clamped)), 4)
        return s_deg

_graph_singleton: Optional[LegalCitationGraph] = None

def get_citation_graph(db: Optional[Session] = None) -> LegalCitationGraph:
    global _graph_singleton
    if _graph_singleton is None:
        _graph_singleton = LegalCitationGraph()
        if db is not None:
            _graph_singleton.build_from_db(db)
    return _graph_singleton
