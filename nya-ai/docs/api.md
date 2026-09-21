# Nya AI — API Reference

Base URL: `http://localhost:8000`

## Case Management
- `POST /cases`: Create a new case profile.
- `POST /cases/from-url`: Acquire, clean, and extract case directly from an Indian Kanoon or public judgment URL.
- `GET /cases`: List all active cases.
- `GET /cases/{id}`: Retrieve full case details.
- `PATCH /cases/{id}`: Update case details.
- `DELETE /cases/{id}`: Delete a case.
- `GET /cases/{id}/dashboard-stats`: Aggregate metrics (documents, chunks, analyses).

## Document Ingestion & Acquisition
- `POST /cases/{id}/documents`: Upload document (.pdf, .docx, .txt, .md).
- `GET /cases/{id}/documents`: List uploaded case documents.
- `DELETE /cases/{id}/documents/{doc_id}`: Remove document.
- `POST /cases/{id}/index`: Re-chunk and re-embed case documents.

## Retrieval & Analysis
- `POST /cases/{id}/analyze`: Run full analysis pipeline (similar cases, provisions, arguments, evidence gaps, contradictions).
- `GET /cases/{id}/similar-cases`: Retrieve and multi-signal rerank precedents.
- `GET /cases/{id}/citation-graph`: Precedent citation network and authority metrics.
- `GET /cases/{id}/legal-provisions`: Extract statutory provisions and applicable sections.
- `GET /cases/{id}/prosecution-defence`: Prosecution vs Defence argument breakdown.
- `GET /cases/{id}/evidence-gaps`: Evidence present vs potential missing verification items.
- `GET /cases/{id}/contradictions`: Contradiction and timeline discrepancy detection.
- `POST /cases/{id}/compare`: Compare current case against a selected historical precedent.
- `GET /cases/{id}/brief`: Generate grounded pre-hearing brief.
- `GET /cases/{id}/brief/export`: Export pre-hearing brief as PDF.

## Search & Precedents
- `GET /api/search?q={query}`: Lexical and semantic search across historical judgments.
- `GET /historical-cases`: List precedent corpus.
- `GET /historical-cases/{id}`: View individual precedent judgment.
- `GET /api/health` or `GET /health`: Health and engine status.
