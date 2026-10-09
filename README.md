# Assistant intelligent agentique multimodal

## Demarrage rapide
```bash
python -m venv venv && source venv/bin/activate   # Windows : venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
docker compose up -d          # PostgreSQL (schema auto-cree) + Qdrant
pytest                        # tests du contrat d'API
uvicorn api.main:app --reload # Swagger : http://localhost:8000/docs
```

## Structure
- `ingestion/` extraction PDF/texte/video   - `rag/` chunking, embeddings, retrieval
- `agent/` outils + graphe LangGraph        - `api/` FastAPI (routes, schemas)
- `ui/` Streamlit                           - `db/schema.sql` schema PostgreSQL
- `config/settings.py` configuration centralisee (lit `.env`)

## Donnees
- PostgreSQL : documents, chunks, tasks, conversations, messages.
- Qdrant : un point par chunk, payload = chunk_id, document_id, source_type, page_number, start_time, end_time.
- Suppression d'un document : la cascade SQL nettoie PostgreSQL ; le code doit aussi supprimer
  les points Qdrant par filtre `document_id`.
