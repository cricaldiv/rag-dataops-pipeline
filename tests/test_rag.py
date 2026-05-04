from app.ingest import load_documents


def test_test_documents_are_valid():
    docs = load_documents("data/test/sample_docs.txt")
    assert len(docs) >= 2
    assert all(len(doc) > 20 for doc in docs)


def test_prod_documents_are_valid():
    docs = load_documents("data/prod/approved_docs.txt")
    assert len(docs) >= 2
    assert all(len(doc) > 20 for doc in docs)
