from langchain_core.documents import Document

from agent.utils import format_documents_as_string


def test_format_documents_as_string():
    docs = [
        Document(
            page_content="hello world",
            metadata={
                "relevance_score": 0.9,
                "doc_id": "doc1",
            },
        ),
        Document(
            page_content="foo bar",
            metadata={
                "relevance_score": 0.8,
                "doc_id": "doc2",
            },
        ),
        Document(
            page_content="lorem ipsum",
            metadata={
                "doc_id": "doc3",
                "relevance_score": 0.5,
            },
        ),
    ]

    docs_string = format_documents_as_string(docs)
    expected_string = "[doc1 | 0.9]\nhello world\n\n[doc2 | 0.8]\nfoo bar\n\n[doc3 | 0.5]\nlorem ipsum"
    assert docs_string == expected_string


def test_format_documents_as_string_empty():
    docs = []
    docs_string = format_documents_as_string(docs)
    assert docs_string == ""  # Should return an empty string for no documents
