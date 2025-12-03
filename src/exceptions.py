"""
Custom exceptions for the RAG system.

Provides a hierarchy of exceptions to distinguish between:
- Legitimate empty results (RAGNotFoundError)
- Data validation/corruption issues (RAGDataError)
- System failures (RAGSystemError)
"""


class RAGError(Exception):
    """Base exception for all RAG-related errors."""
    pass


class RAGDataError(RAGError):
    """
    Data validation or corruption issues.

    Examples:
    - Missing required fields
    - Malformed JSON
    - Schema validation failures
    - Index mismatches
    """
    pass


class RAGSystemError(RAGError):
    """
    System-level failures that require attention.

    Examples:
    - ChromaDB connection failures
    - Embedding model failures
    - File system errors
    - Out of memory
    """
    pass


class RAGNotFoundError(RAGError):
    """
    Legitimate empty results (not an error condition).

    Examples:
    - No documents match query
    - Entity ID not in collection
    - Empty search results

    Note: This should be caught and handled gracefully,
    not propagated as an error to end users.
    """
    pass


class RAGConfigError(RAGError):
    """
    Configuration-related errors.

    Examples:
    - Missing config file
    - Invalid config values
    - Required parameters not set
    """
    pass
