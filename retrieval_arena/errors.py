class RetrievalAuditError(Exception):
    """Base class for expected Retrieval Audit Framework failures."""


class ValidationError(RetrievalAuditError):
    """Raised when a config, dataset, or test output violates the contract."""