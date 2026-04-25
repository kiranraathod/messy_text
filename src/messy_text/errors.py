"""Operational errors surfaced by the classifier runtime."""

from __future__ import annotations


class StageClassifierOperationalError(RuntimeError):
    """Base class for runtime failures unrelated to the text content itself."""

    error_type = "operational_error"

    def to_dict(self) -> dict[str, str]:
        return {
            "error": str(self),
            "error_type": self.error_type,
        }


class ConfigurationError(StageClassifierOperationalError):
    """Raised when runtime configuration is missing or invalid."""

    error_type = "configuration_error"


class InputValidationError(StageClassifierOperationalError):
    """Raised when input data is invalid for classification."""

    error_type = "input_validation_error"


class ProviderError(StageClassifierOperationalError):
    """Raised when the upstream LLM provider request fails."""

    error_type = "provider_error"


class ResponseFormatError(StageClassifierOperationalError):
    """Raised when the upstream provider returns an invalid payload."""

    error_type = "response_format_error"
