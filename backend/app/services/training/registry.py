"""Maps provider name -> TrainingProvider instance. The application calls
`get_provider(job.provider)` and never imports LocalTrainingProvider or
KaggleTrainingProvider directly outside this module — adding a provider is
one new module plus one line here."""
from __future__ import annotations

from app.services.training.kaggle_provider import KaggleTrainingProvider
from app.services.training.local_provider import LocalTrainingProvider
from app.services.training.modal_provider import ModalTrainingProvider
from app.services.training.provider import TrainingProvider

_PROVIDERS: dict[str, TrainingProvider] = {
    "LOCAL": LocalTrainingProvider(),
    "KAGGLE": KaggleTrainingProvider(),
    "MODAL": ModalTrainingProvider(),
}


def get_available_providers() -> list[str]:
    """Providers ready to actually run right now. KAGGLE and MODAL are
    omitted whenever credentials aren't set — this is how optional
    integrations are enforced in practice."""
    return [name for name, provider in _PROVIDERS.items() if provider.is_configured]


def get_provider(name: str) -> TrainingProvider:
    provider = _PROVIDERS.get(name)
    if provider is None or not provider.is_configured:
        raise ValueError(f"Training provider {name!r} is not available")
    return provider
