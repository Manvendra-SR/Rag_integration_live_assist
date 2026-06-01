from __future__ import annotations


class SarvamTranslationProvider:
    """Placeholder boundary.

    The MVP uses Sarvam streaming ASR in translate mode, so translation happens
    inside the ASR provider. Keep this adapter as the future swap point if
    translation becomes a separate step.
    """

    provider_name = "sarvam"

