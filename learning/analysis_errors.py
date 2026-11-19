"""Actionable AI failures without exposing credentials or request/response bodies."""

from datetime import datetime, timezone
from importlib.metadata import version
import json
from uuid import uuid4

from openai import (
    APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError,
    BadRequestError, ContentFilterFinishReasonError, LengthFinishReasonError,
    NotFoundError, PermissionDeniedError, RateLimitError,
)
from pydantic import ValidationError

from market_data.sec import SecDataError


class AnalysisGenerationError(SecDataError):
    def __init__(self, message, details):
        super().__init__(message)
        self.details = details


def explain_api_error(error):
    """Use exception types and allow-listed error codes, never raw API messages."""
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        body = body.get("error", body)
    code = body.get("code") if isinstance(body, dict) else None
    if isinstance(error, LengthFinishReasonError):
        return "Vastaus katkesi pituusrajaan ennen analyysin valmistumista. Kutsu onnistui palveluun asti, mutta keskeneräistä vastausta ei näytetä. Voit yrittää uudelleen painikkeesta."
    if isinstance(error, APITimeoutError):
        return "Tekoälypalvelun vastausta ei saatu määräajassa. Voit yrittää myöhemmin uudelleen painikkeesta."
    if isinstance(error, AuthenticationError):
        return "OpenAI hylkäsi API-avaimen. Tarkista sovelluksen käyttämä OPENAI_API_KEY-asetus."
    if isinstance(error, RateLimitError):
        if code == "insufficient_quota":
            return "OpenAI API -tilin käyttövara tai laskutusraja ei riitä analyysiin. Tarkista API-tilin laskutus ja käyttöraja."
        return "OpenAI-palvelun kutsuraja tuli vastaan. Odota hetki ja yritä uudelleen painikkeesta."
    if isinstance(error, (PermissionDeniedError, NotFoundError)):
        return "OpenAI-tilillä ei ole pääsyä valittuun malliin tai mallia ei löydy. Tarkista mallin käyttöoikeus API-tilillä."
    if isinstance(error, APIConnectionError):
        return "Sovellus ei saanut yhteyttä OpenAI-palveluun. Tarkista verkkoyhteys ja kokeile uudelleen painikkeesta."
    if isinstance(error, BadRequestError):
        return "OpenAI hylkäsi analyysipyynnön asetukset. Virheen tekniset tiedot auttavat korjaamaan pyynnön; saman pyynnön toistaminen ei välttämättä auta."
    if isinstance(error, (ValidationError, ValueError)):
        return "Tekoälypalvelu palautti vastauksen, joka ei vastannut analyysin vaadittua rakennetta. Virheellistä vastausta ei näytetä."
    if isinstance(error, ContentFilterFinishReasonError):
        return "Tekoälypalvelu keskeytti vastauksen sisältösuodatukseen. Analyysiä ei tallennettu valmiina."
    if isinstance(error, AttributeError):
        return "Sovelluksen OpenAI-kirjasto ei tue analyysikutsua. Käynnistä sovellus käyttöohjeen Python-ympäristössä."
    if isinstance(error, APIStatusError):
        return "OpenAI-palvelu palautti palvelinvirheen. Voit yrittää myöhemmin uudelleen painikkeesta."
    return "Tekoälyanalyysi keskeytyi tekniseen virheeseen. Katso virheen tekniset tiedot."


def record_api_error(company, packet, error, *, model, output_limit):
    details = {
        "reference": uuid4().hex[:12], "created_at": datetime.now(timezone.utc).isoformat(),
        "ticker": company.ticker, "fiscal_year": packet["fiscal_year"],
        "model": model, "openai_version": version("openai"),
        "error_type": type(error).__name__, "max_output_tokens": output_limit,
    }
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        details["http_status"] = status
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        body = body.get("error", body)
    if isinstance(body, dict):
        if body.get("code") in {"invalid_api_key", "insufficient_quota", "rate_limit_exceeded", "model_not_found",
                                "context_length_exceeded", "unsupported_parameter", "unsupported_value", "invalid_json_schema"}:
            details["error_code"] = body["code"]
        if body.get("param") in {"response_format", "reasoning_effort", "max_completion_tokens", "messages", "model", "temperature"}:
            details["parameter"] = body["param"]
    completion = getattr(error, "completion", None)
    usage = getattr(completion, "usage", None)
    if usage is not None:
        details["input_tokens"] = usage.prompt_tokens
        details["output_tokens"] = usage.completion_tokens
    try:
        directory = company.data_dir / "analyses" / "errors"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{details['reference']}.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    except OSError:
        # An unwritable diagnostics directory must not hide the original failure.
        pass
    return AnalysisGenerationError(explain_api_error(error) + " Automaattista uusintaa ei tehdä.", details)
