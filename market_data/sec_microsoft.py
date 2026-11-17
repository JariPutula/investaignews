"""Compatibility entry points for the original Microsoft exercise and CLI."""

from market_data import sec
from market_data.companies import get_company
from market_data.financial_metrics import percentage_change, revenue_cagr

MICROSOFT = get_company("MSFT")
CIK = MICROSOFT.cik
SOURCE_URL = MICROSOFT.source_url
DEFAULT_DATA_DIR = MICROSOFT.data_dir
METRICS = sec.CORE_METRICS
SecDataError = sec.SecDataError


def extract_annual_metrics(payload, *, years=4, as_of=None):
    return sec.extract_annual_metrics(payload, company=MICROSOFT, years=years, as_of=as_of)


def download_companyfacts(user_agent):
    return sec.download_companyfacts(user_agent, company=MICROSOFT)


def build_report(*, data_dir=DEFAULT_DATA_DIR, user_agent="", refresh=False, offline=False, as_of=None):
    return sec.build_report(company=MICROSOFT, data_dir=data_dir, user_agent=user_agent, refresh=refresh, offline=offline, as_of=as_of)


def load_report(data_dir=DEFAULT_DATA_DIR):
    return sec.load_report(company=MICROSOFT, data_dir=data_dir)
