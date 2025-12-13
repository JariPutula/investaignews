"""
News Sentinel Agent - A modular news analysis system for investment dashboards.
"""

from news_sentinel.agent import NewsSentinelAgent
from news_sentinel.models import (
    NewsArticle,
    ImpactfulHeadline,
    TickerSentiment,
    NewsAnalysisResult,
)

__all__ = [
    "NewsSentinelAgent",
    "NewsArticle",
    "ImpactfulHeadline",
    "TickerSentiment",
    "NewsAnalysisResult",
]

