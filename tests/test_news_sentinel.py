"""
Unit tests for News Sentinel Agent components.
"""

import json
from datetime import datetime

import pytest

from news_sentinel.models import (
    ImpactfulHeadline,
    NewsAnalysisResult,
    NewsArticle,
    TickerSentiment,
)
from news_sentinel.utils import (
    format_date_for_display,
    group_articles_by_ticker,
    normalize_news_articles,
    parse_tickers,
)


class TestTickerParsing:
    """Tests for ticker parsing utility."""

    def test_parse_comma_separated(self):
        """Test parsing comma-separated tickers."""
        result = parse_tickers("AAPL, MSFT, SPY")
        assert result == ["AAPL", "MSFT", "SPY"]

    def test_parse_space_separated(self):
        """Test parsing space-separated tickers."""
        result = parse_tickers("AAPL MSFT SPY")
        assert result == ["AAPL", "MSFT", "SPY"]

    def test_parse_mixed_separators(self):
        """Test parsing with mixed separators."""
        result = parse_tickers("AAPL, MSFT  SPY,TSLA")
        assert result == ["AAPL", "MSFT", "SPY", "TSLA"]

    def test_parse_lowercase(self):
        """Test that tickers are normalized to uppercase."""
        result = parse_tickers("aapl, msft, spy")
        assert result == ["AAPL", "MSFT", "SPY"]

    def test_parse_removes_duplicates(self):
        """Test that duplicate tickers are removed."""
        result = parse_tickers("AAPL, MSFT, AAPL, SPY, MSFT")
        assert result == ["AAPL", "MSFT", "SPY"]

    def test_parse_empty_string(self):
        """Test parsing empty string."""
        result = parse_tickers("")
        assert result == []

    def test_parse_whitespace_only(self):
        """Test parsing whitespace-only string."""
        result = parse_tickers("   ,  ,  ")
        assert result == []


class TestNewsNormalization:
    """Tests for news article normalization."""

    def test_normalize_deduplicates_by_url(self):
        """Test that articles are deduplicated by URL."""
        article1 = NewsArticle(
            title="Test Article 1",
            snippet="Snippet 1",
            source="Source 1",
            url="https://example.com/article1",
            ticker="AAPL",
            backend="tavily",
        )
        article2 = NewsArticle(
            title="Test Article 2",
            snippet="Snippet 2",
            source="Source 2",
            url="https://example.com/article1",  # Same URL
            ticker="AAPL",
            backend="tavily",
        )
        
        articles = [article1, article2]
        normalized = normalize_news_articles(articles)
        
        assert len(normalized) == 1
        assert normalized[0].title == "Test Article 1"

    def test_normalize_sorts_by_date(self):
        """Test that articles are sorted by date (newest first)."""
        now = datetime.now()
        article1 = NewsArticle(
            title="Old Article",
            snippet="Snippet",
            source="Source",
            url="https://example.com/old",
            date=now.replace(day=1),
            ticker="AAPL",
            backend="tavily",
        )
        article2 = NewsArticle(
            title="New Article",
            snippet="Snippet",
            source="Source",
            url="https://example.com/new",
            date=now,
            ticker="AAPL",
            backend="tavily",
        )
        
        articles = [article1, article2]
        normalized = normalize_news_articles(articles)
        
        assert len(normalized) == 2
        assert normalized[0].title == "New Article"  # Newest first
        assert normalized[1].title == "Old Article"

    def test_normalize_empty_list(self):
        """Test normalizing empty list."""
        result = normalize_news_articles([])
        assert result == []


class TestDateFormatting:
    """Tests for date formatting utility."""

    def test_format_date_recent(self):
        """Test formatting recent date."""
        now = datetime.now()
        result = format_date_for_display(now)
        assert "ago" in result or "Just now" in result

    def test_format_date_none(self):
        """Test formatting None date."""
        result = format_date_for_display(None)
        assert result == "Date not available"

    def test_format_date_old(self):
        """Test formatting old date."""
        old_date = datetime(2020, 1, 1)
        result = format_date_for_display(old_date)
        assert "2020-01-01" in result


class TestGroupArticlesByTicker:
    """Tests for grouping articles by ticker."""

    def test_group_articles(self):
        """Test grouping articles by ticker."""
        article1 = NewsArticle(
            title="Article 1",
            snippet="Snippet",
            source="Source",
            url="https://example.com/1",
            ticker="AAPL",
            backend="tavily",
        )
        article2 = NewsArticle(
            title="Article 2",
            snippet="Snippet",
            source="Source",
            url="https://example.com/2",
            ticker="MSFT",
            backend="tavily",
        )
        article3 = NewsArticle(
            title="Article 3",
            snippet="Snippet",
            source="Source",
            url="https://example.com/3",
            ticker="AAPL",
            backend="tavily",
        )
        
        articles = [article1, article2, article3]
        grouped = group_articles_by_ticker(articles)
        
        assert "AAPL" in grouped
        assert "MSFT" in grouped
        assert len(grouped["AAPL"]) == 2
        assert len(grouped["MSFT"]) == 1


class TestModelValidation:
    """Tests for Pydantic model validation."""

    def test_news_article_validation(self):
        """Test NewsArticle model validation."""
        article = NewsArticle(
            title="Test Article",
            snippet="Test snippet",
            source="Test Source",
            url="https://example.com/article",
            ticker="AAPL",
            backend="tavily",
        )
        
        assert article.ticker == "AAPL"
        assert article.url == "https://example.com/article"

    def test_ticker_sentiment_validation(self):
        """Test TickerSentiment model validation."""
        sentiment = TickerSentiment(
            ticker="AAPL",
            sentiment_label="positive",
            sentiment_score=0.7,
            summary="Test summary",
            key_themes=["earnings"],
            impactful_headlines=[
                ImpactfulHeadline(
                    title="Headline 1",
                    source="Source 1",
                    url="https://example.com/1",
                    why_it_matters="It matters",
                    ticker="AAPL",
                )
            ],
        )
        
        assert sentiment.ticker == "AAPL"
        assert sentiment.sentiment_score == 0.7
        assert len(sentiment.impactful_headlines) == 1

    def test_news_analysis_result_validation(self):
        """Test NewsAnalysisResult model validation."""
        result = NewsAnalysisResult(
            overall_summary="Test summary",
            overall_sentiment_score=0.5,
            ticker_sentiments=[
                TickerSentiment(
                    ticker="AAPL",
                    sentiment_label="positive",
                    sentiment_score=0.7,
                    summary="Test",
                    key_themes=[],
                    impactful_headlines=[],
                )
            ],
            article_counts={"positive": 5, "negative": 2, "neutral": 3},
        )
        
        assert result.overall_sentiment_score == 0.5
        assert len(result.ticker_sentiments) == 1
        assert result.article_counts["positive"] == 5


class TestJSONParsing:
    """Tests for JSON parsing robustness."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON structure."""
        json_data = {
            "overall_summary": "Test summary",
            "overall_sentiment_score": 0.5,
            "ticker_sentiments": [
                {
                    "ticker": "AAPL",
                    "sentiment_label": "positive",
                    "sentiment_score": 0.7,
                    "summary": "Test summary",
                    "key_themes": ["earnings"],
                    "impactful_headlines": [
                        {
                            "title": "Headline 1",
                            "source": "Source 1",
                            "url": "https://example.com/1",
                            "why_it_matters": "It matters",
                        }
                    ],
                }
            ],
            "article_counts": {"positive": 5, "negative": 2, "neutral": 3},
        }
        
        # This should not raise an exception
        result = NewsAnalysisResult.model_validate(json_data)
        assert result.overall_summary == "Test summary"
        assert len(result.ticker_sentiments) == 1

    def test_parse_missing_optional_fields(self):
        """Test parsing JSON with missing optional fields."""
        json_data = {
            "overall_summary": "Test summary",
            "overall_sentiment_score": 0.5,
            "ticker_sentiments": [
                {
                    "ticker": "AAPL",
                    "sentiment_label": "positive",
                    "sentiment_score": 0.7,
                    "summary": "Test summary",
                    "key_themes": [],  # Optional
                    "impactful_headlines": [],  # Optional but should have at least empty list
                }
            ],
        }
        
        # Should work with missing optional fields
        result = NewsAnalysisResult.model_validate(json_data)
        assert result.overall_summary == "Test summary"

