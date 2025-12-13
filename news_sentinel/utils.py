"""
Utility functions for News Sentinel Agent.
"""

import re
from datetime import datetime
from typing import List, Optional

from news_sentinel.models import NewsArticle


def parse_tickers(input_text: str) -> List[str]:
    """
    Parse ticker symbols from user input.
    Handles comma-separated, space-separated, or mixed formats.
    
    Args:
        input_text: User input string (e.g., "AAPL, MSFT, SPY" or "AAPL MSFT SPY")
    
    Returns:
        List of normalized ticker symbols (uppercase, no duplicates)
    
    Examples:
        >>> parse_tickers("AAPL, MSFT, SPY")
        ['AAPL', 'MSFT', 'SPY']
        >>> parse_tickers("AAPL MSFT  SPY")
        ['AAPL', 'MSFT', 'SPY']
        >>> parse_tickers("aapl,msft,spy")
        ['AAPL', 'MSFT', 'SPY']
    """
    if not input_text or not input_text.strip():
        return []
    
    # Split by comma or whitespace
    tickers = re.split(r'[,\s]+', input_text.strip())
    
    # Normalize: uppercase, remove empty strings
    tickers = [t.upper().strip() for t in tickers if t.strip()]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_tickers = []
    for ticker in tickers:
        if ticker not in seen:
            seen.add(ticker)
            unique_tickers.append(ticker)
    
    return unique_tickers


def normalize_news_articles(articles: List[NewsArticle]) -> List[NewsArticle]:
    """
    Normalize and deduplicate news articles.
    
    Args:
        articles: List of news articles
    
    Returns:
        Deduplicated list sorted by date (newest first)
    """
    if not articles:
        return []
    
    # Deduplicate by URL (most reliable identifier)
    seen_urls = set()
    unique_articles = []
    
    for article in articles:
        if article.url not in seen_urls:
            seen_urls.add(article.url)
            unique_articles.append(article)
    
    # Sort by date (newest first), articles without dates go to the end
    def sort_key(article: NewsArticle) -> tuple:
        if article.date:
            return (0, article.date)  # Has date, sort by date (descending handled by reverse)
        return (1, datetime.min)  # No date, put at end
    
    unique_articles.sort(key=sort_key, reverse=True)
    
    return unique_articles


def format_date_for_display(date: Optional[datetime]) -> str:
    """
    Format a datetime object for display in the UI.
    
    Args:
        date: Datetime object or None
    
    Returns:
        Formatted date string or "Date not available"
    """
    if date is None:
        return "Date not available"
    
    try:
        # Format as "YYYY-MM-DD" or relative time if very recent
        now = datetime.now()
        delta = now - date
        
        if delta.days == 0:
            if delta.seconds < 3600:
                minutes = delta.seconds // 60
                return f"{minutes} minutes ago" if minutes > 0 else "Just now"
            hours = delta.seconds // 3600
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif delta.days == 1:
            return "Yesterday"
        elif delta.days < 7:
            return f"{delta.days} days ago"
        else:
            return date.strftime("%Y-%m-%d")
    except Exception:
        return "Date not available"


def group_articles_by_ticker(articles: List[NewsArticle]) -> dict[str, List[NewsArticle]]:
    """
    Group articles by ticker symbol.
    
    Args:
        articles: List of news articles
    
    Returns:
        Dictionary mapping ticker to list of articles
    """
    grouped: dict[str, List[NewsArticle]] = {}
    
    for article in articles:
        ticker = article.ticker
        if ticker not in grouped:
            grouped[ticker] = []
        grouped[ticker].append(article)
    
    return grouped

