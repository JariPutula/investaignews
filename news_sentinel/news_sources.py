"""
Search backend implementations for fetching financial news.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Optional

from news_sentinel.models import NewsArticle
from news_sentinel.config import get_tavily_api_key


class NewsSearchBackend(ABC):
    """Abstract base class for news search backends."""

    @abstractmethod
    def search_news(
        self,
        tickers: List[str],
        days_lookback: int = 7,
        max_articles_per_ticker: int = 10,
    ) -> List[NewsArticle]:
        """
        Search for news articles for the given tickers.
        
        Args:
            tickers: List of ticker symbols to search for
            days_lookback: Number of days to look back
            max_articles_per_ticker: Maximum articles to fetch per ticker
        
        Returns:
            List of NewsArticle objects
        """
        pass


class TavilySearchBackend(NewsSearchBackend):
    """Tavily search backend implementation."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Tavily backend.
        
        Args:
            api_key: Tavily API key. If None, will try to get from config.
        """
        self.api_key = api_key or get_tavily_api_key()
        if not self.api_key:
            raise ValueError(
                "Tavily API key not found. Set TAVILY_API_KEY environment variable or in Streamlit secrets."
            )
        
        try:
            from tavily import TavilyClient
            self.client = TavilyClient(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "tavily-python package not installed. Install with: pip install tavily-python"
            )

    def search_news(
        self,
        tickers: List[str],
        days_lookback: int = 7,
        max_articles_per_ticker: int = 10,
    ) -> List[NewsArticle]:
        """Search for news using Tavily."""
        articles: List[NewsArticle] = []
        
        # Calculate date filter
        max_age_days = days_lookback
        search_query_date = (datetime.now() - timedelta(days=max_age_days)).strftime("%Y-%m-%d")
        
        for ticker in tickers:
            try:
                # Build search query
                query = f"{ticker} stock news financial"
                
                # Search using Tavily
                response = self.client.search(
                    query=query,
                    search_depth="advanced",
                    max_results=max_articles_per_ticker,
                    include_answer=False,
                    include_raw_content=False,
                    include_domains=[],
                    exclude_domains=[],
                )
                
                # Extract results
                results = response.get("results", [])
                
                for result in results[:max_articles_per_ticker]:
                    # Parse date if available
                    published_date = None
                    if "published_date" in result:
                        try:
                            published_date = datetime.fromisoformat(
                                result["published_date"].replace("Z", "+00:00")
                            )
                        except Exception:
                            pass
                    
                    # Create NewsArticle
                    article = NewsArticle(
                        title=result.get("title", "No title"),
                        snippet=result.get("content", "")[:500],  # Limit snippet length
                        source=result.get("url", "").split("/")[2] if result.get("url") else "Unknown",
                        url=result.get("url", ""),
                        date=published_date,
                        ticker=ticker,
                        backend="tavily",
                    )
                    articles.append(article)
                    
            except Exception as e:
                # Log error but continue with other tickers
                print(f"Error fetching news for {ticker} with Tavily: {e}")
                continue
        
        return articles


class DuckDuckGoSearchBackend(NewsSearchBackend):
    """DuckDuckGo search backend implementation."""

    def __init__(self):
        """Initialize DuckDuckGo backend."""
        try:
            from duckduckgo_search import DDGS
            self.ddgs_class = DDGS
        except ImportError:
            raise ImportError(
                "duckduckgo-search package not installed. Install with: pip install duckduckgo-search"
            )

    def search_news(
        self,
        tickers: List[str],
        days_lookback: int = 7,
        max_articles_per_ticker: int = 10,
    ) -> List[NewsArticle]:
        """Search for news using DuckDuckGo."""
        articles: List[NewsArticle] = []
        
        for ticker in tickers:
            try:
                # Build search query - try to include company name if available
                # For now, just use ticker + financial news keywords
                query = f"{ticker} stock news financial"
                
                with self.ddgs_class() as ddgs:
                    # Use news search
                    results = list(
                        ddgs.news(
                            keywords=query,
                            max_results=max_articles_per_ticker,
                        )
                    )
                
                for result in results[:max_articles_per_ticker]:
                    # Parse date if available
                    published_date = None
                    if "date" in result:
                        try:
                            # DuckDuckGo returns dates in various formats
                            date_str = result["date"]
                            # Try to parse common formats
                            for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"]:
                                try:
                                    published_date = datetime.strptime(date_str, fmt)
                                    break
                                except ValueError:
                                    continue
                        except Exception:
                            pass
                    
                    # Create NewsArticle
                    article = NewsArticle(
                        title=result.get("title", "No title"),
                        snippet=result.get("body", result.get("snippet", ""))[:500],
                        source=result.get("source", "Unknown"),
                        url=result.get("url", ""),
                        date=published_date,
                        ticker=ticker,
                        backend="duckduckgo",
                    )
                    articles.append(article)
                    
            except Exception as e:
                # Log error but continue with other tickers
                print(f"Error fetching news for {ticker} with DuckDuckGo: {e}")
                continue
        
        return articles


def create_search_backend(backend_name: str) -> NewsSearchBackend:
    """
    Factory function to create a search backend instance.
    
    Args:
        backend_name: "tavily" or "duckduckgo"
    
    Returns:
        NewsSearchBackend instance
    
    Raises:
        ValueError: If backend_name is not recognized
    """
    backend_name_lower = backend_name.lower()
    
    if backend_name_lower == "tavily":
        return TavilySearchBackend()
    elif backend_name_lower in ["duckduckgo", "ddg"]:
        return DuckDuckGoSearchBackend()
    else:
        raise ValueError(f"Unknown backend: {backend_name}. Use 'tavily' or 'duckduckgo'")

