"""
News Sentinel Agent - Main orchestration class for news analysis.
"""

import json
import re
from typing import List, Optional

from news_sentinel.config import get_openai_api_key, get_openai_model_name
from news_sentinel.logger import get_logger
from news_sentinel.models import (
    NewsAnalysisResult,
    NewsArticle,
    get_news_analysis_json_schema,
)
from news_sentinel.news_sources import NewsSearchBackend, create_search_backend
from news_sentinel.utils import normalize_news_articles

logger = get_logger("news_sentinel.agent")


class NewsSentinelAgent:
    """
    Main agent class for fetching and analyzing financial news.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        openai_model: Optional[str] = None,
    ):
        """
        Initialize the News Sentinel Agent.
        
        Args:
            openai_api_key: OpenAI API key. If None, will try to get from config.
            openai_model: OpenAI model name. If None, will use default from config.
        """
        self.openai_api_key = openai_api_key or get_openai_api_key()
        self.openai_model = openai_model or get_openai_model_name()
        
        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable or in Streamlit secrets."
            )
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.openai_api_key)
            logger.info(f"Initialized NewsSentinelAgent with model: {self.openai_model}")
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

    def fetch_news(
        self,
        tickers: List[str],
        backend_name: str = "tavily",
        days_lookback: int = 7,
        max_articles_per_ticker: int = 10,
    ) -> List[NewsArticle]:
        """
        Fetch news articles for the given tickers.
        
        Args:
            tickers: List of ticker symbols
            backend_name: "tavily" or "duckduckgo"
            days_lookback: Number of days to look back
            max_articles_per_ticker: Maximum articles per ticker
        
        Returns:
            List of NewsArticle objects
        """
        if not tickers:
            logger.warning("No tickers provided for news fetching")
            return []
        
        logger.info(f"Fetching news for {len(tickers)} ticker(s): {', '.join(tickers)}")
        logger.info(f"Backend: {backend_name}, Days lookback: {days_lookback}, Max articles per ticker: {max_articles_per_ticker}")
        
        # Create search backend
        try:
            backend = create_search_backend(backend_name)
            logger.debug(f"Created search backend: {backend_name}")
        except Exception as e:
            logger.error(f"Failed to create search backend {backend_name}: {e}")
            raise
        
        # Fetch news
        try:
            articles = backend.search_news(
                tickers=tickers,
                days_lookback=days_lookback,
                max_articles_per_ticker=max_articles_per_ticker,
            )
            logger.info(f"Fetched {len(articles)} articles from {backend_name}")
        except Exception as e:
            logger.error(f"Error fetching news from {backend_name}: {e}")
            raise
        
        # Normalize and deduplicate
        original_count = len(articles)
        articles = normalize_news_articles(articles)
        if len(articles) < original_count:
            logger.debug(f"Deduplicated articles: {original_count} -> {len(articles)}")
        
        logger.info(f"Total articles after normalization: {len(articles)}")
        return articles

    def analyze_news(
        self,
        articles: List[NewsArticle],
        tickers: List[str],
    ) -> NewsAnalysisResult:
        """
        Analyze news articles using OpenAI to extract sentiment and key insights.
        
        Args:
            articles: List of news articles to analyze
            tickers: List of ticker symbols (for context)
        
        Returns:
            NewsAnalysisResult object
        
        Raises:
            ValueError: If articles list is empty
            RuntimeError: If OpenAI API call fails or returns invalid data
        """
        if not articles:
            logger.warning("No articles provided for analysis")
            raise ValueError("No articles provided for analysis")
        
        logger.info(f"Analyzing {len(articles)} articles for {len(tickers)} ticker(s): {', '.join(tickers)}")
        
        # Prepare articles data for the prompt
        articles_data = []
        for article in articles:
            article_dict = {
                "title": article.title,
                "snippet": article.snippet,
                "source": article.source,
                "url": article.url,
                "date": article.date.isoformat() if article.date else "Unknown",
                "ticker": article.ticker,
            }
            articles_data.append(article_dict)
        
        logger.debug(f"Prepared {len(articles_data)} articles for OpenAI analysis")
        
        # Build the prompt
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(articles_data, tickers)
        
        logger.debug(f"Built prompts (system: {len(system_prompt)} chars, user: {len(user_prompt)} chars)")
        
        # Call OpenAI
        try:
            logger.info(f"Calling OpenAI API with model: {self.openai_model}")
            response = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,  # Lower temperature for more consistent JSON output
                response_format={"type": "json_object"},  # Force JSON output
            )
            
            content = response.choices[0].message.content
            if not content:
                logger.error("OpenAI returned empty response")
                raise RuntimeError("OpenAI returned empty response")
            
            logger.debug(f"Received OpenAI response ({len(content)} characters)")
            logger.debug(f"Response preview: {content[:200]}...")
            
            # Parse JSON with robust error handling
            logger.info("Parsing OpenAI response...")
            analysis_result = self._parse_openai_response(content, tickers)
            
            logger.info(f"Successfully parsed analysis result:")
            logger.info(f"  - Overall sentiment score: {analysis_result.overall_sentiment_score:.2f}")
            logger.info(f"  - Ticker sentiments: {len(analysis_result.ticker_sentiments)}")
            for ts in analysis_result.ticker_sentiments:
                logger.info(f"    * {ts.ticker}: {ts.sentiment_label} ({ts.sentiment_score:.2f}), {len(ts.impactful_headlines)} headlines")
            
            return analysis_result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenAI JSON response: {e}")
            raise RuntimeError(f"Failed to parse OpenAI JSON response: {e}")
        except Exception as e:
            logger.error(f"OpenAI API error: {e}", exc_info=True)
            raise RuntimeError(f"OpenAI API error: {e}")

    def _build_system_prompt(self) -> str:
        """Build the system prompt for OpenAI."""
        schema = get_news_analysis_json_schema()
        
        return f"""You are an expert financial news analyst. Your task is to analyze financial news articles and identify the most impactful items for investors.

You must:
1. Identify ONLY the most impactful news items (earnings, guidance changes, M&A, major corporate actions, regulatory changes, etc.)
2. Classify sentiment for each ticker (positive/neutral/negative) with a numeric score from -1.0 to 1.0
3. Provide a high-level summary of the overall news landscape
4. Highlight key themes: earnings, guidance changes, macro/regulation, M&A, share buybacks, dividends, lawsuits, downgrades, etc.
5. Select 3-5 most impactful headlines per ticker with explanations of why they matter

You must output valid JSON conforming to this schema:
{schema}

Be concise but informative. Focus on actionable insights for investors."""

    def _build_user_prompt(self, articles_data: List[dict], tickers: List[str]) -> str:
        """Build the user prompt with article data."""
        articles_json = json.dumps(articles_data, indent=2)
        
        return f"""Analyze the following financial news articles for tickers: {', '.join(tickers)}

Articles:
{articles_json}

Instructions:
1. Identify the most impactful news items for each ticker (focus on material events)
2. Provide sentiment classification and score for each ticker
3. Create an overall summary (2-4 sentences)
4. Select 3-5 key headlines per ticker with explanations
5. Count articles by sentiment category
6. Output valid JSON following the specified schema"""

    def _parse_openai_response(
        self, content: str, tickers: List[str]
    ) -> NewsAnalysisResult:
        """
        Parse OpenAI response with robust error handling.
        
        Args:
            content: Raw JSON string from OpenAI
            tickers: List of tickers for validation
        
        Returns:
            NewsAnalysisResult object
        
        Raises:
            RuntimeError: If parsing fails or data is invalid
        """
        # Try to extract JSON from the response (in case it's wrapped in markdown)
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse JSON: {e}. Content: {content[:500]}")
        
        # Validate and parse with Pydantic
        try:
            logger.debug("Validating response with Pydantic models...")
            result = NewsAnalysisResult.model_validate(data)
            logger.debug("Pydantic validation successful")
            
            # Post-process: populate missing ticker fields in headlines from parent ticker_sentiment
            headlines_fixed = 0
            for ticker_sentiment in result.ticker_sentiments:
                parent_ticker = ticker_sentiment.ticker
                for headline in ticker_sentiment.impactful_headlines:
                    if headline.ticker is None:
                        headline.ticker = parent_ticker
                        headlines_fixed += 1
            if headlines_fixed > 0:
                logger.debug(f"Fixed {headlines_fixed} headlines with missing ticker fields")
            
            # Ensure all tickers are represented (add neutral sentiment if missing)
            ticker_set = set(tickers)
            existing_tickers = {ts.ticker for ts in result.ticker_sentiments}
            missing_tickers = ticker_set - existing_tickers
            
            from news_sentinel.models import TickerSentiment
            
            for ticker in missing_tickers:
                result.ticker_sentiments.append(
                    TickerSentiment(
                        ticker=ticker,
                        sentiment_label="neutral",
                        sentiment_score=0.0,
                        summary="No significant news found for this ticker.",
                        key_themes=[],
                        impactful_headlines=[],
                    )
                )
            
            return result
            
        except Exception as e:
            logger.warning(f"Pydantic validation failed: {e}. Attempting fallback parsing...")
            # Try to create a minimal valid result as fallback
            try:
                # Extract what we can from the response
                overall_summary = data.get("overall_summary", "Analysis completed.")
                overall_sentiment_score = float(data.get("overall_sentiment_score", 0.0))
                logger.debug(f"Fallback: Extracted overall_summary and sentiment_score ({overall_sentiment_score:.2f})")
                
                # Create basic ticker sentiments
                ticker_sentiments = []
                for ticker in tickers:
                    ticker_data = data.get("ticker_sentiments", [])
                    ticker_info = next(
                        (t for t in ticker_data if t.get("ticker", "").upper() == ticker.upper()),
                        None
                    )
                    
                    if ticker_info:
                        # Fix missing ticker fields in headlines
                        if "impactful_headlines" in ticker_info:
                            for headline in ticker_info["impactful_headlines"]:
                                if "ticker" not in headline or headline.get("ticker") is None:
                                    headline["ticker"] = ticker
                        
                        # Try to validate as TickerSentiment
                        try:
                            from news_sentinel.models import TickerSentiment
                            ticker_sentiment = TickerSentiment.model_validate(ticker_info)
                            # Post-process: ensure all headlines have ticker
                            for headline in ticker_sentiment.impactful_headlines:
                                if headline.ticker is None:
                                    headline.ticker = ticker
                            ticker_sentiments.append(ticker_sentiment)
                        except Exception as validation_error:
                            # If validation fails, create a minimal valid entry
                            from news_sentinel.models import TickerSentiment, ImpactfulHeadline
                            fixed_headlines = []
                            if "impactful_headlines" in ticker_info:
                                for h in ticker_info["impactful_headlines"]:
                                    try:
                                        h["ticker"] = h.get("ticker") or ticker
                                        fixed_headlines.append(ImpactfulHeadline.model_validate(h))
                                    except Exception:
                                        pass  # Skip invalid headlines
                            
                            ticker_sentiments.append(
                                TickerSentiment(
                                    ticker=ticker,
                                    sentiment_label=ticker_info.get("sentiment_label", "neutral"),
                                    sentiment_score=float(ticker_info.get("sentiment_score", 0.0)),
                                    summary=ticker_info.get("summary", "Analysis incomplete."),
                                    key_themes=ticker_info.get("key_themes", []),
                                    impactful_headlines=fixed_headlines,
                                )
                            )
                    else:
                        from news_sentinel.models import TickerSentiment
                        ticker_sentiments.append(
                            TickerSentiment(
                                ticker=ticker,
                                sentiment_label="neutral",
                                sentiment_score=0.0,
                                summary="Analysis incomplete.",
                                key_themes=[],
                                impactful_headlines=[],
                            )
                        )
                
                result = NewsAnalysisResult(
                    overall_summary=overall_summary,
                    overall_sentiment_score=overall_sentiment_score,
                    ticker_sentiments=ticker_sentiments,
                    article_counts=data.get("article_counts", {"positive": 0, "negative": 0, "neutral": 0}),
                )
                
                # Final post-process: ensure all headlines have ticker
                for ticker_sentiment in result.ticker_sentiments:
                    parent_ticker = ticker_sentiment.ticker
                    for headline in ticker_sentiment.impactful_headlines:
                        if headline.ticker is None:
                            headline.ticker = parent_ticker
                
                return result
            except Exception as fallback_error:
                raise RuntimeError(
                    f"Failed to parse OpenAI response and fallback failed: {e}. Fallback error: {fallback_error}"
                )

    def analyze(
        self,
        tickers: List[str],
        backend_name: str = "tavily",
        days_lookback: int = 7,
        max_articles_per_ticker: int = 10,
    ) -> tuple[List[NewsArticle], NewsAnalysisResult]:
        """
        Complete workflow: fetch news and analyze.
        
        Args:
            tickers: List of ticker symbols
            backend_name: "tavily" or "duckduckgo"
            days_lookback: Number of days to look back
            max_articles_per_ticker: Maximum articles per ticker
        
        Returns:
            Tuple of (articles, analysis_result)
        """
        logger.info(f"Starting complete analysis workflow for {len(tickers)} ticker(s)")
        
        # Fetch news
        articles = self.fetch_news(
            tickers=tickers,
            backend_name=backend_name,
            days_lookback=days_lookback,
            max_articles_per_ticker=max_articles_per_ticker,
        )
        
        # Analyze
        analysis_result = self.analyze_news(articles=articles, tickers=tickers)
        
        logger.info("Analysis workflow completed successfully")
        return articles, analysis_result

