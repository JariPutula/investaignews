"""
Pydantic models for News Sentinel Agent.
"""

from datetime import datetime
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class NewsArticle(BaseModel):
    """Represents a single news article fetched from a search backend."""

    title: str
    snippet: str = Field(description="Short excerpt/description of the article")
    source: str = Field(description="Publication/source name")
    url: str = Field(description="Link to original article")
    date: Optional[datetime] = Field(default=None, description="Publication date if available")
    ticker: str = Field(description="Associated ticker symbol")
    backend: str = Field(description="Which backend fetched it: 'tavily' or 'duckduckgo'")

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Ensure URL is not empty."""
        if not v or not v.strip():
            raise ValueError("URL cannot be empty")
        return v.strip()

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        """Normalize ticker to uppercase."""
        return v.upper().strip()


class ImpactfulHeadline(BaseModel):
    """Represents a key headline selected by the AI as impactful for investors."""

    title: str
    source: str
    date: Optional[str] = Field(default=None, description="Formatted date string")
    url: str
    why_it_matters: str = Field(description="Short explanation of why this news matters")
    ticker: str

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        """Normalize ticker to uppercase."""
        return v.upper().strip()


class TickerSentiment(BaseModel):
    """Sentiment analysis for a single ticker."""

    ticker: str
    sentiment_label: Literal["positive", "neutral", "negative"]
    sentiment_score: float = Field(ge=-1.0, le=1.0, description="Sentiment score from -1.0 to 1.0")
    summary: str = Field(description="Brief summary of impactful news for this ticker")
    key_themes: List[str] = Field(
        default_factory=list,
        description="Key themes like 'earnings', 'guidance change', 'M&A', etc."
    )
    impactful_headlines: List[ImpactfulHeadline] = Field(
        default_factory=list,
        description="3-5 key impactful headlines"
    )

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        """Normalize ticker to uppercase."""
        return v.upper().strip()

    @field_validator("impactful_headlines")
    @classmethod
    def validate_headlines_count(cls, v: List[ImpactfulHeadline]) -> List[ImpactfulHeadline]:
        """Ensure we have 3-5 headlines."""
        if len(v) < 3:
            # Allow fewer if that's all we have, but warn
            pass
        elif len(v) > 5:
            # Limit to top 5
            return v[:5]
        return v


class NewsAnalysisResult(BaseModel):
    """Complete analysis result from OpenAI."""

    overall_summary: str = Field(description="High-level natural language summary")
    overall_sentiment_score: float = Field(
        ge=-1.0, le=1.0, description="Overall sentiment score from -1.0 to 1.0"
    )
    ticker_sentiments: List[TickerSentiment]
    article_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of articles by sentiment: {'positive': 5, 'negative': 2, 'neutral': 3}"
    )

    @field_validator("article_counts")
    @classmethod
    def validate_article_counts(cls, v: Dict[str, int]) -> Dict[str, int]:
        """Ensure article_counts has valid keys."""
        valid_keys = {"positive", "negative", "neutral"}
        return {k: v.get(k, 0) for k in valid_keys if k in v or v.get(k, 0) > 0}


def get_news_analysis_json_schema() -> str:
    """
    Returns the JSON schema description for OpenAI prompts.
    This ensures the model outputs the correct structure.
    """
    return """
{
  "type": "object",
  "required": ["overall_summary", "overall_sentiment_score", "ticker_sentiments"],
  "properties": {
    "overall_summary": {
      "type": "string",
      "description": "A high-level natural language summary (2-4 sentences) of the most impactful news across all tickers"
    },
    "overall_sentiment_score": {
      "type": "number",
      "minimum": -1.0,
      "maximum": 1.0,
      "description": "Overall sentiment score: -1.0 (very negative) to 1.0 (very positive)"
    },
    "ticker_sentiments": {
      "type": "array",
      "description": "Sentiment analysis for each ticker",
      "items": {
        "type": "object",
        "required": ["ticker", "sentiment_label", "sentiment_score", "summary", "impactful_headlines"],
        "properties": {
          "ticker": {
            "type": "string",
            "description": "Ticker symbol (e.g., 'AAPL')"
          },
          "sentiment_label": {
            "type": "string",
            "enum": ["positive", "neutral", "negative"],
            "description": "Overall sentiment classification"
          },
          "sentiment_score": {
            "type": "number",
            "minimum": -1.0,
            "maximum": 1.0,
            "description": "Sentiment score for this ticker: -1.0 to 1.0"
          },
          "summary": {
            "type": "string",
            "description": "Brief summary (2-3 sentences) of the most impactful news for this ticker"
          },
          "key_themes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key themes identified: e.g., ['earnings', 'guidance change', 'M&A', 'share buyback', 'dividend', 'lawsuit', 'downgrade', 'macro/regulation']"
          },
          "impactful_headlines": {
            "type": "array",
            "minItems": 3,
            "maxItems": 5,
            "description": "3-5 most impactful headlines for investors",
            "items": {
              "type": "object",
              "required": ["title", "source", "url", "why_it_matters"],
              "properties": {
                "title": {"type": "string", "description": "Headline title"},
                "source": {"type": "string", "description": "Source/publication name"},
                "date": {"type": "string", "description": "Publication date if available (format: YYYY-MM-DD or relative like '2 days ago')"},
                "url": {"type": "string", "description": "Link to original article"},
                "why_it_matters": {
                  "type": "string",
                  "description": "Short explanation (1-2 sentences) of why this news matters for investors"
                }
              }
            }
          }
        }
      }
    },
    "article_counts": {
      "type": "object",
      "description": "Count of articles by sentiment",
      "properties": {
        "positive": {"type": "integer", "minimum": 0},
        "negative": {"type": "integer", "minimum": 0},
        "neutral": {"type": "integer", "minimum": 0}
      }
    }
  }
}
"""

