"""
Configuration constants for portfolio classification.
"""

# ============================================================================
# CLASSIFICATION CONFIGURATION - EASILY EXTENSIBLE
# ============================================================================
# When you add new holdings to holdings_from_op.csv that aren't classified:
# 1. Check the warning section at the top of the dashboard
# 2. Add the company name or relevant keywords to the appropriate dictionary below
# 3. Restart the application
#
# Structure:
# - 'keywords': Company names or words that appear in holding names (case-insensitive)
# - 'company_suffixes': Legal entity suffixes (e.g., 'INC', 'PLC', 'OYJ')
# - 'etf_patterns': Patterns that appear in ETF/index fund names
# ============================================================================

GEOGRAPHY_KEYWORDS = {
    'U.S.': {
        'keywords': ['USA', 'US', 'AMERIKKA', 'AMERICA', 'NASDAQ', 'NYSE', 'UNITED STATES'],
        'company_suffixes': ['INC', 'CORP', 'CO', 'COMPANY'],
        'etf_patterns': ['USA', 'US', 'AMERIKKA']
    },
    'Finland': {
        'keywords': ['DIGIA', 'KOJAMO', 'MANDATUM', 'NOKIA', 'SAMPO', 
                    'TERVEYSTALO', 'UPM', 'FORTUM', 'ORION', 'NORDEA'],
        'company_suffixes': ['OYJ'],
        'etf_patterns': []
    },
    'Europe/Global': {
        'keywords': ['BAE SYSTEMS', 'BAE', 'IPSEN'],
        'company_suffixes': ['PLC', 'SA', 'SPA', 'AS', 'ABP', 'AB', 'AG', 'SE', 'NV'],
        'etf_patterns': ['EMU', 'EUROPE', 'EURO', 'EMERGING', 'WORLD', 'GLOBAL', 'ALLWORLD']
    }
}

SECTOR_KEYWORDS = {
    'Technology': {
        'keywords': ['MICROSOFT', 'ALPHABET', 'AMAZON', 'QUALCOMM', 'NOKIA', 
                    'DIGIA', 'AUTOMATION', 'ROBOTICS', 'QUANTUM', 'SEMICONDUCTOR',
                    'APPLE', 'GOOGLE', 'META', 'TESLA', 'NVIDIA', 'INTEL', 'AMD'],
        'etf_patterns': ['AUTOMATION', 'ROBOTICS', 'QUANTUM', 'SEMICONDUCTOR', 'TECH']
    },
    'Healthcare': {
        'keywords': ['ABBVIE', 'ASTRAZENECA', 'MERCK', 'PFIZER', 'NOVO NORDISK',
                    'ORION', 'IPSEN', 'ORGANON', 'HEALTHCARE', 'AGEING POPULATION',
                    'HEALTHCARE INNOVATION', 'TERVEYSTALO', 'JOHNSON', 'BRISTOL',
                    'ROCHE', 'NOVARTIS', 'SANOFI', 'GLAXOSMITHKLINE'],
        'etf_patterns': ['HEALTHCARE', 'AGEING', 'PHARMA', 'BIOTECH']
    },
    'Financial Services': {
        'keywords': ['BANCO', 'SANTANDER', 'SAMPO', 'MANDATUM', 'NORDEA', 'BANK',
                    'JPMORGAN', 'GOLDMAN', 'MORGAN STANLEY', 'WELLS FARGO', 'CITI'],
        'etf_patterns': ['FINANCIAL', 'BANK', 'INSURANCE']
    },
    'Energy/Utilities': {
        'keywords': ['FORTUM', 'ENEL', 'CLEAN ENERGY', 'UPM', 'EXXON', 'CHEVRON',
                    'SHELL', 'BP', 'TOTAL', 'EQUINOR'],
        'etf_patterns': ['ENERGY', 'CLEAN ENERGY', 'UTILITIES', 'OIL', 'GAS']
    },
    'Real Estate': {
        'keywords': ['KOJAMO', 'REIT', 'REAL ESTATE', 'PROPERTY'],
        'etf_patterns': ['REAL ESTATE', 'PROPERTY', 'REIT']
    },
    'Consumer Discretionary': {
        'keywords': ['WALT DISNEY', 'TELADOC', 'NIKE', 'STARBUCKS', 'MCDONALDS',
                    'HOME DEPOT', 'LOWES', 'TESLA'],
        'etf_patterns': ['CONSUMER DISCRETIONARY', 'RETAIL']
    },
    'Consumer Staples': {
        'keywords': ['NESTLE', 'UNILEVER', 'PROCTER', 'GAMBLE', 'COCA COLA', 'PEPSI'],
        'etf_patterns': ['CONSUMER STAPLES', 'FOOD', 'BEVERAGE']
    },
    'Industrials': {
        'keywords': ['BAE SYSTEMS', 'BROOKFIELD', 'INDUSTRIAL GOODS', 'CATERPILLAR',
                    'BOEING', 'AIRBUS', 'SIEMENS', 'GE'],
        'etf_patterns': ['INDUSTRIAL', 'INDUSTRIAL GOODS']
    },
    'Materials': {
        'keywords': ['RIO TINTO', 'BHP', 'BASF', 'DOW', 'DUPONT', 'LINDE'],
        'etf_patterns': ['MATERIALS', 'BASIC RESOURCES', 'MINING', 'CHEMICALS']
    },
    'Communication Services': {
        'keywords': ['VODAFONE', 'DEUTSCHE TELEKOM', 'TELEFONICA', 'VERIZON', 'AT&T'],
        'etf_patterns': ['TELECOMMUNICATIONS', 'COMMUNICATION', 'TELECOM']
    },
    'Broad Market ETF': {
        'keywords': [],
        'etf_patterns': ['MSCI WORLD', 'FTSE ALLWORLD', 'MSCI EM', 'EMERGING MARKETS',
                        'WIDE MOAT', 'SMALL CAP', 'INDEKSI', 'WORLD', 'MSCI EMU',
                        'ALL-WORLD', 'TOTAL MARKET', 'GLOBAL INDEX']
    },
    'Thematic ETF': {
        'keywords': [],
        'etf_patterns': ['AGEING', 'AUTOMATION', 'ROBOTICS', 'CLEAN ENERGY', 'QUANTUM',
                        'SEMICONDUCTOR', 'HEALTHCARE INNOVATION', 'INDUSTRIAL GOODS',
                        'ESG', 'SUSTAINABLE', 'DIVIDEND', 'GROWTH', 'VALUE']
    }
}

# Default CSV file path (fallback for backward compatibility)
DEFAULT_CSV_PATH = 'holdings_from_op.csv'

# ============================================================================
# HISTORICAL PERFORMANCE CONFIGURATION
# ============================================================================

# User/portfolio name (used in snapshot file naming)
DEFAULT_USER_NAME = "jari"

# Snapshot file naming patterns
# Historical: {DDMMYYYY}_assets_{user}.csv (e.g., 20112025_assets_jari.csv)
# Latest: latest_assets_{user}.csv (e.g., latest_assets_jari.csv)
LATEST_SNAPSHOT_PATTERN = "latest_assets_{user}.csv"
HISTORICAL_SNAPSHOT_PATTERN = "{date}_assets_{user}.csv"

# Date format for parsing snapshot filenames (DDMMYYYY)
SNAPSHOT_DATE_FORMAT = "%d%m%Y"

