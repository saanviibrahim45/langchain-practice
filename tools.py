from langchain.tools import tool

@tool
def reverse_text(text: str) -> str:
    """Reverses a string."""
    return text[::-1]

@tool
def word_count(text: str) -> str:
    """Counts the number of words in a given text."""
    words = text.split()
    return f"Word count: {len(words)}"

@tool
def text_uppercase(text: str) -> str:
    """Converts text to uppercase."""
    return text.upper()

@tool
def get_stock_info(symbol: str) -> str:
    """Gets current stock price and basic info for a given ticker symbol."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info
        hist = ticker.history(period="1d")
        
        if hist.empty or not info:
            return f"No data found for symbol: {symbol}"
        
        current_price = hist['Close'].iloc[-1] if not hist.empty else "N/A"
        company_name = info.get('longName', symbol.upper())
        market_cap = info.get('marketCap', 'N/A')
        
        if market_cap != 'N/A' and isinstance(market_cap, (int, float)):
            market_cap = f"${market_cap:,.0f}"
        
        return f"{company_name} ({symbol.upper()}): ${current_price:.2f}, Market Cap: {market_cap}"
    except ImportError:
        return "yfinance library not installed. Run: pip install yfinance"
    except Exception as e:
        return f"Error getting stock info: {str(e)}"
