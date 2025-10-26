import sqlite3
from datetime import datetime
from typing import List, Tuple, Optional

class ConversationMemory:
    """Memory system for storing and retrieving agent conversations."""
    
    def __init__(self, db_path="conversation_history.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize the database with required tables."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_input TEXT NOT NULL,
                agent_response TEXT NOT NULL,
                session_id TEXT DEFAULT 'default',
                tool_used TEXT DEFAULT NULL,
                context_data TEXT DEFAULT NULL
            )
        """)
        
        # Stock-specific memory table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS stock_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                price REAL,
                market_cap TEXT,
                session_id TEXT DEFAULT 'default'
            )
        """)
        conn.commit()
        conn.close()
    
    def save_exchange(self, user_input: str, agent_response: str, 
                     session_id: str = "default", tool_used: str = None, 
                     context_data: str = None):
        """Save a conversation exchange to memory."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """INSERT INTO conversations 
               (timestamp, user_input, agent_response, session_id, tool_used, context_data) 
               VALUES (?, ?, ?, ?, ?, ?)""",
            (datetime.now().isoformat(), user_input, agent_response, session_id, tool_used, context_data)
        )
        conn.commit()
        conn.close()
    
    def save_stock_query(self, symbol: str, price: float = None, 
                        market_cap: str = None, session_id: str = "default"):
        """Save stock query data for analytics."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """INSERT INTO stock_queries (timestamp, symbol, price, market_cap, session_id) 
               VALUES (?, ?, ?, ?, ?)""",
            (datetime.now().isoformat(), symbol.upper(), price, market_cap, session_id)
        )
        conn.commit()
        conn.close()
    
    def get_conversation_history(self, limit: int = 10, session_id: str = "default") -> List[Tuple]:
        """Get recent conversation history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            """SELECT timestamp, user_input, agent_response, tool_used 
               FROM conversations 
               WHERE session_id = ? 
               ORDER BY id DESC LIMIT ?""",
            (session_id, limit)
        )
        results = cursor.fetchall()
        conn.close()
        return list(reversed(results))  # Chronological order
    
    def get_stock_history(self, symbol: str = None, limit: int = 10, 
                         session_id: str = "default") -> List[Tuple]:
        """Get stock query history, optionally filtered by symbol."""
        conn = sqlite3.connect(self.db_path)
        if symbol:
            cursor = conn.execute(
                """SELECT timestamp, symbol, price, market_cap 
                   FROM stock_queries 
                   WHERE session_id = ? AND symbol = ? 
                   ORDER BY id DESC LIMIT ?""",
                (session_id, symbol.upper(), limit)
            )
        else:
            cursor = conn.execute(
                """SELECT timestamp, symbol, price, market_cap 
                   FROM stock_queries 
                   WHERE session_id = ? 
                   ORDER BY id DESC LIMIT ?""",
                (session_id, limit)
            )
        results = cursor.fetchall()
        conn.close()
        return results
    
    def get_frequently_queried_stocks(self, limit: int = 5, 
                                    session_id: str = "default") -> List[Tuple]:
        """Get most frequently queried stock symbols."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            """SELECT symbol, COUNT(*) as query_count 
               FROM stock_queries 
               WHERE session_id = ? 
               GROUP BY symbol 
               ORDER BY query_count DESC 
               LIMIT ?""",
            (session_id, limit)
        )
        results = cursor.fetchall()
        conn.close()
        return results
    
    def clear_session(self, session_id: str = "default"):
        """Clear all data for a specific session."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM stock_queries WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()