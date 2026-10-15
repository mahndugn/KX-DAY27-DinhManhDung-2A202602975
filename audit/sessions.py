"""Session data shared by the terminal replay tool and Streamlit."""
from common.db import db_conn


async def recent_sessions():
    async with db_conn() as conn:
        async with conn.execute("""
            SELECT thread_id, pr_url, MIN(timestamp) AS started,
                   MAX(timestamp) AS last_event, COUNT(*) AS events,
                   CASE MAX(CASE risk_level WHEN 'high' THEN 3 WHEN 'med' THEN 2 ELSE 1 END)
                       WHEN 3 THEN 'high' WHEN 2 THEN 'med' ELSE 'low' END AS worst_risk
            FROM audit_events GROUP BY thread_id, pr_url
            ORDER BY MAX(timestamp) DESC LIMIT 25
        """) as cur:
            return [dict(row) for row in await cur.fetchall()]
