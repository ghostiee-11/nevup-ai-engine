"""Idempotent seed loader. Reads nevup_seed_dataset.json and upserts into Postgres.
Embeddings are NOT generated here — they're built lazily by the memory layer.
"""
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

from sqlalchemy.dialects.postgresql import insert

from app.config import settings
from app.db import SessionLocal
from app.models import Session as SessionModel
from app.models import Trade, Trader

log = logging.getLogger("seed")


def _parse_dt(s: str | None) -> datetime | None:
    if s is None:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


async def seed() -> dict:
    path = Path(settings.seed_path)
    if not path.exists():
        alt = Path("nevup_seed_dataset.json")
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(
                f"seed file not found at {settings.seed_path} or ./nevup_seed_dataset.json"
            )
    data = json.loads(path.read_text())
    counts = {"traders": 0, "sessions": 0, "trades": 0}
    async with SessionLocal() as db:
        for trader in data["traders"]:
            stmt = insert(Trader).values(
                user_id=trader["userId"],
                name=trader["name"],
                profile=trader.get("profile", {}),
                ground_truth_pathologies=trader.get("groundTruthPathologies", []),
                description=trader.get("description"),
            ).on_conflict_do_update(
                index_elements=[Trader.user_id],
                set_={
                    "name": trader["name"],
                    "profile": trader.get("profile", {}),
                    "ground_truth_pathologies": trader.get("groundTruthPathologies", []),
                    "description": trader.get("description"),
                },
            )
            await db.execute(stmt)
            counts["traders"] += 1
            for sess in trader["sessions"]:
                sess_stmt = insert(SessionModel).values(
                    session_id=sess["sessionId"],
                    user_id=sess["userId"],
                    date=_parse_dt(sess["date"]),
                    trade_count=sess["tradeCount"],
                    win_rate=sess["winRate"],
                    total_pnl=sess["totalPnl"],
                    notes=sess.get("notes") or None,
                    raw=sess,
                ).on_conflict_do_update(
                    index_elements=[SessionModel.session_id],
                    set_={
                        "trade_count": sess["tradeCount"],
                        "win_rate": sess["winRate"],
                        "total_pnl": sess["totalPnl"],
                        "raw": sess,
                    },
                )
                await db.execute(sess_stmt)
                counts["sessions"] += 1
                for trade in sess["trades"]:
                    t_stmt = insert(Trade).values(
                        trade_id=trade["tradeId"],
                        user_id=trade["userId"],
                        session_id=trade["sessionId"],
                        asset=trade["asset"],
                        asset_class=trade["assetClass"],
                        direction=trade["direction"],
                        entry_price=trade["entryPrice"],
                        exit_price=trade.get("exitPrice"),
                        quantity=trade["quantity"],
                        entry_at=_parse_dt(trade["entryAt"]),
                        exit_at=_parse_dt(trade.get("exitAt")),
                        status=trade["status"],
                        outcome=trade.get("outcome"),
                        pnl=trade.get("pnl"),
                        plan_adherence=trade.get("planAdherence"),
                        emotional_state=trade.get("emotionalState"),
                        entry_rationale=trade.get("entryRationale"),
                        revenge_flag=bool(trade.get("revengeFlag", False)),
                    ).on_conflict_do_nothing(index_elements=[Trade.trade_id])
                    await db.execute(t_stmt)
                    counts["trades"] += 1
        await db.commit()
    log.info("seed complete: %s", counts)
    return counts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(seed())
