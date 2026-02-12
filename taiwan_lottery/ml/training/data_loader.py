"""Data loader for ML training — extracts number history from DB."""

from sqlalchemy.ext.asyncio import AsyncSession

from taiwan_lottery.db.crud import lotto649, super_lotto, daily_cash, bingo

# Maps game_type -> (crud_module, number_extractor, max_num, pick_count)
GAME_CONFIG = {
    "lotto649": {
        "crud": lotto649,
        "extract": lambda d: d.numbers_sorted,
        "max_num": 49,
        "pick_count": 6,
    },
    "super_lotto": {
        "crud": super_lotto,
        "extract": lambda d: d.zone1_sorted,
        "max_num": 38,
        "pick_count": 6,
    },
    "daily_cash": {
        "crud": daily_cash,
        "extract": lambda d: d.numbers_sorted,
        "max_num": 39,
        "pick_count": 5,
    },
    "bingo": {
        "crud": bingo,
        "extract": lambda d: d.numbers,
        "max_num": 80,
        "pick_count": 20,
    },
}


async def load_history(
    session: AsyncSession, game_type: str
) -> tuple[list[list[int]], int, int]:
    """Load complete draw history as sorted number lists.

    Returns:
        (history, max_num, pick_count)
    """
    config = GAME_CONFIG.get(game_type)
    if not config:
        raise ValueError(f"Unknown game type: {game_type}")

    crud_module = config["crud"]
    extract_fn = config["extract"]

    draws = await crud_module.get_all_sorted_by_date(session)
    history = [extract_fn(d) for d in draws]

    return history, config["max_num"], config["pick_count"]
