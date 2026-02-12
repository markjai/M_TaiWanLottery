"""Wrapper around TaiwanLotteryCrawler PyPI package.

The taiwanlottery package is synchronous, so we run it in an executor
to avoid blocking the async event loop.
"""

import asyncio
import ssl
from functools import partial

from loguru import logger

# Patch SSL for Python 3.13+ — taiwanlottery.com cert is missing SubjectKeyIdentifier
# which Python 3.13 rejects by default.
try:
    _default_ctx = ssl.create_default_context()
    _default_ctx.check_hostname = False
    _default_ctx.verify_mode = ssl.CERT_NONE

    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

try:
    from TaiwanLottery import TaiwanLotteryCrawler
    import requests

    # Monkey-patch the crawler's get_lottery_result to disable SSL verify
    _original_requests_get = requests.get

    def _patched_get(url, **kwargs):
        kwargs.setdefault("verify", False)
        return _original_requests_get(url, **kwargs)

    requests.get = _patched_get

    _CRAWLER_AVAILABLE = True
except ImportError:
    _CRAWLER_AVAILABLE = False
    logger.warning("taiwanlottery package not installed — scraper will be limited")


class LotteryClient:
    """Async wrapper for TaiwanLotteryCrawler.

    v1.5.1 API: methods take `back_time=['YYYY', 'MM']` (AD year as string).
    """

    def __init__(self):
        if _CRAWLER_AVAILABLE:
            self._crawler = TaiwanLotteryCrawler()
        else:
            self._crawler = None

    async def _run_sync(self, func, *args, **kwargs):
        """Run a synchronous function in executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(func, *args, **kwargs))

    @staticmethod
    def _roc_to_back_time(roc_year: int, month: int) -> list[str]:
        """Convert ROC year + month to back_time format ['YYYY', 'MM']."""
        ad_year = roc_year + 1911
        return [str(ad_year), str(month).zfill(2)]

    async def get_lotto649(self, roc_year: int, month: int):
        """Fetch 大樂透 draws for given ROC year and month."""
        if not self._crawler:
            raise RuntimeError("TaiwanLotteryCrawler not available")
        back_time = self._roc_to_back_time(roc_year, month)
        logger.debug("Fetching lotto649 for {}", back_time)
        return await self._run_sync(self._crawler.lotto649, back_time)

    async def get_super_lotto(self, roc_year: int, month: int):
        """Fetch 威力彩 draws for given ROC year and month."""
        if not self._crawler:
            raise RuntimeError("TaiwanLotteryCrawler not available")
        back_time = self._roc_to_back_time(roc_year, month)
        logger.debug("Fetching super_lotto for {}", back_time)
        return await self._run_sync(self._crawler.super_lotto, back_time)

    async def get_daily_cash(self, roc_year: int, month: int):
        """Fetch 今彩539 draws for given ROC year and month."""
        if not self._crawler:
            raise RuntimeError("TaiwanLotteryCrawler not available")
        back_time = self._roc_to_back_time(roc_year, month)
        logger.debug("Fetching daily_cash for {}", back_time)
        return await self._run_sync(self._crawler.daily_cash, back_time)


# Singleton
lottery_client = LotteryClient()
