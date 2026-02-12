"""Pydantic schemas for lottery data."""

from datetime import date, datetime
from pydantic import BaseModel


# --- Lotto 6/49 (大樂透) ---

class Lotto649DrawSchema(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    draw_term: str
    draw_date: date
    num_1: int
    num_2: int
    num_3: int
    num_4: int
    num_5: int
    num_6: int
    special_num: int
    numbers_sorted: list[int]
    sum_total: int
    odd_count: int
    span: int


# --- Super Lotto (威力彩) ---

class SuperLottoDrawSchema(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    draw_term: str
    draw_date: date
    zone1_num_1: int
    zone1_num_2: int
    zone1_num_3: int
    zone1_num_4: int
    zone1_num_5: int
    zone1_num_6: int
    zone2_num: int
    zone1_sorted: list[int]
    sum_zone1: int
    odd_count: int
    span: int


# --- Daily Cash 5/39 (今彩539) ---

class DailyCashDrawSchema(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    draw_term: str
    draw_date: date
    num_1: int
    num_2: int
    num_3: int
    num_4: int
    num_5: int
    numbers_sorted: list[int]
    sum_total: int
    odd_count: int
    span: int


# --- Bingo Bingo (賓果賓果) ---

class BingoDrawSchema(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    draw_term: str
    draw_datetime: datetime
    numbers: list[int]
    sum_total: int
    odd_count: int
    sector_1_count: int
    sector_2_count: int
    sector_3_count: int
    sector_4_count: int


# --- Paginated response ---

class PaginatedResponse(BaseModel):
    items: list
    total: int
    page: int
    page_size: int
    total_pages: int


# --- Scraper ---

class ScrapeRequest(BaseModel):
    game_type: str  # lotto649 / super_lotto / daily_cash / bingo


class ScrapeStatusSchema(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    game_type: str
    status: str
    records_found: int
    records_inserted: int
    error_message: str | None
    started_at: datetime
    finished_at: datetime | None


class BackfillRequest(BaseModel):
    game_type: str
    year_start: int = 93  # 民國年
    year_end: int | None = None  # None = current year
