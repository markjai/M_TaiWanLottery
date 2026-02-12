"""Pydantic schemas for statistics."""

from pydantic import BaseModel


class NumberFrequency(BaseModel):
    number: int
    count: int
    percentage: float


class HotColdAnalysis(BaseModel):
    hot_numbers: list[NumberFrequency]
    cold_numbers: list[NumberFrequency]
    window_size: int


class GapAnalysis(BaseModel):
    number: int
    current_gap: int
    average_gap: float
    max_gap: int
    gap_ratio: float  # current_gap / average_gap


class PairFrequency(BaseModel):
    pair: list[int]
    count: int
    percentage: float


class SumDistribution(BaseModel):
    range_start: int
    range_end: int
    count: int
    percentage: float


class StatisticsResponse(BaseModel):
    game_type: str
    total_draws: int
    frequency: list[NumberFrequency] | None = None
    hot_cold: HotColdAnalysis | None = None
    gaps: list[GapAnalysis] | None = None
    pairs: list[PairFrequency] | None = None
    sum_distribution: list[SumDistribution] | None = None
