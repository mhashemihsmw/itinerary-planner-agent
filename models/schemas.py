"""
Pydantic models for AI Trip Planner Core
"""
from datetime import date
from typing import List, Optional

from pydantic import BaseModel, Field


class TravelFacts(BaseModel):
    """Extracted travel facts from caption text (Step 1)"""
    destination: str = Field(..., description="Primary destination name")
    best_season: str = Field(..., description="Best time to visit")
    recommended_duration_days: int = Field(..., description="Recommended trip duration in days")
    attractions: List[str] = Field(..., description="Key attractions mentioned")
    food_suggestions: List[str] = Field(..., description="Local food recommendations")
    travel_style: str = Field(..., description="Travel style (e.g., adventure, relaxed, cultural)")


class DailyActivity(BaseModel):
    """Daily activity plan"""
    day: int = Field(..., description="Day number")
    title: str = Field(..., description="Title for the day")
    activities: List[str] = Field(..., description="List of activities for the day")


class UserPreferences(BaseModel):
    """User-provided constraints that override video-derived suggestions."""

    start_date: Optional[date] = Field(
        None,
        description="Fixed trip start date (ISO YYYY-MM-DD)",
    )
    end_date: Optional[date] = Field(
        None,
        description="Fixed trip end date (ISO YYYY-MM-DD)",
    )
    flexible_month: Optional[str] = Field(
        None,
        description="Flexible month window (YYYY-MM) if exact dates are not known",
    )
    flexible_days: Optional[int] = Field(
        None,
        description="Number of days within the flexible month",
        ge=1,
        le=60,
    )
    budget: Optional[str] = Field(
        None,
        description="value | balanced | luxury",
    )
    pace: Optional[str] = Field(
        None,
        description="relaxed | active",
    )
    food_preference: Optional[str] = Field(
        None,
        description="vegan | vegetarian | all-eater",
    )
    extra_notes: Optional[str] = Field(
        None,
        description="Free-text notes such as start time, late nights, nightlife, etc.",
        max_length=800,
    )


class Itinerary(BaseModel):
    """Complete itinerary (Step 2 - Final Output)"""
    destination: str
    best_season: str
    duration_days: int
    daily_plan: List[DailyActivity]
    key_attractions: List[str]
    food_suggestions: List[str]
    start_date: Optional[date] = Field(None, description="Planned start date, if provided")
    end_date: Optional[date] = Field(None, description="Planned end date, if provided")
    flexible_month: Optional[str] = Field(None, description="Flexible month window if dates are not fixed")
    flexible_days: Optional[int] = Field(None, description="Requested days within the flexible month")
    budget: Optional[str] = Field(None, description="value | balanced | luxury")
    pace: Optional[str] = Field(None, description="relaxed | active")
    food_preference: Optional[str] = Field(None, description="vegan | vegetarian | all-eater")
    extra_notes: Optional[str] = Field(None, description="Free-text user notes")
