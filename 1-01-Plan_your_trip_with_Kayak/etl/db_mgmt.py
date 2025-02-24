# -*- coding: utf-8 -*-
"""
Classes that represent the tables of the database with SQLAlchemy.
"""

from sqlalchemy import Table, ForeignKey
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped, mapped_column


class Base(DeclarativeBase):
    pass


# =============================================================================
# 
# =============================================================================

class Location(Base):
    """
    Declarative 'locations' table structure.
    """
    __tablename__ = "locations"
    
    location_id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    country: Mapped[str]
    latitude: Mapped[float]
    longitude: Mapped[float]
    
    def __repr__(self):
        str_ = (
            f"Location(location_id={self.location_id!r}, "
            f"name={self.name!r}, "
            f"country={self.country!r}, "
            f"latitude={self.latitude!r}, "
            f"longitude={self.longitude!r})"
        )
        return str_


class WeatherIndicator(Base):
    """
    Declarative 'weather_indicators' table structure.
    """
    __tablename__ = "weather_indicators"
    
    location_id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[str]
    min_temperature_cels: Mapped[float | None]
    max_temperature_cels: Mapped[float | None]
    sunshine_duration_h: Mapped[float | None]
    precipitation_sum_mm: Mapped[float | None]
    
    def __repr__(self):
        str_ = (
            f"WeatherIndicator(location_id={self.location_id!r}, "
            f"start_date={self.start_date!r}, "
            f"min_temperature_cels={self.min_temperature_cels!r}, "
            f"max_temperature_cels={self.max_temperature_cels!r}, "
            f"sunshine_duration_h={self.sunshine_duration_h!r}, "
            f"precipitation_sum_mm={self.precipitation_sum_mm!r})"
        )
        return str_
    

class Hotel(Base):
    """
    Declarative 'hotels' table structure.
    """
    __tablename__ = "hotels"
    
    hotel_id: Mapped[int] = mapped_column(primary_key=True)
    location_id: Mapped[int] = mapped_column(ForeignKey("locations.location_id"))
    url: Mapped[str]
    name: Mapped[str]
    description: Mapped[str | None]
    rating: Mapped[float | None]
    georating: Mapped[float | None]
    
    def __repr__(self):
        str_ = (
            f"Hotel(hotel_id={self.hotel_id!r}, "
            f"location_id={self.location_id!r}, "
            f"url={self.url!r}, "
            f"name={self.name!r}, "
            f"description={self.description!r}, "
            f"rating={self.rating!r}, "
            f"georating={self.georating!r})"
        )
        return str_


def reflect_db(metadata_obj, engine)-> tuple[Table, Table, Table]:
    """
    Create tables by loading information from database.
    """
    locations = Table("locations", metadata_obj, autoload_with=engine)
    weather_indicators = Table("weather_indicators", metadata_obj,
                               autoload_with=engine)
    hotels = Table("hotels", metadata_obj, autoload_with=engine)
    return (locations, weather_indicators, hotels)