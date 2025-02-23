#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import sys
import csv
import time

from sqlalchemy import create_engine, URL, inspect, Table
from sqlalchemy import select, text
from sqlalchemy import Integer, String, Float, ForeignKey
from sqlalchemy.orm import Session
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


"""
Creation of PostgreSQL database container and start server
$ initdb -D <name>
$ pg_ctl -D <name> -l logfile start

Create user and database
$ createuser <username>
$ createdb --owner=<username> <dbname>

Connect with
"postgresql://<usenamer>@<hostname>:<port>/<dbname>"

Stop server
$ ps aux | grep postgres
$ kill </postgres -D <name> PID>
"""



# =============================================================================
# 
# =============================================================================

class Location(Base):
    """
    Declarative 'locations' table structure.
    """
    __tablename__ = "locations"
    
    location_id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String)
    country: Mapped[str] = mapped_column(String)
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    
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
    start_date: Mapped[str] = mapped_column()
    min_temperature_C: Mapped[float | None]
    max_temperature_C: Mapped[float | None]
    sunshine_duration_h: Mapped[float | None]
    precipitation_sum_mm: Mapped[float | None]
    
    def __repr__(self):
        str_ = (
            f"WeatherIndicator(location_id={self.location_id!r}, "
            f"start_date={self.start_date!r}, "
            f"min_temperature_C={self.min_temperature_C!r}, "
            f"max_temperature_C={self.max_temperature_C!r}, "
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
    name: Mapped[str] = mapped_column(String)
    description: Mapped[str | None] = mapped_column(String)
    rating: Mapped[float | None] = mapped_column(String)
    georating: Mapped[float | None] = mapped_column(Integer)
    
    def __repr__(self):
        str_ = (
            f"Hotel(hotel_id={self.hotel_id!r}, "
            f"location_id={self.location_id!r}, "
            f"url={self.url!r}, "
            f"name={self.name!r}, "
            f"description={self.description!r}, "
            f"latitude={self.latitude!r}, "
            f"longitude={self.longitude!r})"
        )
        return str_


def reflect_db(metadata_obj, engine)-> tuple[Table, Table, Table]:
    """
    Create tables by loading information from database.
    """
    locations = Table("locations", metadata_obj, autoload_with=engine)
    weather_indicators = Table("weather_indicators", metadata_obj, autoload_with=engine)
    hotels = Table("hotels", metadata_obj, autoload_with=engine)
    return (locations, weather_indicators, hotels)


# =============================================================================
# 
# =============================================================================

# with open('../neondb_access_keys.key', 'rt', encoding='utf-8') as f:
#     PGHOST = f.readline().split("'")[1]
#     PGDATABASE = f.readline().split("'")[1]
#     PGUSER = f.readline().split("'")[1]
#     PGPASSWORD = f.readline().split("'")[1]

# url = URL.create(
#     "postgresql+psycopg",
#     username=PGUSER,
#     password=PGPASSWORD,
#     host=PGHOST,
#     database=PGDATABASE,
# )

# # engine = create_engine(url, echo=False)
# # inspector = inspect(engine)






# # Load locations of interest
# with open("../data/locations.csv", 'rt', encoding='utf-8') as f:
#     reader = csv.reader(f, delimiter=';')
#     next(reader, None) # remove header
#     locations = [Location(location_id=row[0], name=row[1], country=row[2],
#                           latitude=row[3], longitude=row[4])
#                  for row in reader]

# weather_indicators = []

# hotels = []

# # sys.exit()

# engine = create_engine("sqlite+pysqlite:///:memory:", echo=True)
# inspector = inspect(engine)
# Base.metadata.create_all(engine)

# with Session(engine) as session:
#     session.add_all(locations)
#     session.add_all(weather_indicators)
#     session.add_all(hotels)
#     session.commit()

# import pandas as pd

# a = pd.read_sql(select(Location.__table__), con=engine)


# Location.__table__.drop(engine)

# print(Base.metadata.tables.keys())

# with engine.connect() as conn:
#     result = conn.execute(text("select 'hello world'"))
#     a = result.all()
#     print('res', result.all())


# inspector.get_table_names(schema='main')




