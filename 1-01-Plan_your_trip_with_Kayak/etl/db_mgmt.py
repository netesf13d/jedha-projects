#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import psycopg
from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import declarative_base, Session


Base = declarative_base()


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

class Customer(Base):
    __tablename__ = "customers"
    
    id = Column(Integer, primary_key=True)
    name = Column(String)
    country = Column(String)
    job = Column(String)
    age = Column(Integer)
    
    
    def __repr__(self):
        str_ = (
            "<"
            f"User(id={self.id}, "
            f"name={self.name}, "
            f"country={self.country}, "
            f"job={self.job}, "
            f"age={self.age}"
            ">"
        )
        return str_


# =============================================================================
# 
# =============================================================================

host = "127.0.0.1"
port = 5432
user = "user"
passwd = ""
dbname = "userdb"

sqlalchemy_driver = "+psycopg2"
conn_str = f"postgresql{sqlalchemy_driver}://{user}@{host}:{port}/{dbname}"


engine = create_engine(conn_str, echo=True)

Base.metadata.create_all(engine)


# =============================================================================
# 
# =============================================================================

customers = [
    Customer(id=1, name="Sauerkraut", country="Germany", job="engineer", age=37),
    Customer(id=2, name="Jones", country="United Kingdom", job="journalist", age=52),
    Customer(id=3, name="Dupont", country="France", job="dancer", age=25),
]

with Session(engine) as session:
    for customer in customers:
        session.add(customer)
    session.commit()
    
    customer = session.query(Customer)
    print(customer.all())




# conn = psycopg2.connect(conn_str)
# conn = psycopg2.connect(dbname="userdb", host="127.0.0.1", port=5432, user="user")





