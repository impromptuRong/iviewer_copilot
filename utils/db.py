from sqlalchemy import Column, Float, String, Integer, Table, DateTime
from sqlalchemy.orm import DeclarativeBase
# from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime


class Base(DeclarativeBase):
    def to_dict(self):
        return {k: self.__dict__[k] for k in self.__table__.columns.keys()}
# Base = declarative_base()


class Annotation(Base):
    __tablename__ = 'annotation'

    id = Column(Integer, primary_key=True, index=True)
    x0 = Column(Float)
    y0 = Column(Float)
    x1 = Column(Float)
    y1 = Column(Float)
    xc = Column(Float)
    yc = Column(Float)
    poly_x = Column(String)
    poly_y = Column(String)
    label = Column(String, index=True)
    description = Column(String)
    annotator = Column(String, index=True)
    project_id = Column(String, index=True)
    group_id = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        item = super().to_dict()
        item['created_at'] = item['created_at'].isoformat() if item['created_at'] else None

        return item


class Cache(Base):
    __tablename__ = 'cache'

    id = Column(Integer, primary_key=True, index=True)
    registry = Column(String, index=True)
    tile_id = Column(String)
    # status = Column(String)
