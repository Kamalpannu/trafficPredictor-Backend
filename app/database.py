import os
from sqlalchemy.orm import declarative_base
from dotenv import load_dotenv
from typing import AsyncGenerator

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
Base = declarative_base()

# Only create async engine if using asyncpg (for FastAPI runtime)
if DATABASE_URL and DATABASE_URL.startswith("postgresql+asyncpg://"):
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    engine = create_async_engine(DATABASE_URL, echo=True, future=True)
    SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
    async def get_db() -> AsyncGenerator["AsyncSession", None] :
        async with SessionLocal() as session:
            yield session
else:
    # For Alembic, just define Base
    engine = None
    SessionLocal = None