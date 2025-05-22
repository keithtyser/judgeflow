from datetime import datetime
from typing import Optional
from sqlmodel import Field, SQLModel

class Score(SQLModel, table=True):
    """Model for storing evaluation scores from metrics."""
    id: Optional[int] = Field(default=None, primary_key=True)
    row_id: str = Field(index=True)
    metric: str = Field(index=True)
    score: float
    revised_score: Optional[float] = None
    revision_delta: Optional[float] = None
    critique: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        arbitrary_types_allowed = True 