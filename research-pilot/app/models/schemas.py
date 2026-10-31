from pydantic import BaseModel, Field


class Paper(BaseModel):
    title: str
    # 自动把 API 返回的 publication_year 映射为 year
    year: int | None = Field(default=None, validation_alias="publication_year")
    doi: str | None = None
    cited_by_count: int = Field(default=0, ge=0)
    # 自动把 API 返回的 id 映射为 url
    url: str = Field(validation_alias="id")


class TaskPlan(BaseModel):
    steps: list[str] = Field(min_length=1, max_length=6)