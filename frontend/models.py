from dataclasses import dataclass


@dataclass
class Station:
    id: int
    name: str
    description: str
    created_at: str
    updated_at: str
    status: str
    lat: float
    lng: float
    last_temp: float
    predictions: list[float]
    preds_datetime: list[str]
    rules: list[tuple[int, str]]
    alerts: list[str]

