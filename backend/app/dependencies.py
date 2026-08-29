from fastapi import Request

from .data.source import RawTableStore
from .scorer import Scorer


def get_store(request: Request) -> RawTableStore:
    return request.app.state.store


def get_scorer(request: Request) -> Scorer:
    return request.app.state.scorer
