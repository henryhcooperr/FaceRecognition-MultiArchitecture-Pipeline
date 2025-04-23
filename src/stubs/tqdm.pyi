from typing import Any, Iterable, Optional, TypeVar

T = TypeVar('T')

def tqdm(iterable: Iterable[T], desc: Optional[str] = None) -> Iterable[T]: ... 