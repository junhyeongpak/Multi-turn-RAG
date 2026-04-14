"""
Elasticsearch 연결을 한 군데서만 만든다.
다른 모든 모듈/스크립트는 get_es() 를 호출해 동일 인스턴스를 받는다.
"""

from functools import lru_cache

from elasticsearch import Elasticsearch

from src.config import ES_API_KEY, ES_URL


@lru_cache(maxsize=1)
def get_es() -> Elasticsearch:
    if not ES_URL or not ES_API_KEY:
        raise EnvironmentError(
            "ES_URL / ES_API_KEY 가 설정되어 있지 않습니다. .env 파일을 확인하세요."
        )
    es = Elasticsearch(ES_URL, api_key=ES_API_KEY)
    # 연결 확인 (실패시 즉시 예외)
    es.info()
    return es


if __name__ == "__main__":
    es = get_es()
    print("ES connected:", es.info()["version"]["number"])
