# Qdrant 공용 서버 사용법

## 서버 시작

```bash
# 프로젝트 루트에서 실행
cd /Users/junseok/Projects/SKN21-3rd-1TEAM
docker-compose up -d
```

## 서버 상태 확인

```bash
# 브라우저에서 확인
open http://localhost:6333/dashboard

# 또는 컬렉션 목록 확인
curl http://localhost:6333/collections
```

## 팀별 사용법

### A-TEAM

```python
import sys
sys.path.append('/Users/junseok/Projects/SKN21-3rd-1TEAM/shared')
from qdrant_config import get_client, get_team_collection_name

client = get_client()

# A팀 컬렉션 이름
collection = get_team_collection_name('a', 'labor_laws')  # -> 'a_labor_laws'
```

### B-TEAM

```python
# 같은 방식으로 'b_' 접두사 사용
collection = get_team_collection_name('b', 'labor_laws')  # -> 'b_labor_laws'
```

## 팀원 접속 (같은 WiFi)

```python
import os
os.environ['QDRANT_HOST'] = '192.168.x.x'  # 서버 호스트 IP

from qdrant_config import get_client
client = get_client()
```

## 컬렉션 네이밍 규칙

| 팀     | 컬렉션 예시                                              |
| ------ | -------------------------------------------------------- |
| A-TEAM | `a_labor_laws`, `a_civil_laws`, `a_moel_interpretations` |
| B-TEAM | `b_labor_laws`, `b_civil_laws`, `b_moel_interpretations` |
