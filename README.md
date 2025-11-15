# Backend API

FastAPI 기반 백엔드 서버

## 프로젝트 구조

```
Backend/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── __init__.py
│   │       └── health.py       # Health check 엔드포인트
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py           # 설정 관리
│   ├── models/                 # 데이터베이스 모델
│   ├── schemas/                # Pydantic 스키마
│   ├── services/               # 비즈니스 로직
│   └── main.py                 # FastAPI 애플리케이션
├── tests/
│   ├── __init__.py
│   └── test_main.py            # 테스트 파일
├── .env.example                # 환경 변수 예시
├── .gitignore
├── docker-compose.yml          # Docker Compose 설정
├── Dockerfile                  # Docker 이미지 설정
├── pyproject.toml              # 프로젝트 설정
├── requirements.txt            # 프로덕션 의존성
├── requirements-dev.txt        # 개발 의존성
└── README.md
```

## 기능

- ✅ FastAPI 프레임워크
- ✅ 자동 API 문서화 (Swagger UI, ReDoc)
- ✅ Pydantic를 통한 데이터 검증
- ✅ CORS 미들웨어
- ✅ 환경 변수 관리
- ✅ Docker 지원
- ✅ 테스트 설정
- ✅ 코드 포맷팅 (Black, Ruff)

## 시작하기

### 필수 요구사항

- Python 3.11+
- pip

### 설치

1. 저장소 클론
```bash
git clone <repository-url>
cd Backend
```

2. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 의존성 설치
```bash
pip install -r requirements.txt
# 개발 환경
pip install -r requirements-dev.txt
```

4. 환경 변수 설정
```bash
cp .env.example .env
# .env 파일을 편집하여 필요한 설정을 변경하세요
```

### 실행

#### 로컬 개발 서버
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Docker로 실행
```bash
docker-compose up --build
```

서버가 실행되면 다음 주소에서 접근할 수 있습니다:
- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API 엔드포인트

### Root
- `GET /` - 기본 정보

### Health Check
- `GET /api/v1/health` - 서버 상태 확인

### API 문서
- `GET /docs` - Swagger UI
- `GET /redoc` - ReDoc
- `GET /openapi.json` - OpenAPI 스키마

## 테스트

```bash
pytest
```

특정 테스트 파일 실행:
```bash
pytest tests/test_main.py
```

커버리지와 함께 실행:
```bash
pytest --cov=app tests/
```

## 코드 포맷팅 및 린팅

```bash
# Black으로 코드 포맷팅
black app/ tests/

# Ruff로 린팅
ruff check app/ tests/

# Ruff로 자동 수정
ruff check --fix app/ tests/
```

## 개발 가이드

### 새로운 엔드포인트 추가

1. `app/api/v1/` 디렉토리에 새 파일 생성 (예: `users.py`)
2. 라우터 정의 및 엔드포인트 구현
3. `app/api/v1/__init__.py`에 라우터 추가

예시:
```python
# app/api/v1/users.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/users")
async def get_users():
    return {"users": []}

# app/api/v1/__init__.py
from app.api.v1 import health, users

api_router.include_router(users.router, prefix="/users", tags=["users"])
```

### 환경 변수 추가

1. `app/core/config.py`의 `Settings` 클래스에 변수 추가
2. `.env.example` 파일 업데이트

## 배포

### Docker를 사용한 배포

```bash
# 이미지 빌드
docker build -t backend-api .

# 컨테이너 실행
docker run -p 8000:8000 --env-file .env backend-api
```

### 프로덕션 설정

프로덕션 환경에서는:
- `.env` 파일에서 `DEBUG=False` 설정
- 적절한 CORS origins 설정
- 데이터베이스 연결 설정
- 로깅 구성
- HTTPS 사용

## 라이선스

이 프로젝트는 LICENSE 파일에 명시된 라이선스를 따릅니다.