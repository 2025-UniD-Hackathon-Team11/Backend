# Backend API

## 로컬 실행

1. **환경 설정**
```bash
pip install -r requirements.txt
```

2. **환경 변수 설정**
- `.env.example`을 `.env`로 복사
- `OPENAI_API_KEY` 설정
- `credential.json` (Google Cloud TTS) 파일 추가

3. **서버 실행**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Docker 실행

1. **이미지 빌드**
```bash
docker build -t backend-api:latest .
```

2. **컨테이너 실행**
```bash
docker run -d --name backend-api -p 8000:8000 backend-api:latest
```

3. **로그 확인**
```bash
docker logs -f backend-api
```

## API 문서

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 프로젝트 구조

```
Backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 앱 초기화
│   ├── routers/
│   │   ├── __init__.py
│   │   └── lectures.py      # 강의 관련 API
│   └── data/
│       ├── lectures.json    # 강의 목록 데이터
│       └── frames.json      # 프레임 매핑 데이터
├── requirements.txt
└── README.md
```
