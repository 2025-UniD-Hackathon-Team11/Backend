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

      "end_time": 15.5
    }
  ]
}
```

## 데이터 구조

### `app/data/lectures.json`
강의 메타데이터 관리

### `app/data/frames.json`
각 강의의 프레임 이미지와 타임스탬프 매핑 정보
- `frame_image`: 고유 프레임 이미지 파일명
- `start_time`: 프레임 시작 시간(초)
- `end_time`: 프레임 종료 시간(초)

## 프론트엔드 구현 가이드

1. 강의 목록 페이지: `GET /api/lectures` 호출하여 리스트 표시
2. 강의 선택 시: `GET /api/lectures/{lecture_id}` 호출
3. 모든 프레임 이미지 미리 로드
4. 현재 재생 시간에 따라 해당 프레임 표시
   - 예: 현재 시간 20초 → `start_time <= 20 < end_time` 인 프레임 찾아서 표시

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
