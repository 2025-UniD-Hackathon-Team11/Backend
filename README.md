# AI Lecture Platform API

3D 모션 기반 AI 강의 플랫폼 백엔드 서버

## 아키텍처 개요

- **영상 방식**: PPT처럼 정적 프레임들로 구성 (움직이는 객체 없음)
- **프레임 전달**: 처음에 모든 고유 프레임 이미지를 내려주고, 시간 매핑 정보로 프론트에서 프레임 전환
- **데이터 저장**: DB 없이 JSON 파일로 간단하게 관리 (해커톤용)

## 설치

```bash
pip install -r requirements.txt
```

## 실행

```bash
uvicorn app.main:app --reload
```

## API 문서

서버 실행 후 다음 URL에서 확인 가능합니다:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API 엔드포인트

### Lectures

#### 1. 강의 목록 조회
```
GET /api/lectures
```
- 모든 강의 리스트 반환
- 썸네일, 제목, 설명, 강사, 카테고리, 재생시간 포함

**응답 예시:**
```json
[
  {
    "id": 1,
    "title": "파이썬 기초 - 변수와 자료형",
    "description": "파이썬의 기본 변수와 자료형에 대해 배웁니다",
    "thumbnail": "lecture_1_thumb.jpg",
    "duration": 300,
    "instructor": "김코딩",
    "category": "Programming"
  }
]
```

#### 2. 강의 상세 정보 조회
```
GET /api/lectures/{lecture_id}
```
- 강의 기본 정보 + 모든 프레임 정보 반환
- 프론트에서 영상 재생 시 필요한 모든 데이터 한 번에 제공

**응답 예시:**
```json
{
  "id": 1,
  "title": "파이썬 기초 - 변수와 자료형",
  "description": "파이썬의 기본 변수와 자료형에 대해 배웁니다",
  "thumbnail": "lecture_1_thumb.jpg",
  "duration": 300,
  "instructor": "김코딩",
  "category": "Programming",
  "frames": [
    {
      "frame_image": "lecture_1_frame_001.jpg",
      "start_time": 0.0,
      "end_time": 15.5
    },
    {
      "frame_image": "lecture_1_frame_002.jpg",
      "start_time": 15.5,
      "end_time": 45.2
    }
  ]
}
```

#### 3. 강의 프레임 정보만 조회
```
GET /api/lectures/{lecture_id}/frames
```
- 프레임 정보만 필요한 경우 사용

**응답 예시:**
```json
{
  "lecture_id": 1,
  "frames": [
    {
      "frame_image": "lecture_1_frame_001.jpg",
      "start_time": 0.0,
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
