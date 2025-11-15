from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from typing import List, Dict, Any
from pydantic import BaseModel
import json
import os

router = APIRouter()


class UpdateLastPositionRequest(BaseModel):
    """마지막 시청 지점 업데이트 요청"""
    last_position: float

# JSON 파일 경로
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
LECTURES_FILE = os.path.join(BASE_DIR, "data", "lectures.json")


def load_lectures() -> List[Dict]:
    """강의 목록 로드"""
    with open(LECTURES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_lectures(lectures: List[Dict]) -> None:
    """강의 목록 저장"""
    with open(LECTURES_FILE, "w", encoding="utf-8") as f:
        json.dump(lectures, f, ensure_ascii=False, indent=2)


def load_lecture_metadata(lecture_id: int) -> Dict:
    """특정 강의의 전체 데이터 로드 (프레임, 음성 등 모든 정보)"""
    lecture_file = os.path.join(BASE_DIR, "data", f"{lecture_id}", "lecture.json")
    with open(lecture_file, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("", response_model=List[Dict[str, Any]])
async def get_lectures():
    """
    모든 강의 목록 조회
    - 썸네일, 제목, 설명, 강사, 카테고리 등 포함
    """
    try:
        lectures = load_lectures()
        return lectures
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="강의 데이터를 찾을 수 없습니다")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="강의 데이터 형식이 올바르지 않습니다")


@router.get("/{lecture_id}/metadata", response_model=Dict[str, Any])
async def get_lecture_metadata(lecture_id: int):
    """
    영상 시청 시작 시 호출 - 영상 전체 데이터 조회
    - id, 제목, 전체 영상 길이
    - frames: 프레임 목록 (이름 + 시작시간)
    - narrations: 음성 나레이션 (오디오 파일명, 텍스트, 시작/종료 시간)
    - 영상 처음 들어갈 때 한 번만 호출
    """
    try:
        metadata = load_lecture_metadata(lecture_id)
        return metadata
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="강의 데이터를 찾을 수 없습니다")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="데이터 형식이 올바르지 않습니다")


@router.get("/{lecture_id}/frame/{frame_name}")
async def get_frame_image(lecture_id: int, frame_name: str):
    """
    프레임 이미지 요청 - 실제 이미지 파일 반환
    - 프론트에서 프레임 이름으로 요청하면 해당 이미지 파일 반환
    - 예: GET /api/lectures/1/frame/frame_266.jpg
    """
    try:
        frame_path = os.path.join(BASE_DIR, "data", f"{lecture_id}", "frames", frame_name)
        
        if not os.path.exists(frame_path):
            raise HTTPException(status_code=404, detail="프레임 이미지를 찾을 수 없습니다")
        
        return FileResponse(frame_path, media_type="image/jpeg")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 로드 중 오류 발생: {str(e)}")


@router.get("/{lecture_id}/audio/{audio_file}")
async def get_audio_file(lecture_id: int, audio_file: str):
    """
    오디오 파일 다운로드 - MP3 파일 반환
    - 프론트에서 오디오 파일명으로 요청하면 해당 MP3 파일 반환
    - 예: GET /api/lectures/1/audio/audio_0.mp3
    """
    try:
        audio_path = os.path.join(BASE_DIR, "data", f"{lecture_id}", "audio", audio_file)
        
        if not os.path.exists(audio_path):
            raise HTTPException(status_code=404, detail="오디오 파일을 찾을 수 없습니다")
        
        return FileResponse(audio_path, media_type="audio/mpeg")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"오디오 로드 중 오류 발생: {str(e)}")


@router.get("/{lecture_id}/thumbnail")
async def get_lecture_thumbnail(lecture_id: int):
    """
    강의 썸네일 이미지 요청
    - lecture_id가 1인 경우, 하드코딩된 썸네일 이미지 반환
    - 그 외의 경우 404 에러 반환
    """
    if lecture_id in [1,2,3]:
        thumbnail_path = os.path.join(BASE_DIR, "data", f"{lecture_id}", "frames", "frame_0.jpg")
        if not os.path.exists(thumbnail_path):
            raise HTTPException(status_code=404, detail="썸네일 이미지를 찾을 수 없습니다")
        return FileResponse(thumbnail_path, media_type="image/jpeg")
    else:
        raise HTTPException(status_code=404, detail="해당 강의의 썸네일을 찾을 수 없습니다")


@router.put("/{lecture_id}/last-position")
async def update_last_position(lecture_id: int, request: UpdateLastPositionRequest):
    """
    강의 마지막 시청 지점 업데이트
    - lecture_id에 해당하는 강의의 last_position을 업데이트
    - lectures.json과 lecture.json 파일 모두 업데이트
    - 업데이트된 전체 강의 메타데이터 반환
    
    Args:
        lecture_id: 강의 ID
        request: { last_position: float } - 마지막 시청 지점 (초 단위)
    
    Returns:
        업데이트된 전체 강의 메타데이터 (frames, narrations 등 포함)
    """
    try:
        # 1. lectures.json 업데이트
        lectures = load_lectures()
        lecture_found = False
        for lecture in lectures:
            if lecture["id"] == lecture_id:
                lecture["last_position"] = request.last_position
                lecture_found = True
                break
        
        if not lecture_found:
            raise HTTPException(status_code=404, detail=f"강의 ID {lecture_id}를 찾을 수 없습니다")
        
        save_lectures(lectures)
        
        # 2. lecture.json 업데이트
        lecture_file = os.path.join(BASE_DIR, "data", f"{lecture_id}", "lecture.json")
        with open(lecture_file, "r", encoding="utf-8") as f:
            lecture_metadata = json.load(f)
        
        lecture_metadata["last_position"] = request.last_position
        
        with open(lecture_file, "w", encoding="utf-8") as f:
            json.dump(lecture_metadata, f, ensure_ascii=False, indent=2)
        
        # 3. 업데이트된 전체 메타데이터 반환
        return lecture_metadata
    
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="강의 데이터를 찾을 수 없습니다")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="강의 데이터 형식이 올바르지 않습니다")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"시청 지점 업데이트 중 오류 발생: {str(e)}")
