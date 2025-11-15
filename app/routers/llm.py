"""
LLM API 라우터
RAG + ChatGPT 기반 질문 응답 API
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
from pathlib import Path
import json
import hashlib
from app.services.llm_service import LLMService

router = APIRouter()

# LLM 서비스 캐싱 (강의별)
_llm_service_cache: Dict[int, LLMService] = {}


def get_llm_service(lecture_id: int) -> LLMService:
    """서비스 캠싱 및 재사용"""
    if lecture_id not in _llm_service_cache:
        _llm_service_cache[lecture_id] = LLMService(lecture_id)
    return _llm_service_cache[lecture_id]


def _get_current_narration(lecture_id: int, audio_index: int) -> Optional[str]:
    """현재 재생 중인 음성의 텍스트 가져오기"""
    try:
        lecture_file = Path(f"app/data/{lecture_id}/lecture.json")
        with open(lecture_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            narrations = data.get("narrations", [])
            if 0 <= audio_index < len(narrations):
                return narrations[audio_index].get("text", "")
    except Exception:
        pass
    return None


def _get_next_narration(lecture_id: int, audio_index: int) -> Optional[str]:
    """다음에 재생될 음성의 텍스트 가져오기"""
    try:
        lecture_file = Path(f"app/data/{lecture_id}/lecture.json")
        with open(lecture_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            narrations = data.get("narrations", [])
            next_index = audio_index + 1
            if 0 <= next_index < len(narrations):
                return narrations[next_index].get("text", "")
    except Exception:
        pass
    return None


async def _generate_tts_for_answer(answer: str, lecture_id: int, speaking_rate: float = 1.05) -> Optional[str]:
    """LLM 답변을 TTS로 변환하여 저장
    
    Args:
        answer: 답변 텍스트
        lecture_id: 강의 ID
        speaking_rate: 음성 속도 (0.25~4.0, 기본 1.05)
    """
    from google.cloud.texttospeech import TextToSpeechClient, SynthesisInput, VoiceSelectionParams, AudioConfig, AudioEncoding
    from dotenv import load_dotenv
    
    load_dotenv()
    
    try:
        # TTS 클라이언트 초기화
        client = TextToSpeechClient()
        
        # 파일명 생성 (답변 해시값 사용)
        answer_hash = hashlib.md5(answer.encode()).hexdigest()[:8]
        filename = f"llm_answer_{answer_hash}.mp3"
        audio_dir = Path(f"app/data/{lecture_id}/audio")
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = audio_dir / filename
        
        # 이미 생성된 파일이 있으면 재사용
        if audio_path.exists():
            return f"/api/lectures/{lecture_id}/audio/{filename}"
        
        # TTS 설정
        synthesis_input = SynthesisInput(text=answer)
        voice = VoiceSelectionParams(
            language_code="ko-KR",
            name="ko-KR-Wavenet-D"
        )
        audio_config = AudioConfig(
            audio_encoding=AudioEncoding.MP3,
            speaking_rate=speaking_rate,
            pitch=0.0
        )
        
        # TTS 생성
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        # 파일 저장
        with open(audio_path, "wb") as out:
            out.write(response.audio_content)
        
        return f"/api/lectures/{lecture_id}/audio/{filename}"
        
    except Exception as e:
        print(f"TTS 생성 실패: {e}")
        return None


class QuestionRequest(BaseModel):
    """질문 요청 모델"""
    question: str
    lecture_id: int = 1
    current_audio_index: Optional[int] = None  # 현재 재생 중인 음성 인덱스
    condition: Optional[int] = 2  # 사용자 상태: 1(안좋음), 2(정상), 3(좋음)


class QuestionResponse(BaseModel):
    """질문 응답 모델"""
    answer: str
    audio_url: Optional[str] = None  # TTS 생성된 오디오 URL


class SentenceResponse(BaseModel):
    """문장 분리 응답 모델 (TTS용)"""
    sentences: List[str]


class TextOnlyResponse(BaseModel):
    """텍스트만 응답 모델"""
    answer: str


@router.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    강의 관련 질문에 답변 (TTS 포함)
    
    - 현재 재생 중인 음성 정보를 받아 맥락 유지
    - RAG를 통해 관련 문서 검색
    - 페르소나를 적용한 ChatGPT 답변 생성
    - TTS로 음성 생성 후 URL 반환
    """
    try:
        # LLM 서비스 가져오기 (캐싱)
        llm_service = get_llm_service(request.lecture_id)
        
        # 현재 재생 중인 내용 가져오기
        current_audio_context = None
        if request.current_audio_index is not None:
            current_audio_context = _get_current_narration(request.lecture_id, request.current_audio_index)
        
        # RAG + LLM 파이프라인 실행 (현재 맥락 전달, condition 전달)
        result = llm_service.ask(
            question=request.question,
            current_audio_context=current_audio_context,
            condition=request.condition
        )
        answer = result["answer"]
        
        # TTS 생성 (condition 1일 때 속도 느리게)
        speaking_rate = 0.9 if request.condition == 1 else 1.05
        audio_url = await _generate_tts_for_answer(answer, request.lecture_id, speaking_rate)
        
        return QuestionResponse(
            answer=answer,
            audio_url=audio_url
        )
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"강의 데이터를 찾을 수 없습니다: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM 처리 중 오류 발생: {str(e)}"
        )


@router.post("/ask-split", response_model=SentenceResponse)
async def ask_question_with_split(request: QuestionRequest):
    """
    강의 관련 질문에 답변 (문장 분리 포함)
    
    - RAG를 통해 관련 문서 검색
    - 페르소나를 적용한 ChatGPT 답변 생성
    - TTS 생성을 위해 문장 단위로 분리
    """
    try:
        # LLM 서비스 가져오기 (캐싱)
        llm_service = get_llm_service(request.lecture_id)
        
        # 현재 재생 중인 내용 가져오기
        current_audio_context = None
        if request.current_audio_index is not None:
            current_audio_context = _get_current_narration(request.lecture_id, request.current_audio_index)
        
        # RAG + LLM 파이프라인 실행 (문장 분리 포함, condition 전달)
        result = llm_service.ask_and_split(
            question=request.question,
            current_audio_context=current_audio_context,
            condition=request.condition
        )
        
        return SentenceResponse(
            sentences=result["sentences"]
        )
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"강의 데이터를 찾을 수 없습니다: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM 처리 중 오류 발생: {str(e)}"
        )


@router.post("/ask-text", response_model=TextOnlyResponse)
async def ask_question_text_only(request: QuestionRequest):
    """
    강의 관련 질문에 텍스트 답변만 반환 (TTS 없음)
    
    - 현재 재생 중인 음성 정보를 받아 맥락 유지
    - RAG를 통해 관련 문서 검색
    - 페르소나를 적용한 ChatGPT 답변 생성
    - 텍스트만 반환 (음성 생성 없음, 전환 멘트 제외)
    """
    try:
        # LLM 서비스 가져오기 (캐싱)
        llm_service = get_llm_service(request.lecture_id)
        
        # 현재 재생 중인 내용 가져오기
        current_audio_context = None
        if request.current_audio_index is not None:
            current_audio_context = _get_current_narration(request.lecture_id, request.current_audio_index)
        
        # RAG + LLM 파이프라인 실행 (현재 맥락 전달, 전환 멘트 제외, condition 전달)
        result = llm_service.ask(
            question=request.question,
            current_audio_context=current_audio_context,
            include_transition=False,  # 텍스트 전용이므로 전환 멘트 제외
            condition=request.condition
        )
        
        return TextOnlyResponse(
            answer=result["answer"]
        )
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"강의 데이터를 찾을 수 없습니다: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM 처리 중 오류 발생: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """LLM 서비스 상태 확인"""
    return {
        "status": "healthy",
        "service": "LLM API"
    }
