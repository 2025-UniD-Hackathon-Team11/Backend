"""
LLM 서비스
RAG + 페르소나 + ChatGPT API 통합
"""

import os
import json
from typing import Dict, List, Optional
from openai import OpenAI
from app.services.rag_service import create_rag_service


# 시스템 프롬프트 (모든 강의에 공통 적용)
SYSTEM_PROMPT = """당신은 학생들을 가르치는 전문 강사입니다. 다음 규칙을 반드시 따르세요:

## 핵심 원칙

1. **강의 맥락 유지**: 학생에게 현재 진행 중인 강의 내용이 제공됩니다. 질문이 현재 강의 내용과 관련이 있다면, 해당 맥락을 이어가며 답변하세요.

2. **관련성 검증**: 질문이 강의 주제와 전혀 관련이 없는 경우, 정중하게 거절하세요. 다만 STT를 이용하기 때문에 조금의 오류가 있을 수 있음을 감안하여, 질문이 강의와 관련이 없다고 판단될 때만 거절합니다.
   - 예: "죄송하지만 그 질문은 현재 강의 주제와 관련이 없어요. 강의 내용에 대해 질문해 주시면 답변 드리겠습니다. 그럼 강의를 계속 들어볼까요?"

3. **답변 후 전환**: 질문에 답변한 후에는 반드시 자연스럽게 강의로 돌아가는 전환 표현을 사용하세요. 매번 다른 표현을 사용하여 자연스럽게 변화를 주세요.
   - 전환 표현 예시: "그럼 강의를 계속 들어볼까요?", "이어서 들어보시죠!", "자, 계속 진행해볼까요?", "다시 강의로 돌아가 볼까요?"

## 답변 형식 (TTS 최적화)

- **형식**: 반드시 하나의 단락으로만 구성
- **분량**: 2-3문장 (최대 5문장 이내)
- **내용**: 질문의 핵심만 간결하게 답변
- **어투**: 친근한 한국어 (-요 70%, -니다 30%)
- **금지 요소**: 특수문자, 기호, 따옴표 반복, 코드블록, 이모티콘, 줄바꿈, 불필요한 목록 표시, 괄호 남용, 영어 축약어, 과도한 감탄 표현

## 답변 가이드라인

1. **명확성**: 전문 용어는 쉬운 말로 풀어서 설명하세요.
2. **구조화**: 복잡한 개념은 단계별로 나누어 설명하세요.
3. **예시 활용**: 추상적인 개념은 구체적인 예시와 함께 설명하세요.
4. **학생 중심**: 학생의 질문 의도를 파악하고 그에 맞춰 답변하세요.
5. **정확성**: 제공된 강의 자료를 기반으로 답변하세요. 확실하지 않은 내용은 솔직하게 인정하세요.
"""


class LLMService:
    """ChatGPT API + RAG + 페르소나 통합 서비스"""
    
    def __init__(self, lecture_id: int):
        """
        Args:
            lecture_id: 강의 ID
        """
        self.lecture_id = lecture_id
        self.rag_service = create_rag_service(lecture_id)
        
        # OpenAI 클라이언트 초기화
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        
        self.client = OpenAI(api_key=api_key)
        
        # 페르소나 설정 로드
        self._load_persona_config()
    
    def _load_persona_config(self):
        """페르소나 설정 파일 로드"""
        persona_file = f"app/data/{self.lecture_id}/llm/persona.json"
        
        if not os.path.exists(persona_file):
            raise FileNotFoundError(f"페르소나 파일을 찾을 수 없습니다: {persona_file}")
        
        with open(persona_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.persona_text = config.get("persona", "")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 500)
    
    def _build_system_prompt(self) -> str:
        """
        최종 시스템 프롬프트 생성 (시스템 프롬프트 + 페르소나)
        
        Returns:
            완성된 시스템 프롬프트
        """
        return f"{SYSTEM_PROMPT}\n\n## 당신의 역할\n{self.persona_text}"
    
    def _build_user_message(
        self, 
        question: str, 
        rag_context: str,
        current_audio_context: Optional[str] = None
    ) -> str:
        """
        사용자 메시지 구성
        
        Args:
            question: 사용자 질문
            rag_context: RAG로 검색된 컨텍스트
            current_audio_context: 현재 재생 중인 강의 내용
        
        Returns:
            구조화된 사용자 메시지
        """
        message_parts = []
        
        # 현재 강의 맥락 추가
        if current_audio_context:
            message_parts.append(f"[현재 학생이 듣고 있는 강의 내용]\n{current_audio_context}\n")
        
        # RAG 컨텍스트 추가
        if rag_context and rag_context != "관련 문서를 찾을 수 없습니다.":
            message_parts.append(f"[관련 강의 자료]\n{rag_context}\n")
        
        # 구분선 및 질문
        message_parts.append("---")
        message_parts.append(f"\n학생 질문: {question}")
        
        return "\n".join(message_parts)
    
    def ask(
        self, 
        question: str, 
        rag_top_k: int = 3,
        model: str = "gpt-4o-mini",
        current_audio_context: Optional[str] = None,
        include_transition: bool = True,
        condition: Optional[int] = 2
    ) -> Dict:
        """
        질문에 대한 답변 생성 (RAG + ChatGPT)
        
        Args:
            question: 사용자 질문
            rag_top_k: RAG 검색 시 가져올 문서 청크 수
            model: OpenAI 모델 (gpt-4o, gpt-4o-mini, gpt-3.5-turbo)
            current_audio_context: 현재 재생 중인 강의 내용
            include_transition: 강의 전환 멘트 포함 여부 (기본값: True)
            condition: 사용자 상태 (1: 안좋음, 2: 정상, 3: 좋음)
        
        Returns:
            {
                "answer": "답변 텍스트",
                "rag_context": "사용된 컨텍스트",
                "sources": ["출처1", "출처2"],
                "model": "사용된 모델",
                "tokens": {"prompt": 100, "completion": 200, "total": 300}
            }
        """
        # 1. RAG로 관련 문서 검색
        rag_data = self.rag_service.prepare_llm_request(question, top_k=rag_top_k)
        rag_context = rag_data["context"]
        
        # 2. 프롬프트 구성 (전환 멘트 여부 및 condition에 따라)
        if include_transition:
            system_prompt = self._build_system_prompt()
        else:
            # 전환 멘트 없는 버전 (텍스트 전용)
            system_prompt = SYSTEM_PROMPT.replace(
                '3. **답변 후 전환**: 질문에 답변한 후에는 반드시 자연스럽게 강의로 돌아가는 전환 표현을 사용하세요. 매번 다른 표현을 사용하여 자연스럽게 변화를 주세요.\n   - 전환 표현 예시: "그럼 강의를 계속 들어볼까요?", "이어서 들어보시죠!", "자, 계속 진행해볼까요?", "다시 강의로 돌아가 볼까요?"',
                '3. **답변 완결성**: 질문에 대한 답변만 제공하세요. 추가적인 전환 표현이나 강의 복귀 멘트는 필요 없습니다.'
            ) + f"\n\n## 당신의 역할\n{self.persona_text}"
        
        # condition에 따른 답변 스타일 조정
        if condition == 1:
            # 컨디션 안좋음: 짧고 간단하게
            system_prompt += "\n\n## 특별 지시\n사용자의 컨디션이 좋지 않습니다. 답변을 **반드시 1-2문장 이내로** 작성하세요. 핵심만 간결하게 전달하고, 부가 설명이나 예시는 절대 포함하지 마세요. 짧고 명확한 답변만 제공하세요."
        elif condition == 3:
            # 컨디션 좋음: 자세하고 길게
            system_prompt += "\n\n## 특별 지시\n사용자의 컨디션이 매우 좋습니다. 답변을 **4-6문장으로** 작성하세요. 개념 설명, 구체적인 예시, 추가 설명을 포함하여 깊이 있고 상세한 답변을 제공하세요."
        
        # max_tokens는 기본값 사용 (답변이 잘리는 것 방지)
        max_tokens = self.max_tokens
        
        user_message = self._build_user_message(question, rag_context, current_audio_context)
        
        # 3. ChatGPT API 호출
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.temperature,
                max_tokens=max_tokens
            )
            
            answer = response.choices[0].message.content
            
            # 4. 출처 정보 추출
            sources = []
            if "persona" in rag_data and "name" in rag_data["persona"]:
                sources.append(rag_data["persona"]["name"])
            
            return {
                "answer": answer,
                "rag_context": rag_context,
                "sources": sources,
                "model": model,
                "tokens": {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens
                }
            }
            
        except Exception as e:
            raise Exception(f"ChatGPT API 호출 실패: {str(e)}")
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        답변을 문장 단위로 분리
        
        Args:
            text: 전체 답변 텍스트
        
        Returns:
            문장 리스트
        """
        import re
        
        # 한국어 문장 분리 (., !, ? 기준)
        sentences = re.split(r'([.!?]\s+)', text)
        
        # 구두점과 문장 다시 결합
        result = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")
            sentence = sentence.strip()
            if sentence:
                result.append(sentence)
        
        # 마지막 문장 처리
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            result.append(sentences[-1].strip())
        
        return result
    
    def ask_and_split(
        self, 
        question: str, 
        rag_top_k: int = 3,
        model: str = "gpt-4o-mini",
        current_audio_context: Optional[str] = None,
        include_transition: bool = True,
        condition: Optional[int] = 2
    ) -> Dict:
        """
        질문에 대한 답변 생성 후 문장 단위로 분리
        
        Args:
            question: 사용자 질문
            rag_top_k: RAG 검색 시 가져올 문서 청크 수
            model: OpenAI 모델
            current_audio_context: 현재 재생 중인 강의 내용
            include_transition: 강의 전환 멘트 포함 여부 (기본값: True)
            condition: 사용자 상태 (1: 안좋음, 2: 정상, 3: 좋음)
        
        Returns:
            {
                "sentences": ["문장1", "문장2", ...],
                "full_answer": "전체 답변",
                "rag_context": "사용된 컨텍스트",
                "sources": ["출처1", "출처2"],
                "model": "사용된 모델",
                "tokens": {...}
            }
        """
        # 답변 생성 (전환 멘트 여부 및 condition 전달)
        result = self.ask(question, rag_top_k, model, current_audio_context, include_transition, condition)
        
        # 문장 분리
        sentences = self.split_into_sentences(result["answer"])
        
        return {
            "sentences": sentences,
            "full_answer": result["answer"],
            "rag_context": result["rag_context"],
            "sources": result["sources"],
            "model": result["model"],
            "tokens": result["tokens"]
        }


def create_llm_service(lecture_id: int) -> LLMService:
    """
    LLM 서비스 인스턴스 생성
    
    Args:
        lecture_id: 강의 ID
    
    Returns:
        LLMService 인스턴스
    """
    return LLMService(lecture_id)


# 사용 예시
if __name__ == "__main__":
    # 강의 1번의 LLM 서비스 생성
    llm = create_llm_service(lecture_id=1)
    
    # 질문
    question = "컨볼루션이 뭔가요?"
    
    # 답변 생성 (문장 분리 포함)
    result = llm.ask_and_split(question)
    
    print("=== 질문 ===")
    print(question)
    print("\n=== 답변 (문장 분리) ===")
    for i, sentence in enumerate(result["sentences"], 1):
        print(f"{i}. {sentence}")
    
    print(f"\n=== 메타 정보 ===")
    print(f"출처: {', '.join(result['sources'])}")
    print(f"모델: {result['model']}")
    print(f"토큰: {result['tokens']}")
