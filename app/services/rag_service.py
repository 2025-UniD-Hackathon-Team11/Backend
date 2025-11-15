"""
RAG (Retrieval-Augmented Generation) 서비스
강의별 페르소나와 문서를 활용한 질의응답 시스템
RTX 2060 최적화 - 경량 임베딩 모델 + FAISS 벡터 검색
"""

import json
import os
from typing import List, Dict, Optional, Tuple
import numpy as np
from pathlib import Path
import pickle
import torch


class RAGService:
    """강의별 RAG 서비스 - RTX 2060 최적화"""
    
    def __init__(self, lecture_id: int, base_dir: str = None):
        """
        Args:
            lecture_id: 강의 ID
            base_dir: 데이터 베이스 디렉토리 (기본값: app/data)
        """
        self.lecture_id = lecture_id
        
        if base_dir is None:
            current_dir = Path(__file__).parent.parent
            base_dir = current_dir / "data"
        
        self.base_dir = Path(base_dir)
        self.llm_dir = self.base_dir / str(lecture_id) / "llm"
        self.documents_dir = self.llm_dir / "documents"
        self.embeddings_dir = self.llm_dir / "embeddings"
        self.persona_file = self.llm_dir / "persona.json"
        
        # 임베딩 모델 (lazy loading)
        self._embedding_model = None
        self._faiss_index = None
        self._section_index = None  # 섹션 제목 인덱스
        self._chunks_metadata = []
        self._sections = []  # 섹션 데이터
        
        # 페르소나 로드
        self.persona = self._load_persona()
        
        # 임베딩 모델 즉시 로드 (인덱스 빌드 전에 먼저!)
        self._load_embedding_model()
        
        # 섹션 기반 문서 로드
        self._load_sections()
        
        # 문서 로드 및 인덱스 준비
        self.documents = []
        self._load_documents()
        self._load_or_build_index()
    
    def _load_embedding_model(self):
        """임베딩 모델 즉시 로드"""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(
                'paraphrase-multilingual-MiniLM-L12-v2',
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
    
    @property
    def embedding_model(self):
        """임베딩 모델 반환"""
        return self._embedding_model
    
    def _load_persona(self) -> Dict:
        """페르소나 설정 로드"""
        if not self.persona_file.exists():
            raise FileNotFoundError(f"페르소나 파일을 찾을 수 없습니다: {self.persona_file}")
        
        with open(self.persona_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _load_sections(self):
        """섹션 데이터 로드 (sections.json)"""
        sections_file = self.documents_dir / "sections.json"
        
        if sections_file.exists():
            with open(sections_file, "r", encoding="utf-8") as f:
                self._sections = json.load(f)
        else:
            self._sections = []
    
    def _load_documents(self):
        """문서 로드"""
        if self.documents_dir.exists():
            for doc_file in self.documents_dir.glob("*.txt"):
                with open(doc_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    self.documents.append({
                        "filename": doc_file.name,
                        "content": content
                    })
    
    def _split_into_chunks(self, text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        """
        텍스트를 청크로 분할 (RTX 2060 최적화: 작은 청크)
        
        Args:
            text: 원본 텍스트
            chunk_size: 청크 크기 (문자 수) - 300자로 줄임
            overlap: 청크 간 오버랩 크기
        
        Returns:
            청크 리스트
        """
        chunks = []
        
        # 문단 단위로 먼저 분할
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(para) <= chunk_size:
                chunks.append(para)
            else:
                # 긴 문단은 chunk_size로 분할
                start = 0
                while start < len(para):
                    end = start + chunk_size
                    chunk = para[start:end]
                    chunks.append(chunk.strip())
                    start += chunk_size - overlap
        
        return chunks
    
    def _load_or_build_index(self):
        """FAISS 인덱스 로드 또는 생성"""
        index_file = self.embeddings_dir / "faiss_index.bin"
        metadata_file = self.embeddings_dir / "chunks_metadata.pkl"
        
        # 기존 인덱스가 있으면 로드
        if index_file.exists() and metadata_file.exists():
            try:
                import faiss
                self._faiss_index = faiss.read_index(str(index_file))
                with open(metadata_file, "rb") as f:
                    self._chunks_metadata = pickle.load(f)
                return
            except Exception:
                pass
        
        # 인덱스 새로 생성
        self._build_index()
    
    def _build_index(self):
        """FAISS 인덱스 생성 (계층적: 섹션 제목 + 내용)"""
        if self._sections:
            self._build_hierarchical_index()
        elif self.documents:
            self._build_flat_index()
    
    def _build_hierarchical_index(self):
        """계층적 인덱스: 1) 섹션 제목 검색 → 2) 관련 섹션 내용 반환"""
        import faiss
        
        # 1단계: 섹션 제목 임베딩
        section_titles = [s["title"] for s in self._sections]
        title_embeddings = self.embedding_model.encode(
            section_titles,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # 섹션 제목 인덱스 생성
        dimension = title_embeddings.shape[1]
        self._section_index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(title_embeddings)
        self._section_index.add(title_embeddings)
        
        # 2단계: 섹션 메타데이터 저장
        self._chunks_metadata = []
        for section in self._sections:
            self._chunks_metadata.append({
                "text": section["content"],
                "title": section["title"],
                "source": f"Section: {section['title']}",
                "page": section.get("page", 0)
            })
        
        # 인덱스 저장
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._section_index, str(self.embeddings_dir / "section_index.bin"))
        with open(self.embeddings_dir / "sections_metadata.pkl", "wb") as f:
            pickle.dump(self._chunks_metadata, f)
    
    def _build_flat_index(self):
        """일반 인덱스: 문서를 청크로 분할하여 검색"""
        import faiss
        
        # 모든 문서를 청크로 분할
        all_chunks = []
        for doc in self.documents:
            chunks = self._split_into_chunks(doc["content"])
            for chunk in chunks:
                all_chunks.append({
                    "text": chunk,
                    "source": doc["filename"]
                })
        
        if not all_chunks:
            return
        
        self._chunks_metadata = all_chunks
        
        # 임베딩 생성 (배치 처리)
        texts = [chunk["text"] for chunk in all_chunks]
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,  # RTX 2060 최적화
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # FAISS 인덱스 생성 (IndexFlatIP: 내적 기반 유사도)
        dimension = embeddings.shape[1]
        self._faiss_index = faiss.IndexFlatIP(dimension)
        
        # 정규화 (코사인 유사도 사용)
        faiss.normalize_L2(embeddings)
        self._faiss_index.add(embeddings)
        
        # 인덱스 저장
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._faiss_index, str(self.embeddings_dir / "faiss_index.bin"))
        with open(self.embeddings_dir / "chunks_metadata.pkl", "wb") as f:
            pickle.dump(self._chunks_metadata, f)
    
    def rebuild_index(self):
        """인덱스 강제 재생성 (문서 추가/수정 시 호출)"""
        self._build_index()
    
    def get_system_prompt(self) -> str:
        """시스템 프롬프트 반환"""
        return self.persona.get("system_prompt", "")
    
    def get_persona_info(self) -> Dict:
        """페르소나 정보 반환"""
        return self.persona.get("persona", {})
    
    def add_document(self, filename: str, content: str) -> bool:
        """
        새 문서 추가
        
        Args:
            filename: 파일명
            content: 문서 내용
        
        Returns:
            성공 여부
        """
        try:
            # 문서 저장
            doc_path = self.documents_dir / filename
            self.documents_dir.mkdir(parents=True, exist_ok=True)
            
            with open(doc_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            # 메모리에도 추가
            self.documents.append({
                "filename": filename,
                "content": content
            })
            
            # 인덱스 재생성
            self.rebuild_index()
            
            return True
        except Exception as e:
            print(f"문서 추가 실패: {str(e)}")
            return False
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        쿼리와 관련된 문서 청크 검색 (계층적 검색)
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 결과 개수
        
        Returns:
            관련 청크 리스트 (각 청크는 {'content', 'score', 'source', 'title'} 포함)
        """
        # 계층적 검색: 섹션 제목 기반
        if self._section_index is not None:
            return self._hierarchical_search(query, top_k)
        # 일반 검색: 청크 기반 (fallback)
        elif self._faiss_index is not None:
            return self._flat_search(query, top_k)
        else:
            return []
    
    def _hierarchical_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        계층적 검색: 1) 제목으로 관련 섹션 찾기 → 2) 섹션 전체 내용 반환
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 섹션 수
        
        Returns:
            관련 섹션 리스트
        """
        import faiss
        
        # 쿼리 임베딩
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        )
        faiss.normalize_L2(query_embedding)
        
        # 섹션 제목으로 검색
        scores, indices = self._section_index.search(query_embedding, min(top_k, len(self._chunks_metadata)))
        
        results = []
        print(f"\n🔎 계층적 RAG 검색 (Heading 2 기반)")
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self._chunks_metadata):
                section = self._chunks_metadata[idx]
                results.append({
                    "content": section["text"],
                    "title": section["title"],
                    "score": float(score),
                    "source": section["source"]
                })
                print(f"  ✓ [{float(score):.3f}] {section['title']}")
        
        return results
    
    def _flat_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        일반 검색: 모든 청크에서 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 청크 수
        
        Returns:
            관련 청크 리스트
        """
        import faiss
        
        # 쿼리 임베딩
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        )
        faiss.normalize_L2(query_embedding)
        
        # FAISS 검색
        scores, indices = self._faiss_index.search(query_embedding, top_k)
        
        # 결과 구성
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self._chunks_metadata):
                chunk_meta = self._chunks_metadata[idx]
                results.append({
                    "content": chunk_meta["text"],
                    "score": float(score),
                    "source": chunk_meta["source"]
                })
        
        return results
    
    def build_rag_context(self, query: str, top_k: int = 3) -> str:
        """
        RAG를 위한 컨텍스트 구성
        
        Args:
            query: 사용자 질문
            top_k: 검색할 문서 청크 개수
        
        Returns:
            LLM에 전달할 컨텍스트 문자열
        """
        relevant_chunks = self.retrieve_relevant_chunks(query, top_k)
        
        if not relevant_chunks:
            return "관련 강의 자료를 찾을 수 없습니다."
        
        context_parts = ["다음은 관련된 강의 자료입니다:\n"]
        
        for i, chunk in enumerate(relevant_chunks, 1):
            context_parts.append(f"\n[참고자료 {i}] (출처: {chunk['source']}, 관련도: {chunk['score']:.2f})")
            context_parts.append(chunk['content'])
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def prepare_llm_request(self, user_question: str, top_k: int = 3) -> Dict:
        """
        LLM 요청을 위한 데이터 준비
        
        Args:
            user_question: 사용자 질문
            top_k: 검색할 문서 청크 개수
        
        Returns:
            LLM 요청 데이터 (system_prompt, context, question 등)
        """
        system_prompt = self.get_system_prompt()
        context = self.build_rag_context(user_question, top_k)
        
        return {
            "system_prompt": system_prompt,
            "context": context,
            "question": user_question,
            "persona": self.get_persona_info(),
            "temperature": self.persona.get("temperature", 0.7),
            "max_tokens": self.persona.get("max_tokens", 1000)
        }


def create_rag_service(lecture_id: int) -> RAGService:
    """
    RAG 서비스 인스턴스 생성
    
    Args:
        lecture_id: 강의 ID
    
    Returns:
        RAGService 인스턴스
    """
    return RAGService(lecture_id)


# 사용 예시
if __name__ == "__main__":
    # 강의 1번의 RAG 서비스 생성
    print("🚀 RAG 서비스 초기화 중...")
    rag = create_rag_service(lecture_id=1)
    
    # 페르소나 정보 확인
    print("\n=== 페르소나 정보 ===")
    print(json.dumps(rag.get_persona_info(), indent=2, ensure_ascii=False))
    
    # 질문 예시
    questions = [
        "컨볼루션이 뭔가요?",
        "슬라이딩 윈도우는 어떻게 작동하나요?",
        "CNN에서 padding은 왜 사용하나요?"
    ]
    
    for question in questions:
        print(f"\n{'='*60}")
        print(f"질문: {question}")
        print("="*60)
        
        # 관련 문서 검색
        chunks = rag.retrieve_relevant_chunks(question, top_k=2)
        
        print("\n검색 결과:")
        for i, chunk in enumerate(chunks, 1):
            print(f"\n[{i}] 출처: {chunk['source']} (점수: {chunk['score']:.3f})")
            print(f"내용: {chunk['content'][:150]}...")
        
        # LLM 요청 데이터 준비
        llm_request = rag.prepare_llm_request(question, top_k=2)
        print(f"\n✅ LLM 요청 데이터 준비 완료")
