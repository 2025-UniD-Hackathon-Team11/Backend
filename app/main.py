from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import lectures, llm
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

app = FastAPI(
    title="AI Lecture Platform API",
    description="AI 강의 플랫폼 - 3D 모션 기반 강의 서비스",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    """서버 시작시 모든 리소스 미리 로드"""
    from app.routers.llm import get_llm_service
    from pathlib import Path
    import json
    
    try:
        print("🚀 서버 초기화 중...")
        
        # 1. PDF를 sections.json으로 자동 변환 (없을 경우)
        lecture_id = 1
        documents_dir = Path(f"app/data/{lecture_id}/llm/documents")
        sections_file = documents_dir / "sections.json"
        
        if not sections_file.exists():
            print("📚 PDF 섹션 추출 시작...")
            pdf_files = list(documents_dir.glob("*.pdf"))
            
            if pdf_files:
                from utils.documents.parse_pdf_sections import extract_sections_by_font_size
                
                all_sections = []
                for pdf_file in pdf_files:
                    print(f"  - {pdf_file.name} 처리 중...")
                    sections = extract_sections_by_font_size(str(pdf_file), auto_detect=True)
                    all_sections.extend(sections)
                
                # sections.json 저장
                with open(sections_file, "w", encoding="utf-8") as f:
                    json.dump(all_sections, f, ensure_ascii=False, indent=2)
                
                print(f"✅ 총 {len(all_sections)}개 섹션 추출 완료!")
            else:
                print("⚠️  PDF 파일이 없습니다. 기존 txt 파일 사용")
        else:
            print("✅ sections.json 이미 존재")
        
        # 2. LLM 서비스 로드 (자동으로 계층적 인덱스 생성)
        print("🔧 LLM 서비스 초기화 중...")
        llm_service = get_llm_service(lecture_id=lecture_id)
        
        print("✅ 서버 준비 완료!\n")
    except Exception as e:
        print(f"⚠️  초기화 실패: {e}")
        import traceback
        traceback.print_exc()


# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(lectures.router, prefix="/api/lectures", tags=["lectures"])
app.include_router(llm.router, prefix="/api/llm", tags=["llm"])


@app.get("/")
async def root():
    return {"message": "FastAPI 서버가 실행중입니다"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
