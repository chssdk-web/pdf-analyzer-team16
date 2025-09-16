import streamlit as st
import PyPDF2
import os
import tempfile
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import google.generativeai as genai
import json
import re
from typing import List, Dict, Tuple
import numpy as np

# 페이지 설정
st.set_page_config(
    page_title="PDF 문서 분석 AI",
    page_icon="📄",
    layout="wide"
)

# 세션 상태 초기화
if 'vectordb' not in st.session_state:
    st.session_state.vectordb = None
if 'collection' not in st.session_state:
    st.session_state.collection = None
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
if 'page_mapping' not in st.session_state:
    st.session_state.page_mapping = {}

class PDFAnalyzer:
    def __init__(self):
        self.embedding_model = None
        self.vectordb = None
        self.collection = None
        self.page_mapping = {}
        
    def initialize_models(self):
        """모델 초기화"""
        if self.embedding_model is None:
            with st.spinner("임베딩 모델을 로딩중입니다..."):
                self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        if self.vectordb is None:
            self.vectordb = chromadb.Client(Settings(
                persist_directory="./chroma_db",
                anonymized_telemetry=False
            ))
            
        return True
    
    def extract_text_from_pdf(self, pdf_file):
        """PDF에서 텍스트 추출 (페이지별로 분리)"""
        pages_text = []
        try:
            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_file.read())
                tmp_file_path = tmp_file.name
            
            # PDF 읽기
            with open(tmp_file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        pages_text.append({
                            'page_num': page_num + 1,
                            'text': page_text.strip()
                        })
            
            # 임시 파일 삭제
            os.unlink(tmp_file_path)
            
        except Exception as e:
            st.error(f"PDF 텍스트 추출 중 오류 발생: {str(e)}")
            return []
        
        return pages_text
    
    def smart_text_chunking(self, pages_text: List[Dict], chunk_size: int = 1200, overlap: int = 200):
        """스마트 텍스트 청킹 - 문맥을 고려한 분할"""
        chunks = []
        
        for page_data in pages_text:
            page_num = page_data['page_num']
            text = page_data['text']
            
            # 문단 단위로 먼저 분할
            paragraphs = re.split(r'\n\s*\n', text)
            
            current_chunk = ""
            current_size = 0
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                # 문장 단위로 세분화
                sentences = re.split(r'(?<=[.!?])\s+', para)
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    sentence_size = len(sentence)
                    
                    # 청크 크기 초과 시 새 청크 생성
                    if current_size + sentence_size > chunk_size and current_chunk:
                        chunks.append({
                            'text': current_chunk.strip(),
                            'page_num': page_num,
                            'chunk_id': len(chunks)
                        })
                        
                        # 오버랩을 위한 컨텍스트 유지
                        overlap_sentences = current_chunk.split('.')[-3:]  # 마지막 3문장 유지
                        current_chunk = '. '.join(overlap_sentences).strip() + '. ' + sentence
                        current_size = len(current_chunk)
                    else:
                        current_chunk += ' ' + sentence
                        current_size += sentence_size + 1
            
            # 마지막 청크 추가
            if current_chunk.strip():
                chunks.append({
                    'text': current_chunk.strip(),
                    'page_num': page_num,
                    'chunk_id': len(chunks)
                })
        
        return chunks
    
    def create_vector_database(self, chunks: List[Dict], filename: str):
        """향상된 벡터 데이터베이스 생성"""
        try:
            # 기존 컬렉션 삭제
            try:
                self.vectordb.delete_collection(name="documents")
            except:
                pass
            
            # 새 컬렉션 생성
            self.collection = self.vectordb.create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            
            # 페이지 매핑 초기화
            self.page_mapping = {}
            
            # 청크들을 임베딩하고 저장
            with st.spinner(f"문서를 벡터화하고 있습니다... ({len(chunks)}개 청크)"):
                progress_bar = st.progress(0)
                
                for i, chunk_data in enumerate(chunks):
                    chunk_text = chunk_data['text']
                    page_num = chunk_data['page_num']
                    chunk_id = chunk_data['chunk_id']
                    
                    # 임베딩 생성
                    embedding = self.embedding_model.encode([chunk_text])[0].tolist()
                    
                    # 고유 ID 생성
                    unique_id = f"{filename}_{chunk_id}"
                    
                    # 벡터 DB에 저장
                    self.collection.add(
                        embeddings=[embedding],
                        documents=[chunk_text],
                        metadatas=[{
                            "source": filename, 
                            "chunk_id": chunk_id,
                            "page_num": page_num,
                            "text_length": len(chunk_text)
                        }],
                        ids=[unique_id]
                    )
                    
                    # 페이지 매핑 저장
                    self.page_mapping[unique_id] = {
                        'page_num': page_num,
                        'text': chunk_text,
                        'chunk_id': chunk_id
                    }
                    
                    progress_bar.progress((i + 1) / len(chunks))
                
                progress_bar.empty()
            
            # 세션 상태에 페이지 매핑 저장
            st.session_state.page_mapping = self.page_mapping
            
            return True
            
        except Exception as e:
            st.error(f"벡터 데이터베이스 생성 중 오류: {str(e)}")
            return False
    
    def enhanced_search(self, query: str, n_results: int = 8) -> List[Dict]:
        """향상된 검색 - 페이지 정보와 관련성 점수 포함"""
        if not self.collection:
            return []
        
        try:
            # 질문을 임베딩으로 변환
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            
            # 유사한 청크 검색
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            enhanced_results = []
            if results['documents']:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0], 
                    results['metadatas'][0], 
                    results['distances'][0]
                )):
                    # 관련성 점수 계산 (거리를 유사도로 변환)
                    similarity_score = max(0, 1 - distance)
                    
                    enhanced_results.append({
                        'text': doc,
                        'page_num': metadata['page_num'],
                        'chunk_id': metadata['chunk_id'],
                        'similarity_score': similarity_score,
                        'rank': i + 1
                    })
            
            return enhanced_results
            
        except Exception as e:
            st.error(f"검색 중 오류 발생: {str(e)}")
            return []
    
    def generate_contextual_answer(self, question: str, search_results: List[Dict], api_key: str):
        """맥락을 고려한 답변 생성"""
        if not search_results:
            return "관련 정보를 찾을 수 없습니다.", []
        
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-pro-latest')
            
            # 페이지별로 결과 그룹화
            page_groups = {}
            for result in search_results:
                page_num = result['page_num']
                if page_num not in page_groups:
                    page_groups[page_num] = []
                page_groups[page_num].append(result)
            
            # 컨텍스트 구성 - 페이지 순서대로 정렬
            context_parts = []
            source_info = []
            
            for page_num in sorted(page_groups.keys()):
                page_results = page_groups[page_num]
                # 각 페이지에서 가장 관련성 높은 청크들 선택
                page_results.sort(key=lambda x: x['similarity_score'], reverse=True)
                
                for i, result in enumerate(page_results[:2]):  # 페이지당 최대 2개 청크
                    context_parts.append(f"[페이지 {page_num}, 발췌 {len(context_parts)+1}]\n{result['text']}")
                    source_info.append({
                        'page_num': page_num,
                        'text': result['text'],
                        'similarity_score': result['similarity_score'],
                        'excerpt_num': len(context_parts)
                    })
            
            context = "\n\n".join(context_parts)
            
            prompt = f"""
다음은 PDF 문서에서 발췌한 내용들입니다:

{context}

질문: {question}

지시사항:
1. 위 문서 발췌 내용에서만 답변하세요.
2. 답변은 반드시 원문 그대로 인용하세요. 절대 요약하거나 바꿔쓰지 마세요.
3. 여러 발췌 내용이 관련있다면 모두 포함하세요.
4. 문서에 해당 정보가 없으면 "관련 정보를 찾을 수 없습니다"라고 답하세요.
5. 답변할 때 반드시 출처를 명시하세요: [페이지 X, 발췌 Y]
6. 질문과 직접 관련된 핵심 정보를 빠뜨리지 마세요.
7. 원문의 숫자, 날짜, 고유명사는 정확히 그대로 인용하세요.
8. 각 발췌 내용이 어느 페이지에서 나온 것인지 명확히 표시하세요.
"""
            
            response = model.generate_content(prompt)
            return response.text, source_info
            
        except Exception as e:
            return f"답변 생성 중 오류 발생: {str(e)}", []

# 메인 애플리케이션
def main():
    st.title("📄 PDF 문서 분석 AI (고도화 버전)")
    st.markdown("PDF 문서를 업로드하고 자연어로 질문하세요! 정확한 페이지 출처와 함께 답변을 제공합니다.")
    
    # 사이드바 - API 키 설정
    with st.sidebar:
        st.header("⚙️ 설정")
        api_key = st.text_input("Google Gemini API 키", type="password", 
                               help="https://aistudio.google.com/app/apikey 에서 발급받으세요")
        
        if not api_key:
            st.warning("API 키를 입력해주세요!")
        
        st.markdown("---")
        st.markdown("### 📖 사용법")
        st.markdown("""
        1. Google Gemini API 키 입력
        2. PDF 파일 업로드
        3. 문서 처리 대기
        4. 질문 입력하여 검색
        """)
        
        st.markdown("---")
        st.markdown("### 🎯 개선사항")
        st.markdown("""
        - ✅ 정확한 페이지 출처 표시
        - ✅ 맥락 고려한 청킹
        - ✅ 관련성 점수 표시
        - ✅ 페이지별 그룹화
        """)
    
    # PDF 분석기 초기화
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = PDFAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # 모델 초기화
    if not st.session_state.get('models_initialized', False):
        if analyzer.initialize_models():
            st.session_state.models_initialized = True
            st.session_state.embedding_model = analyzer.embedding_model
            st.session_state.vectordb = analyzer.vectordb
    
    # 파일 업로드
    uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=['pdf'])
    
    if uploaded_file is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info(f"📁 업로드된 파일: {uploaded_file.name}")
        
        with col2:
            process_button = st.button("🚀 문서 처리 시작", type="primary")
        
        # 문서 처리
        if process_button and api_key:
            with st.spinner("PDF 문서를 고도화된 방식으로 처리하고 있습니다..."):
                # 페이지별 텍스트 추출
                st.write("🔍 페이지별 텍스트 추출 중...")
                pages_text = analyzer.extract_text_from_pdf(uploaded_file)
                
                if pages_text:
                    total_chars = sum(len(page['text']) for page in pages_text)
                    st.success(f"✅ 텍스트 추출 완료 ({len(pages_text)}페이지, 총 {total_chars:,} 글자)")
                    
                    # 스마트 청크 분할
                    st.write("🧠 스마트 텍스트 분할 중...")
                    chunks = analyzer.smart_text_chunking(pages_text)
                    st.success(f"✅ 텍스트 분할 완료 ({len(chunks)}개 청크)")
                    
                    # 페이지별 청크 분포 표시
                    page_distribution = {}
                    for chunk in chunks:
                        page_num = chunk['page_num']
                        page_distribution[page_num] = page_distribution.get(page_num, 0) + 1
                    
                    st.info(f"📊 페이지별 청크 분포: {dict(sorted(page_distribution.items()))}")
                    
                    # 벡터 데이터베이스 생성
                    st.write("🗄️ 향상된 벡터 데이터베이스 생성 중...")
                    if analyzer.create_vector_database(chunks, uploaded_file.name):
                        st.success("🎉 문서 처리 완료! 이제 정확한 출처와 함께 질문할 수 있습니다.")
                        st.session_state.documents_loaded = True
                        st.session_state.collection = analyzer.collection
                    else:
                        st.error("❌ 벡터 데이터베이스 생성 실패")
                else:
                    st.error("❌ 텍스트 추출 실패")
    
    # 질문 섹션
    if st.session_state.documents_loaded and api_key:
        st.markdown("---")
        st.header("💬 문서에 질문하기")
        
        # 예시 질문들
        st.markdown("**💡 질문 예시:**")
        example_questions = [
            "올해 태양광 투자 계획은 어떻게 돼?",
            "주요 재무 지표는 무엇인가?",
            "리스크 요인은 무엇인가?",
            "향후 전망은 어떤가?",
            "매출 목표는 얼마인가?",
            "주요 사업 분야는?"
        ]
        
        cols = st.columns(3)
        for i, question in enumerate(example_questions):
            with cols[i % 3]:
                if st.button(question, key=f"example_{i}"):
                    st.session_state.current_question = question
        
        # 질문 입력
        question = st.text_input("질문을 입력하세요:", 
                                value=st.session_state.get('current_question', ''),
                                placeholder="예: 올해 매출 목표는 얼마인가요?")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            search_button = st.button("🔍 정밀 검색", type="primary")
        with col2:
            clear_button = st.button("🗑️ 초기화")
        
        if clear_button:
            st.session_state.current_question = ""
            st.rerun()
        
        if search_button and question:
            with st.spinner("관련 내용을 정밀 검색하고 맥락을 고려한 답변을 생성하고 있습니다..."):
                # 향상된 검색
                search_results = analyzer.enhanced_search(question, n_results=8)
                
                if search_results:
                    # 맥락적 답변 생성
                    answer, source_info = analyzer.generate_contextual_answer(question, search_results, api_key)
                    
                    # 결과 표시
                    st.markdown("### 📋 답변")
                    st.markdown(answer)
                    
                    # 출처 정보 표시
                    if source_info:
                        st.markdown("### 📍 출처 정보")
                        source_df_data = []
                        for info in source_info:
                            source_df_data.append({
                                "발췌 번호": info['excerpt_num'],
                                "페이지": info['page_num'],
                                "관련성": f"{info['similarity_score']:.2%}",
                                "텍스트 길이": f"{len(info['text'])}자"
                            })
                        
                        if source_df_data:
                            st.dataframe(source_df_data, use_container_width=True)
                    
                    # 상세 참조 내용
                    with st.expander("📚 상세 참조 내용 보기"):
                        for i, result in enumerate(search_results[:6]):
                            st.markdown(f"**📄 페이지 {result['page_num']} | 관련성: {result['similarity_score']:.2%} | 순위: {result['rank']}**")
                            st.text_area(
                                f"내용 {i+1}", 
                                result['text'], 
                                height=120, 
                                key=f"detail_chunk_{i}"
                            )
                            st.markdown("---")
                else:
                    st.warning("⚠️ 관련 정보를 찾을 수 없습니다.")
    
    elif not api_key:
        st.warning("⚠️ Google Gemini API 키를 입력해주세요.")
    elif not st.session_state.documents_loaded:
        st.info("📄 PDF 파일을 업로드하고 처리해주세요.")
    
    # 하단 정보
    st.markdown("---")
    st.markdown("**🔧 고도화 기능:** 정확한 페이지 출처 표시, 맥락 고려 청킹, 관련성 점수, 페이지별 그룹화")
    st.markdown("**💡 주의사항:** 현재 텍스트만 분석합니다. 이미지, 표, 그래프 분석 기능은 추후 추가 예정입니다.")

if __name__ == "__main__":
    main()
