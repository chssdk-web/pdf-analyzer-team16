import streamlit as st
import PyPDF2
import os
import tempfile
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import json
import re
from typing import List, Dict, Tuple
import numpy as np
import pickle

# 페이지 설정
st.set_page_config(
    page_title="PDF 문서 분석 AI",
    page_icon="📄",
    layout="wide"
)

# 세션 상태 초기화
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
if 'chunks_data' not in st.session_state:
    st.session_state.chunks_data = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = []

class PDFAnalyzer:
    def __init__(self):
        self.embedding_model = None
        self.chunks_data = []
        self.embeddings = []
        
    def initialize_models(self):
        """모델 초기화"""
        if self.embedding_model is None:
            with st.spinner("임베딩 모델을 로딩중입니다..."):
                self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        return True
    
    def extract_text_from_pdf(self, pdf_file):
        """PDF에서 텍스트 추출 (페이지별로 분리)"""
        pages_text = []
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_file.read())
                tmp_file_path = tmp_file.name
            
            with open(tmp_file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        pages_text.append({
                            'page_num': page_num + 1,
                            'text': page_text.strip()
                        })
            
            os.unlink(tmp_file_path)
            
        except Exception as e:
            st.error(f"PDF 텍스트 추출 중 오류 발생: {str(e)}")
            return []
        
        return pages_text
    
    def smart_text_chunking(self, pages_text: List[Dict], chunk_size: int = 1200, overlap: int = 200):
        """스마트 텍스트 청킹"""
        chunks = []
        
        for page_data in pages_text:
            page_num = page_data['page_num']
            text = page_data['text']
            
            paragraphs = re.split(r'\n\s*\n', text)
            
            current_chunk = ""
            current_size = 0
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                sentences = re.split(r'(?<=[.!?])\s+', para)
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    sentence_size = len(sentence)
                    
                    if current_size + sentence_size > chunk_size and current_chunk:
                        chunks.append({
                            'text': current_chunk.strip(),
                            'page_num': page_num,
                            'chunk_id': len(chunks)
                        })
                        
                        overlap_sentences = current_chunk.split('.')[-3:]
                        current_chunk = '. '.join(overlap_sentences).strip() + '. ' + sentence
                        current_size = len(current_chunk)
                    else:
                        current_chunk += ' ' + sentence
                        current_size += sentence_size + 1
            
            if current_chunk.strip():
                chunks.append({
                    'text': current_chunk.strip(),
                    'page_num': page_num,
                    'chunk_id': len(chunks)
                })
        
        return chunks
    
    def create_embeddings(self, chunks: List[Dict], filename: str):
        """임베딩 생성 및 저장"""
        try:
            self.chunks_data = chunks
            self.embeddings = []
            
            with st.spinner(f"문서를 벡터화하고 있습니다... ({len(chunks)}개 청크)"):
                progress_bar = st.progress(0)
                
                for i, chunk_data in enumerate(chunks):
                    chunk_text = chunk_data['text']
                    embedding = self.embedding_model.encode([chunk_text])[0]
                    self.embeddings.append(embedding)
                    
                    progress_bar.progress((i + 1) / len(chunks))
                
                progress_bar.empty()
            
            # 세션 상태에 저장
            st.session_state.chunks_data = self.chunks_data
            st.session_state.embeddings = self.embeddings
            
            return True
            
        except Exception as e:
            st.error(f"임베딩 생성 중 오류: {str(e)}")
            return False
    
    def cosine_similarity(self, a, b):
        """코사인 유사도 계산"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def enhanced_search(self, query: str, n_results: int = 8) -> List[Dict]:
        """향상된 검색"""
        if not self.chunks_data or not self.embeddings:
            return []
        
        try:
            query_embedding = self.embedding_model.encode([query])[0]
            
            similarities = []
            for i, embedding in enumerate(self.embeddings):
                similarity = self.cosine_similarity(query_embedding, embedding)
                similarities.append({
                    'index': i,
                    'similarity': similarity,
                    'chunk_data': self.chunks_data[i]
                })
            
            # 유사도 순으로 정렬
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            results = []
            for i, item in enumerate(similarities[:n_results]):
                chunk_data = item['chunk_data']
                results.append({
                    'text': chunk_data['text'],
                    'page_num': chunk_data['page_num'],
                    'chunk_id': chunk_data['chunk_id'],
                    'similarity_score': item['similarity'],
                    'rank': i + 1
                })
            
            return results
            
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
            
            context_parts = []
            source_info = []
            
            for page_num in sorted(page_groups.keys()):
                page_results = page_groups[page_num]
                page_results.sort(key=lambda x: x['similarity_score'], reverse=True)
                
                for i, result in enumerate(page_results[:2]):
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
    st.title("📄 PDF 문서 분석 AI (안정화 버전)")
    st.markdown("PDF 문서를 업로드하고 자연어로 질문하세요! 정확한 페이지 출처와 함께 답변을 제공합니다.")
    
    # 사이드바
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
    
    # PDF 분석기 초기화
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = PDFAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # 모델 초기화
    if not st.session_state.get('models_initialized', False):
        if analyzer.initialize_models():
            st.session_state.models_initialized = True
            st.session_state.embedding_model = analyzer.embedding_model
    
    # 파일 업로드
    uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=['pdf'])
    
    if uploaded_file is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info(f"📁 업로드된 파일: {uploaded_file.name}")
        
        with col2:
            process_button = st.button("🚀 문서 처리 시작", type="primary")
        
        if process_button and api_key:
            with st.spinner("PDF 문서를 처리하고 있습니다..."):
                pages_text = analyzer.extract_text_from_pdf(uploaded_file)
                
                if pages_text:
                    total_chars = sum(len(page['text']) for page in pages_text)
                    st.success(f"✅ 텍스트 추출 완료 ({len(pages_text)}페이지, 총 {total_chars:,} 글자)")
                    
                    chunks = analyzer.smart_text_chunking(pages_text)
                    st.success(f"✅ 텍스트 분할 완료 ({len(chunks)}개 청크)")
                    
                    if analyzer.create_embeddings(chunks, uploaded_file.name):
                        st.success("🎉 문서 처리 완료! 이제 질문할 수 있습니다.")
                        st.session_state.documents_loaded = True
                    else:
                        st.error("❌ 임베딩 생성 실패")
                else:
                    st.error("❌ 텍스트 추출 실패")
    
    # 질문 섹션
    if st.session_state.documents_loaded and api_key:
        st.markdown("---")
        st.header("💬 문서에 질문하기")
        
        question = st.text_input("질문을 입력하세요:", 
                                placeholder="예: 올해 매출 목표는 얼마인가요?")
        
        if st.button("🔍 검색", type="primary") and question:
            with st.spinner("검색 중..."):
                # 세션에서 데이터 복원
                analyzer.chunks_data = st.session_state.chunks_data
                analyzer.embeddings = st.session_state.embeddings
                
                search_results = analyzer.enhanced_search(question, n_results=8)
                
                if search_results:
                    answer, source_info = analyzer.generate_contextual_answer(question, search_results, api_key)
                    
                    st.markdown("### 📋 답변")
                    st.markdown(answer)
                    
                    if source_info:
                        st.markdown("### 📍 출처 정보")
                        for info in source_info:
                            st.write(f"📄 페이지 {info['page_num']} | 관련성: {info['similarity_score']:.2%}")
                    
                    with st.expander("📚 상세 참조 내용 보기"):
                        for i, result in enumerate(search_results[:5]):
                            st.markdown(f"**📄 페이지 {result['page_num']} | 관련성: {result['similarity_score']:.2%}**")
                            st.text_area(f"내용 {i+1}", result['text'], height=120, key=f"chunk_{i}")
                            st.markdown("---")
                else:
                    st.warning("⚠️ 관련 정보를 찾을 수 없습니다.")
    
    elif not api_key:
        st.warning("⚠️ Google Gemini API 키를 입력해주세요.")
    elif not st.session_state.documents_loaded:
        st.info("📄 PDF 파일을 업로드하고 처리해주세요.")

if __name__ == "__main__":
    main()
