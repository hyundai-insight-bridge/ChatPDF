import os
import uuid
import gradio as gr
from openai import OpenAI

# LangChain 관련 모듈 Import
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


# [이슈 #34] OpenAI API 키 유효성 검증 및 페이지(Step 1 -> Step 2) 전환 함수
def verify_and_switch_page(openai_key):
    if not openai_key:
        return "⚠️ API 키를 입력해 주세요.", gr.update(visible=True), gr.update(visible=False)
    
    try:
        # OpenAI API 호출 테스트
        client = OpenAI(api_key=openai_key)
        client.models.list()
        
        # 검증 성공 시: Step 1 숨김, Step 2 메인 대시보드 노출
        return "", gr.update(visible=False), gr.update(visible=True)
    except Exception as e:
        return "❌ 유효하지 않은 API 키입니다. 키를 다시 확인해 주세요.", gr.update(visible=True), gr.update(visible=False)


# PDF 처리 중 시작 상태 안내 함수
def start_processing(pdf_path):
    if pdf_path is None:
        return "⚠️ PDF 파일을 먼저 선택해 주세요."
    return "⌛ **매뉴얼 PDF를 읽고 분석하는 중입니다... 잠시만 기다려 주세요.**"


# PDF 데이터베이스 구축 함수
def build_db(pdf_path, openai_key):
    if not openai_key:
        return None, "⚠️ 먼저 API 키를 입력해 주세요."
    if pdf_path is None:
        return None, ""

    try:
        pages = PyPDFLoader(pdf_path).load_and_split()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=25,
            length_function=len,
            is_separator_regex=False,
        )
        texts = text_splitter.split_documents(pages)

        embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=openai_key,
        )

        db = Chroma.from_documents(
            texts,
            embeddings_model,
            collection_name=f"pdf_{uuid.uuid4().hex}"
        )
        return db, f"✅ **매뉴얼 분석 완료!** (총 {len(pages)}페이지 / {len(texts)}개 청크 적용됨)"
    except Exception as e:
        return None, f"❌ 매뉴얼 분석 중 오류 발생: {str(e)}"


# [이슈 #14] RAG 대화 스트리밍 및 출처(파일명 + 페이지) 표기 함수
def answer(question, openai_key, db, history):
    if not openai_key:
        raise gr.Error("먼저 API 키를 입력하고 인증해 주세요.")
    if db is None:
        raise gr.Error("사이드바에서 기기 사용설명서(PDF)를 업로드 후 분석이 완료될 때까지 기다려 주세요.")
    if not question:
        raise gr.Error("질문을 입력하세요.")

    llm = ChatOpenAI(temperature=0, openai_api_key=openai_key)
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(),
        llm=llm
    )

    # [이슈 #14, #38] 시스템 프롬프트: '매뉴얼 헬퍼' 가이드라인 및 출처 강제 명시
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "당신은 스마트 가전 및 디지털 기기 설명서를 바탕으로 답하는 안내 도우미 '매뉴얼 헬퍼'입니다.\n\n"
         "아래 [자료]에 있는 내용만 근거로 한국어로 정확하게 답하세요.\n"
         "자료에 없으면 '설명서에서 확인되지 않습니다.'라고만 답하세요.\n"
         "답변의 가장 마지막 줄에는 참고한 자료의 [출처: 파일명 (p.페이지)] 정보를 반드시 그대로 명시하세요.\n\n"
         "[자료]\n{context}"),
        ("human", "{question}"),
    ])

    generate_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_key)

    # [이슈 #14] PDF 파일명 및 페이지 번호(1-based) 포맷팅 헬퍼
    def format_docs(docs):
        formatted_list = []
        for doc in docs:
            source_path = doc.metadata.get('source', '문서.pdf')
            file_name = os.path.basename(source_path)

            page = doc.metadata.get('page', '?')
            page_num = page + 1 if isinstance(page, int) else page

            formatted_list.append(f"[출처: {file_name} (p.{page_num})]\n{doc.page_content}")

        return "\n\n".join(formatted_list)

    rag_chain = (
        {"context": retriever_from_llm | format_docs, "question": RunnablePassthrough()}
        | prompt
        | generate_llm
        | StrOutputParser()
    )

    # [이슈 #14] 히스토리 대화 기록 업데이트 및 로딩 표시
    history = history or []
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": "🤖 매뉴얼에서 정답과 출처 페이지를 찾는 중입니다..."})

    yield "", history

    text = ""
    # [이슈 #14] 답변 스트리밍 업데이트
    for chunk in rag_chain.stream(question):
        text += chunk
        history[-1]["content"] = text
        yield "", history


# -------------------------------------------------------------
# [이슈 #38] Gradio UI 구성 (AI 컨시어지 레퍼런스 대시보드 스타일)
# -------------------------------------------------------------
with gr.Blocks(
    title="매뉴얼 헬퍼 - Manual Helper",
    theme=gr.themes.Soft(primary_hue="blue")
) as demo:

    # -------------------------------------------------------------
    # [이슈 #34, #38] Step 1. 최초 진입 시 API 키 입력 및 인증 화면
    # -------------------------------------------------------------
    with gr.Column(visible=True) as step1_page:
        with gr.Row():
            with gr.Column(scale=1, min_width=150):
                gr.Image(
                    "매뉴얼 헬퍼 AI 로고.png", 
                    show_label=False, 
                    container=False, 
                    height=140
                )
            with gr.Column(scale=4):
                gr.Markdown(
                    """
                    # 매뉴얼 헬퍼 (Manual Helper)
                    ### 💻 🎮 🖥️ ⌨️ 스마트 가전 · 디지털 매뉴얼 AI 도우미
                    > **서비스 이용을 위해 OpenAI API Key를 먼저 인증해 주세요.**
                    """
                )
        gr.Markdown("---")
        
        with gr.Group():
            gr.Markdown("### 🔑 Step 1. OpenAI API 키 인증")
            gr.Markdown("OpenAI API Key (`sk-...`)를 입력하고 아래 버튼을 클릭하세요.")
            
            openai_key = gr.Textbox(
                label="OpenAI API Key", 
                placeholder="sk-... 형식의 키를 입력하세요", 
                type="password"
            )
            verify_btn = gr.Button("인증 및 매뉴얼 헬퍼 시작하기 🚀", variant="primary")
            key_status = gr.Markdown()

            gr.Markdown("---")
            with gr.Row():
                gr.Markdown("❓ **API 키 발급 방법을 모른다면?**")
                get_key_btn = gr.Button(
                    "🔗 OpenAI API 키 발급받기", 
                    link="https://platform.openai.com/api-keys",
                    size="sm"
                )

    # -------------------------------------------------------------
    # [이슈 #34, #38] Step 2. 레퍼런스 스타일 대시보드 메인 화면 (기본값: 숨김)
    # -------------------------------------------------------------
    with gr.Column(visible=False) as step2_page:
        with gr.Row():
            # [좌측 사이드바 영역]
            with gr.Column(scale=1, min_width=260):
                # 브랜딩 영역
                with gr.Row():
                    gr.Image(
                        "매뉴얼 헬퍼 AI 로고.png", 
                        show_label=False, 
                        container=False, 
                        height=50
                    )
                gr.Markdown("### **매뉴얼 헬퍼**\n`MANUAL HELPER AI`")
                
                # 새 대화 시작 버튼
                new_chat_btn = gr.Button("+ 새 대화 시작", variant="primary", size="lg")
                
                gr.Markdown("---")
                
                # 파일 업로드 관리 및 처리 상태 표기
                gr.Markdown("#### 📁 매뉴얼 파일 업로드")
                pdf_file = gr.File(
                    label="기기 사용설명서(PDF)", 
                    file_types=[".pdf"], 
                    type="filepath"
                )
                status = gr.Markdown("📄 *사용설명서 PDF를 업로드해 주세요.*")
                db_state = gr.State()
                
                gr.Markdown("---\n##### 메인 메뉴")
                gr.Markdown("- 💬 현재 대화\n- ⚙️ Settings\n- ❓ Help")
                
                gr.Markdown("---")
                # 계정 정보 및 로그아웃/초기화 버튼
                gr.Markdown("👤 **사용자 인증됨**\n`Premium Account`")
                reset_key_btn = gr.Button("🚪 API 키 다시 설정 (로그아웃)", size="sm")

            # [우측 메인 영역]
            with gr.Column(scale=4):
                # 프로필 / 상태 헤더
                with gr.Row():
                    gr.Markdown("### 🤖 **매뉴얼 AI 어시스턴트** `🟢 온라인` ")

                with gr.Tabs():
                    # 1. 💬 현재 대화 탭
                    with gr.TabItem("💬 현재 대화"):
                        chatbot = gr.Chatbot(
                            label="Q&A 대화 내역", 
                            height=520
                        )

                        with gr.Row():
                            question = gr.Textbox(
                                label="메시지를 입력하세요...", 
                                placeholder="예: 이엑스 듀오 패드로 PC 엑스인풋 무선 연결하는 법 알려줘",
                                scale=4
                            )
                            ask_btn = gr.Button("전송 📤", variant="primary", scale=1)
                        
                        gr.Markdown("<center><small>AI 어시스턴트의 답변은 참고용이며, 중요한 사항은 제공된 [출처: 페이지]를 반드시 확인하세요.</small></center>")

                    # 2. ⚙️ Settings 탭
                    with gr.TabItem("⚙️ Settings"):
                        gr.Markdown("## ⚙️ 서비스 환경설정")
                        with gr.Group():
                            gr.Markdown("### 🔑 등록된 OpenAI API Key")
                            gr.Markdown("API 키를 변경하거나 다시 인증하려면 아래 버튼을 클릭하여 초기 화면으로 이동하세요.")
                            settings_reset_key_btn = gr.Button(
                                "🔑 API 키 다시 설정 (초기 화면으로 이동)", 
                                variant="primary",
                                size="sm"
                            )

                    # 3. ❓ Help 탭
                    with gr.TabItem("❓ Help"):
                        gr.Markdown("## ❓ 매뉴얼 헬퍼 사용 안내")
                        gr.Markdown(
                            """
                            ### 📌 사용 방법
                            1. 왼쪽 사이드바의 **`기기 사용설명서(PDF)`** 영역에 갖고 계신 PDF 매뉴얼을 업로드합니다.
                            2. 사이드바 하단 메시지가 **`✅ 매뉴얼 분석 완료!`**로 바뀌면 질문을 입력합니다.
                            3. **`💬 현재 대화`** 탭에서 연결 방법, 단축키, 부팅 설정 등 궁금한 내용을 질문합니다.
                            
                            ---
                            
                            ### 💡 주요 질의 예시
                            - **노트북**: "Windows 11 설치 시 F10 부팅 메뉴 진입 방법 알려줘"
                            - **게임패드**: "EX DUO 패드로 PC 무선 연결하는 순서 알려줘"
                            - **라즈베리파이 케이스**: "Argon One v2 전원 버튼 설정하는 법 알려줘"
                            
                            ---
                            
                            ### 🔍 출처 표기 안내
                            - 답변의 가장 마지막 줄에는 참고한 **`[출처: 파일명 (p.페이지)]`** 정보가 명시되어 매뉴얼 원본을 쉽게 찾아보실 수 있습니다.
                            """
                        )

    # -------------------------------------------------------------
    # 이벤트 핸들러 바인딩
    # -------------------------------------------------------------
    # 1. API 키 검증 및 Step 2 메인 대시보드 화면 전환
    verify_btn.click(
        verify_and_switch_page,
        inputs=[openai_key],
        outputs=[key_status, step1_page, step2_page]
    )

    # 2. 초기 화면 복귀 이벤트
    reset_key_btn.click(
        lambda: ("", gr.update(visible=True), gr.update(visible=False)),
        outputs=[key_status, step1_page, step2_page]
    )
    settings_reset_key_btn.click(
        lambda: ("", gr.update(visible=True), gr.update(visible=False)),
        outputs=[key_status, step1_page, step2_page]
    )

    # 3. 새 대화 시작 버튼 (대화창 초기화)
    new_chat_btn.click(
        lambda: None,
        outputs=[chatbot]
    )

    # 4. PDF 업로드 및 분석 상태 표시 (로딩 메시지 -> 완료 메시지 체이닝)
    pdf_file.upload(
        start_processing,
        inputs=[pdf_file],
        outputs=[status]
    ).then(
        build_db, 
        inputs=[pdf_file, openai_key], 
        outputs=[db_state, status]
    )

    # 5. 질문 제출 및 답변 스트리밍
    ask_btn.click(
        answer,
        inputs=[question, openai_key, db_state, chatbot],
        outputs=[question, chatbot]
    )
    question.submit(
        answer,
        inputs=[question, openai_key, db_state, chatbot],
        outputs=[question, chatbot]
    )

if __name__ == "__main__":
    demo.launch()