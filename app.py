import os
import uuid
import tempfile
from datetime import datetime

import gradio as gr

# LangChain 관련 import os
import uuid
import tempfile
from datetime import datetime

import gradio as gr

# LangChain 관련 모듈 Import
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def build_db(pdf_path, openai_key):
    if not openai_key:
        return None, "⚠️ OpenAI API 키를 먼저 입력한 뒤 PDF를 다시 올려주세요."
    if pdf_path is None:
        return None, ""

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
    return db, f"✅ 처리 완료:{len(pages)}페이지 →{len(texts)}개 청크"


def answer(question, openai_key, db, history):
    """
    [이슈 #14 작업]
    Gradio Chatbot 연동 및 대화 기록(history) 업데이트, 출처 표기 RAG 연동 함수
    """
    if db is None:
        raise gr.Error("먼저 API 키를 입력하고 PDF를 업로드하세요.")
    if not question:
        raise gr.Error("질문을 입력하세요.")

    llm = ChatOpenAI(temperature=0, openai_api_key=openai_key)
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(),
        llm=llm
    )

    # [이슈 #14 작업] LLM이 답변 마지막 줄에 지정된 출처 양식을 반드시 포함하도록 프롬프트 작성
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "당신은 설명서를 바탕으로 답하는 안내 도우미입니다.\n\n"
         "아래 [자료]에 있는 내용만 근거로 한국어로 정확하게 답하세요.\n"
         "자료에 없으면 '설명서에서 확인되지 않습니다.'라고만 답하세요.\n"
         "답변의 가장 마지막 줄에는 참고한 자료의 [출처: 파일명 (p.페이지)] 정보를 반드시 그대로 명시하세요.\n\n"
         "[자료]\n{context}"),
        ("human", "{question}"),
    ])

    generate_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_key)

    # [이슈 #14 작업] RAG 검색 컨텍스트에 PDF 파일명 및 페이지 번호(1-indexed)를 포맷팅하여 결합
    def format_docs(docs):
        formatted_list = []
        for doc in docs:
            # metadata에서 파일명 추출 (경로 제외)
            source_path = doc.metadata.get('source', '문서.pdf')
            file_name = os.path.basename(source_path)

            # 페이지 번호 보정 (PyPDFLoader의 0-based index -> 1-based index)
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

    # [이슈 #14 작업] 대화 기록 유지 및 사용자/어시스턴트 메시지 추가
    history = history or []
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": "🔎 설명서에서 찾는 중..."})

    # [이슈 #14 작업] 1차 Yield: 질문 전송 즉시 Textbox 입력창을 초기화("")하고 대기 상태를 UI 대화창에 반영
    yield "", history

    text = ""
    # [이슈 #14 작업] 2차 Yield: RAG 답변을 스트리밍하며 어시스턴트의 답변 문구를 실시간 업데이트
    for chunk in rag_chain.stream(question):
        text += chunk
        history[-1]["content"] = text
        yield "", history


# 메시지 content를 문자열로 변환 (Gradio 버전에 따라 str 또는 [{"type": "text", "text": ...}] 형태로 들어옴)
def content_to_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text") or content.get("content") or ""
    if isinstance(content, (list, tuple)):
        return "\n".join(content_to_text(c) for c in content)
    return str(content)


# 대화 기록(chatbot)을 텍스트 파일로 저장하고 다운로드 버튼에 연결
def export_history(history, pdf_path):
    if not history:
        return gr.DownloadButton(value=None, interactive=False)

    now = datetime.now()
    pdf_name = os.path.basename(pdf_path) if pdf_path else "-"

    lines = [
        "ChatPDF 대화 기록",
        f"PDF: {pdf_name}",
        f"저장 시각: {now:%Y-%m-%d %H:%M:%S}",
        "=" * 50,
        "",
    ]
    turn = 0
    for message in history:
        role = message.get("role")
        text = content_to_text(message.get("content", "")).strip()
        if role == "user":
            turn += 1
            lines.append(f"[질문 {turn}]")
        else:
            lines.append(f"[답변 {turn}]")
        lines.append(text)
        lines.append("")
        if role != "user":
            lines.append("-" * 50)
            lines.append("")

    path = os.path.join(
        tempfile.gettempdir(),
        f"chat_history_{now:%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}.txt",
    )
    # utf-8-sig: 윈도우 메모장에서 열어도 한글이 깨지지 않음
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines))

    return gr.DownloadButton(value=path, interactive=True)


# [이슈 #14 작업] Gradio UI 및 이벤트 핸들러 구성
with gr.Blocks(title="ChatPDF") as demo:
    gr.Markdown("# ChatPDF\n---")

    openai_key = gr.Textbox(label="OPEN_AI_API_KEY", type="password")
    pdf_file = gr.File(label="PDF 파일을 올려주세요!", file_types=[".pdf"], type="filepath")
    status = gr.Markdown()
    db_state = gr.State()

    gr.Markdown("---\n## PDF에게 질문해보세요!!")

    # [이슈 #14 작업] 대화 내역 유지를 위한 gr.Chatbot 컴포넌트 추가
    chatbot = gr.Chatbot(label="대화 내용")

    question = gr.Textbox(label="질문을 입력하세요")
    ask_btn = gr.Button("질문하기", variant="primary")

    # 대화 기록 다운로드 버튼 (첫 답변이 끝나기 전에는 비활성화)
    download_btn = gr.DownloadButton("💾 대화 내용 다운로드 (.txt)", interactive=False)

    gr.Markdown("[☕ Buy me a coffee](https://www.buymeacoffee.com/{계정 ID})")

    pdf_file.upload(build_db, inputs=[pdf_file, openai_key], outputs=[db_state, status])

    # [이슈 #14 작업] 버튼 클릭 시 입력값 초기화 및 대화 기록(chatbot) 상태 갱신
    # 답변이 끝나면 대화 기록을 txt 파일로 만들어 다운로드 버튼에 연결
    ask_btn.click(
        answer,
        inputs=[question, openai_key, db_state, chatbot],
        outputs=[question, chatbot]
    ).then(export_history, inputs=[chatbot, pdf_file], outputs=download_btn)

    # [이슈 #14 작업] 엔터키 제출 시 입력값 초기화 및 대화 기록(chatbot) 상태 갱신
    ask_btn  # (위와 동일한 흐름을 엔터키에도 연결)
    question.submit(
        answer,
        inputs=[question, openai_key, db_state, chatbot],
        outputs=[question, chatbot]
    ).then(export_history, inputs=[chatbot, pdf_file], outputs=download_btn)

if __name__ == "__main__":
    demo.launch()모듈 Import
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def build_db(pdf_path, openai_key):
    if not openai_key:
        return None, "⚠️ OpenAI API 키를 먼저 입력한 뒤 PDF를 다시 올려주세요."
    if pdf_path is None:
        return None, ""

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
    return db, f"✅ 처리 완료:{len(pages)}페이지 →{len(texts)}개 청크"


def answer(question, openai_key, db, history):
    """
    [이슈 #14 작업]
    Gradio Chatbot 연동 및 대화 기록(history) 업데이트, 출처 표기 RAG 연동 함수
    """
    if db is None:
        raise gr.Error("먼저 API 키를 입력하고 PDF를 업로드하세요.")
    if not question:
        raise gr.Error("질문을 입력하세요.")

    llm = ChatOpenAI(temperature=0, openai_api_key=openai_key)
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(),
        llm=llm
    )

    # [이슈 #14 작업] LLM이 답변 마지막 줄에 지정된 출처 양식을 반드시 포함하도록 프롬프트 작성
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "당신은 설명서를 바탕으로 답하는 안내 도우미입니다.\n\n"
         "아래 [자료]에 있는 내용만 근거로 한국어로 정확하게 답하세요.\n"
         "자료에 없으면 '설명서에서 확인되지 않습니다.'라고만 답하세요.\n"
         "답변의 가장 마지막 줄에는 참고한 자료의 [출처: 파일명 (p.페이지)] 정보를 반드시 그대로 명시하세요.\n\n"
         "[자료]\n{context}"),
        ("human", "{question}"),
    ])

    generate_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_key)

    # [이슈 #14 작업] RAG 검색 컨텍스트에 PDF 파일명 및 페이지 번호(1-indexed)를 포맷팅하여 결합
    def format_docs(docs):
        formatted_list = []
        for doc in docs:
            # metadata에서 파일명 추출 (경로 제외)
            source_path = doc.metadata.get('source', '문서.pdf')
            file_name = os.path.basename(source_path)

            # 페이지 번호 보정 (PyPDFLoader의 0-based index -> 1-based index)
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

    # [이슈 #14 작업] 대화 기록 유지 및 사용자/어시스턴트 메시지 추가
    history = history or []
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": "🔎 설명서에서 찾는 중..."})

    # [이슈 #14 작업] 1차 Yield: 질문 전송 즉시 Textbox 입력창을 초기화("")하고 대기 상태를 UI 대화창에 반영
    yield "", history

    text = ""
    # [이슈 #14 작업] 2차 Yield: RAG 답변을 스트리밍하며 어시스턴트의 답변 문구를 실시간 업데이트
    for chunk in rag_chain.stream(question):
        text += chunk
        history[-1]["content"] = text
        yield "", history


# 메시지 content를 문자열로 변환 (Gradio 버전에 따라 str 또는 [{"type": "text", "text": ...}] 형태로 들어옴)
def content_to_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text") or content.get("content") or ""
    if isinstance(content, (list, tuple)):
        return "\n".join(content_to_text(c) for c in content)
    return str(content)


# 대화 기록(chatbot)을 텍스트 파일로 저장하고 다운로드 버튼에 연결
def export_history(history, pdf_path):
    if not history:
        return gr.DownloadButton(value=None, interactive=False)

    now = datetime.now()
    pdf_name = os.path.basename(pdf_path) if pdf_path else "-"

    lines = [
        "ChatPDF 대화 기록",
        f"PDF: {pdf_name}",
        f"저장 시각: {now:%Y-%m-%d %H:%M:%S}",
        "=" * 50,
        "",
    ]
    turn = 0
    for message in history:
        role = message.get("role")
        text = content_to_text(message.get("content", "")).strip()
        if role == "user":
            turn += 1
            lines.append(f"[질문 {turn}]")
        else:
            lines.append(f"[답변 {turn}]")
        lines.append(text)
        lines.append("")
        if role != "user":
            lines.append("-" * 50)
            lines.append("")

    path = os.path.join(
        tempfile.gettempdir(),
        f"chat_history_{now:%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}.txt",
    )
    # utf-8-sig: 윈도우 메모장에서 열어도 한글이 깨지지 않음
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines))

    return gr.DownloadButton(value=path, interactive=True)


# [이슈 #14 작업] Gradio UI 및 이벤트 핸들러 구성
with gr.Blocks(title="ChatPDF") as demo:
    gr.Markdown("# ChatPDF\n---")

    openai_key = gr.Textbox(label="OPEN_AI_API_KEY", type="password")
    pdf_file = gr.File(label="PDF 파일을 올려주세요!", file_types=[".pdf"], type="filepath")
    status = gr.Markdown()
    db_state = gr.State()

    gr.Markdown("---\n## PDF에게 질문해보세요!!")

    # [이슈 #14 작업] 대화 내역 유지를 위한 gr.Chatbot 컴포넌트 추가
    chatbot = gr.Chatbot(label="대화 내용")

    question = gr.Textbox(label="질문을 입력하세요")
    ask_btn = gr.Button("질문하기", variant="primary")

    # 대화 기록 다운로드 버튼 (첫 답변이 끝나기 전에는 비활성화)
    download_btn = gr.DownloadButton("💾 대화 내용 다운로드 (.txt)", interactive=False)

    gr.Markdown("[☕ Buy me a coffee](https://www.buymeacoffee.com/{계정 ID})")

    pdf_file.upload(build_db, inputs=[pdf_file, openai_key], outputs=[db_state, status])

    # [이슈 #14 작업] 버튼 클릭 시 입력값 초기화 및 대화 기록(chatbot) 상태 갱신
    # 답변이 끝나면 대화 기록을 txt 파일로 만들어 다운로드 버튼에 연결
    ask_btn.click(
        answer,
        inputs=[question, openai_key, db_state, chatbot],
        outputs=[question, chatbot]
    ).then(export_history, inputs=[chatbot, pdf_file], outputs=download_btn)

    # [이슈 #14 작업] 엔터키 제출 시 입력값 초기화 및 대화 기록(chatbot) 상태 갱신
    question.submit(
        answer,
        inputs=[question, openai_key, db_state, chatbot],
        outputs=[question, chatbot]
    ).then(export_history, inputs=[chatbot, pdf_file], outputs=download_btn)

if __name__ == "__main__":
    demo.launch()