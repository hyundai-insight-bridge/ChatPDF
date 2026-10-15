import os
import uuid
import gradio as gr

# LangChain 관련 모듈 Import
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langsmith import Client


def build_db(pdf_path, openai_key):
    if not openai_key:
        return None, "⚠️ OpenAI API 키를 먼저 입력한 뒤 PDF를 다시 올려주세요."
    if pdf_path is None:
        return None, ""

    pages = list()

    for path in pdf_path:
        pages.extend(PyPDFLoader(path).load_and_split())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=25,
        length_function=len, is_separator_regex=False,
    )
    texts = text_splitter.split_documents(pages)

    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-large", openai_api_key=openai_key,
    )

    db = Chroma.from_documents(
        texts, embeddings_model, collection_name=f"pdf_{uuid.uuid4().hex}"
    )
    return db, f"✅ 처리 완료:{len(pages)}페이지 →{len(texts)}개 청크"

def answer(question, openai_key, db):
    if db is None:
        raise gr.Error("먼저 API 키를 입력하고 PDF를 업로드하세요.")
    if not question:
        raise gr.Error("질문을 입력하세요.")

    llm = ChatOpenAI(temperature=0, openai_api_key=openai_key)
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(), llm=llm
    )
#     prompt = ChatPromptTemplate.from_messages([
#     ("system", "당신은 설명서를 바탕으로 답하는 안내 도우미입니다.\n\n" 
#      "아래 [자료]에 있는 내용만 근거로 한국어로 답하세요\n"
#      "자료에 없으면 '설명서에서 확인되지 않습니다.'라고만 답하세요.\n"
#      "자료에 '불가능하다', '받지 않는다'처럼 없음이나 금지로 적힌 내용도 확인된 내용이므로 그대로 답하세요.\n"
#      "답변은 자료에서 확인될 때만 마지막 줄에 출처(페이지)를 적으세요.\n\n"
#      "[자료]\n{context}"),
#     ("human", "{question}"),])

    client = Client()
    prompt = client.pull_prompt("rlm/rag-prompt", dangerously_pull_public_prompt=True)
    generate_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_key)

    def format_docs(docs):
        return "\n\n".join(
            f"[출처: {doc.metadata.get('page', '?')}쪽]\n{doc.page_content}" for doc in docs
        )

    rag_chain = (
        {"context": retriever_from_llm | format_docs, "question": RunnablePassthrough()}
        | prompt
        | generate_llm
        | StrOutputParser()
    )
    yield "🔎 설명서에서 찾는 중..."          # 상태 메시지를 먼저 화면에 내보냄
    
    text = ""
    for chunk in rag_chain.stream(question):
        text += chunk
        yield text

with gr.Blocks(title="ChatPDF") as demo:
    gr.Markdown("# ChatPDF\n---")

    openai_key = gr.Textbox(label="OPEN_AI_API_KEY", type="password")
    pdf_files = gr.File(label="PDF 파일을 올려주세요!", file_count="multiple", file_types=[".pdf"], type="filepath")
    status = gr.Markdown()
    db_state = gr.State()

    gr.Markdown("---\n## PDF에게 질문해보세요!!")
    question = gr.Textbox(label="질문을 입력하세요")
    ask_btn = gr.Button("질문하기", variant="primary")
    output = gr.Markdown()

    gr.Markdown("[☕ Buy me a coffee](https://www.buymeacoffee.com/{계정 ID})")

    pdf_files.upload(build_db, inputs=[pdf_files, openai_key], outputs=[db_state, status])
    ask_btn.click(answer, inputs=[question, openai_key, db_state], outputs=output)
    question.submit(answer, inputs=[question, openai_key, db_state], outputs=output)

if __name__ == "__main__":
    demo.launch()