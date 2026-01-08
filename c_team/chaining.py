import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from stores import get_vector_store

load_dotenv()


def get_retriever():
    vector_store = get_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": 5})


def get_chain():
    retriever = get_retriever()

    llm = ChatOpenAI(model_name="gpt-5", api_key=os.getenv("OPENAI_API_KEY"))
    prompt = ChatPromptTemplate.from_template(
        """
            당신은 **형사법** 법령 검색 도움 봇입니다. 
            제공된 {context}를 바탕으로 사용자의 {question}에 답변해 주세요.
            Question이 **형사법**에 해당하지 않는 법령에 대한 질문이라면 관련 법령을 검색할 수 없다고 답변하고 전문가의 도움을 받기를 제안하세요.

            **Required Rule**
                1. 반드시 답변의 근거가 된 참고자료({context})를 포함해야 합니다.
                2. {context}가 없을 경우 관련 법령을 검색할 수 없다고 답변하고 전문가의 도움을 받기를 제안하세요.
                3. **형사법**에 대한 답변만을 생성해야 합니다.
                4. 당신은 법령 정보만 제공합니다. 추가 의견은 추가하지 않습니다.

            Context:
            {context}

            Question:
            {question}
        """
    )

    def format_str(docs: list) -> str:
        return "\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_str, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
