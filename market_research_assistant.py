import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

st.sidebar.title("Settings")
llm_options = ["gemini-1.5-flash"]
llm_choice = st.sidebar.selectbox("Select LLM", llm_options)
api_key = st.sidebar.text_input("API Key", type="password")

st.title("Market Research Assistant")
industry = st.text_input("Enter an industry:")

if st.button("Generate Report"):
    if not industry.strip():
        st.error("Please provide an industry.")
    else:
        # Step 2: Get 5 URLs
        retriever = WikipediaRetriever()
        docs = retriever.get_relevant_documents(industry)
        urls = [doc.metadata['source'] for doc in docs[:5]]
        st.subheader("Relevant Wikipedia Pages:")
        for url in urls:
            st.write(url)
        
        # Step 3: Generate report
        llm = ChatGoogleGenerativeAI(model=llm_choice, api_key=api_key, temperature=0)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
        query = f"Provide a concise market research report on the {industry} industry, less than 500 words, based on the retrieved information."
        report = qa_chain.run(query)
        st.subheader("Industry Report:")
        st.write(report)
        st.write(f"Word count: {len(report.split())}")