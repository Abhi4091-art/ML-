import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.retrievers import WikipediaRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Q0: Sidebar with LLM selection and API Key
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("Enter your Google API Key", type="password")
llm_choice = st.sidebar.selectbox(
    "Select LLM",
    ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest"]
)
st.title("Market Research Assistant")

# Q1: Get Industry Input
industry = st.text_input("Enter the industry you want to research:")

if industry and api_key:
    # Q2: Retrieve 5 relevant Wikipedia pages
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.invoke(industry)
    
    # Display the retrieved URLs
    st.subheader("Sources Retrieved:")
    for doc in docs:
        st.write(f"- {doc.metadata['source']} (Title: {doc.metadata['title']})")
    
    # Prepare the context for the LLM
    context_text = "\n\n".join([d.page_content for d in docs])
    
    # Q3: Generate Report (Standard RAG Pipeline)
    # This prompt ensures the report is based ONLY on the retrieved text and is <500 words.
    template = """You are a market research assistant. 
    Write a report on the industry: {industry}.
    Base your report ONLY on the following Wikipedia information:
    {context}
    
    The report must be less than 500 words.
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(model=llm_choice, google_api_key=api_key)
    output_parser = StrOutputParser()
    
    # Create the chain
    chain = prompt | llm | output_parser
    
    st.subheader("Industry Report")
    with st.spinner("Generating report..."):
        response = chain.invoke({"industry": industry, "context": context_text})
        st.write(response)

elif not api_key:
    st.warning("Please enter your API Key in the sidebar.")

# Q1 Fix: If no industry is provided, ask the user for an update.
else:
    st.info("Please enter an industry above to generate a report.")


