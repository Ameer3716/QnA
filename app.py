import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os

st.set_page_config(page_title="Medical QA", layout="wide")
st.title("Medical RAG QA System")
st.caption("Powered by TinyLlama & Open Source")

@st.cache_resource
def load_system():
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        if not os.path.exists("medical_faiss_index"):
            st.error("FAISS index not found. Please ensure medical_faiss_index folder is in the repository.")
            st.stop()
        
        vectorstore = FAISS.load_local(
            "medical_faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1
        )
        
        llm = HuggingFacePipeline(pipeline=pipe)
        
        template = """<|system|>
You are a medical assistant. Answer based only on the context.</s>
<|user|>
Context: {context}

Question: {question}</s>
<|assistant|>
"""
        
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        return chain
    except Exception as e:
        st.error(f"Error loading system: {str(e)}")
        raise e

with st.spinner("Loading AI models (this may take 2-3 minutes)..."):
    try:
        qa_chain = load_system()
        st.success("System ready")
    except Exception as e:
        st.error(f"Failed to load system: {e}")
        st.stop()

col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input("Ask a medical question:", 
                          placeholder="e.g., What are symptoms of diabetes?")

if st.button("Get Answer", type="primary"):
    if query:
        with st.spinner("Searching medical records..."):
            try:
                result = qa_chain.invoke({"query": query})
                
                answer = result['result']
                if '<|assistant|>' in answer:
                    answer = answer.split('<|assistant|>')[-1].strip()
                
                st.markdown("### Answer")
                st.info(answer)
                
                st.markdown("### Source Documents")
                for i, doc in enumerate(result['source_documents'], 1):
                    with st.expander(f"Source {i}: {doc.metadata.get('specialty', 'N/A')}"):
                        st.text(doc.page_content[:600])
                        
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a question")

st.sidebar.header("System Info")
st.sidebar.markdown("**Tech Stack:**")
st.sidebar.markdown("- TinyLlama 1.1B")
st.sidebar.markdown("- FAISS Vector Store")
st.sidebar.markdown("- MiniLM Embeddings")
