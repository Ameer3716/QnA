import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

st.set_page_config(page_title="Medical QA", page_icon="🏥", layout="wide")
st.title("Medical RAG QA System")
st.caption("Powered by TinyLlama & Open Source")

@st.cache_resource
def load_system():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    vectorstore = FAISS.load_local("medical_faiss_index", embeddings, allow_dangerous_deserialization=True)
    
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

with st.spinner("Loading AI models (this may take 2-3 minutes)..."):
    try:
        qa_chain = load_system()
        st.success("System ready")
    except Exception as e:
        st.error(f"Error loading system: {e}")
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

st.sidebar.header("System Evaluation")
st.sidebar.info("Test with 30 predefined medical queries")

eval_queries = [
    "What are symptoms of diabetes?", "How to diagnose hypertension?", 
    "Treatment for asthma?", "Signs of heart failure?", "Kidney disease tests?",
    "What is pneumonia?", "Symptoms of stroke?", "How to treat COPD?",
    "What causes chest pain?", "Diagnosis of arthritis?", "Treatment for depression?",
    "What is anemia?", "Symptoms of thyroid disorder?", "How to manage obesity?",
    "What is sepsis?", "Treatment for migraine?", "Diagnosis of cancer?",
    "What causes fatigue?", "How to treat allergies?", "Symptoms of liver disease?",
    "What is epilepsy?", "Treatment for anxiety?", "Diagnosis of tuberculosis?",
    "What causes dizziness?", "How to manage pain?", "Symptoms of infection?",
    "What is dementia?", "Treatment for ulcers?", "Diagnosis of osteoporosis?",
    "What causes fever?"
]

if st.sidebar.button("Run Full Evaluation"):
    progress = st.sidebar.progress(0)
    status = st.sidebar.empty()
    results = []
    
    for i, q in enumerate(eval_queries):
        status.text(f"Processing {i+1}/30...")
        try:
            res = qa_chain.invoke({"query": q})
            answer = res['result']
            if '<|assistant|>' in answer:
                answer = answer.split('<|assistant|>')[-1].strip()
            results.append(f"Q{i+1}: {q}\n\nA: {answer[:250]}...\n\n{'='*80}\n")
        except:
            results.append(f"Q{i+1}: {q}\n\nA: [Error processing query]\n\n{'='*80}\n")
        
        progress.progress((i + 1) / 30)
    
    status.text("Complete")
    
    full_text = "\n".join(results)
    st.sidebar.download_button(
        "Download Results",
        full_text,
        "medical_rag_evaluation.txt",
        mime="text/plain"
    )

st.sidebar.markdown("---")
st.sidebar.markdown("**Tech Stack:**")
st.sidebar.markdown("• TinyLlama 1.1B")
st.sidebar.markdown("• FAISS Vector Store")
st.sidebar.markdown("• MiniLM Embeddings")
