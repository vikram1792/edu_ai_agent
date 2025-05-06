import streamlit as st
from app_logic import app, format_output, save_pdf  

st.set_page_config(page_title="EduAI Assistant", layout="centered")

st.markdown(
    "<h1 style='text-align: center; color: #2c3e50;'>📘 EduAI Assistant</h1>", 
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; color: #7f8c8d;'>Get summaries, MCQs, bullet points, and book recommendations from PDFs or web topics — perfect for government exam prep!</p>",
    unsafe_allow_html=True
)

option = st.radio(
    "Choose your input type:",
    ("Upload PDF", "Search Topic"),
    horizontal=True
)

input_data = {}

if option == "Upload PDF":
    uploaded_file = st.file_uploader("Upload your study PDF", type=["pdf"])
    if uploaded_file:
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        input_data["pdf_path"] = file_path

elif option == "Search Topic":
    topic = st.text_input("Enter a topic to search (e.g., GST, Indian Constitution, Neutron Star):")
    if topic:
        input_data["topic"] = topic.strip()

if st.button("Generate Study Material", use_container_width=True):
    if not input_data:
        st.warning("Please upload a file or enter a topic.")
    else:
        with st.spinner("Processing... Please wait."):
            result = app.invoke(input_data)
            output_text = format_output(result)

        st.success("Study Material Ready!")
        st.markdown("### Output Preview")
        st.markdown(
            f"<div style='white-space:pre-wrap;word_wrap:break-word;background-color:#f0f0f0;padding:10px;border-radius:5px;'>{output_text[:5000]}<br><br>...(truncated)</div>",
        unsafe_allow_html=True
        )

        save_pdf(output_text)
        with open("output.pdf", "rb") as f:
            st.download_button(
                label="📥 Download Full PDF",
                data=f,
                file_name="study_material.pdf",
                mime="application/pdf"
            )

st.markdown(
    "<hr><p style='text-align: center; color: #bdc3c7;'>Created with LangGraph, OpenAI & Streamlit</p>",
    unsafe_allow_html=True
)