import streamlit as st
import google.generativeai as genai
from utils.file_parser import extract_text_from_pdf, extract_text_from_docx
from utils.pinecone_helper import (
    upsert_resume,
    query_resume_context,
    delete_resume_index,
)
import os
from dotenv import load_dotenv

load_dotenv()

# === CONFIGURE GEMINI ===
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel(model_name="gemini-2.0-flash")

# === Streamlit UI ===
st.title("📄 RAG-based Resume Interview Evaluator")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "questions" not in st.session_state:
    st.session_state.questions = []
if "answers" not in st.session_state:
    st.session_state.answers = []

resume_file = st.file_uploader("Upload your Resume (PDF or DOCX)", type=["pdf", "docx"])

if resume_file:
    # Parse uploaded file
    file_text = ""
    if resume_file.type == "application/pdf":
        file_text = extract_text_from_pdf(resume_file)
    elif (
        resume_file.type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        file_text = extract_text_from_docx(resume_file)

    st.success("✅ Resume processed successfully!")

    first_name = st.text_input("Enter candidate's first name")

    if st.button("Index Resume in Pinecone"):
        upsert_resume(file_text, first_name)
        st.success("📌 Resume content indexed successfully!")

    # Interview begins
    if st.button("Start Interview"):
        context = query_resume_context(
            "generate initial interview question", first_name
        )
        response = model.generate_content(
            f"Based on this resume: \n{context}\n\nAsk the first interview question."
        )
        first_question = response.text.strip()

        st.session_state.questions.append(first_question)
        st.session_state.chat_history.append(
            {"role": "interviewer", "text": first_question}
        )

if st.session_state.questions:
    st.subheader("🗣️ Interview Session")

    current_q_index = len(st.session_state.answers)
    if current_q_index < len(st.session_state.questions):
        current_question = st.session_state.questions[current_q_index]
        st.markdown(f"**Q{current_q_index+1}: {current_question}**")

        user_input = st.text_input("Your Answer", key=f"answer_{current_q_index}")

        if user_input:
            st.session_state.answers.append(user_input)
            st.session_state.chat_history.append(
                {"role": "candidate", "text": user_input}
            )

            # Ask next question using Pinecone + Gemini
            if current_q_index < 4:
                context = query_resume_context(user_input, first_name)
                chat_text = "\n".join(
                    [
                        f"{msg['role'].capitalize()}: {msg['text']}"
                        for msg in st.session_state.chat_history
                    ]
                )

                follow_up_prompt = f"""
Resume Context:
{context}

Ongoing Interview:
{chat_text}

Now, ask the next relevant technical question.
"""
                response = model.generate_content(follow_up_prompt)
                next_question = response.text.strip()

                st.session_state.questions.append(next_question)
                st.session_state.chat_history.append(
                    {"role": "interviewer", "text": next_question}
                )
                st.rerun()

# Final Evaluation
if len(st.session_state.answers) == 5:
    st.subheader("🏁 Final Evaluation")

    if st.button("Evaluate Candidate"):
        full_chat = "\n".join(
            [
                f"{m['role'].capitalize()}: {m['text']}"
                for m in st.session_state.chat_history
            ]
        )
        resume_context = query_resume_context(
            "Evaluate based on full interview.", first_name
        )

        final_prompt = f"""
Based on the following interview and the resume context, decide if the candidate should be selected for a software developer role.

Resume Context:
{resume_context}

Interview:
{full_chat}

Return the output in this format:
Final Decision: [Selected / Not Selected]
Reason: [2-3 sentence justification]
"""
        result = model.generate_content(final_prompt)
        st.success("✅ Final Decision:")
        st.markdown(result.text.strip())
        delete_resume_index(first_name)
        # st.info("Your resume data has been deleted from our database for privacy.")
