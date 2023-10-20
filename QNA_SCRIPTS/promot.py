from langchain import PromptTemplate
def create_prompt_template():
    # prepare the template we will use when prompting the AI
    promote_template = """Use the provided context to answer the user's question.
    If you don't know the answer, respond with "I do not know".

    Context: {context}
    Question: {question}
    Answer:
    """

    prompt = PromptTemplate(
        template=promote_template,
        input_variables=['context', 'question']
    )
    