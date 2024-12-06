from typing import Annotated, List, TypedDict
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain_core.documents import Document
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
load_dotenv()

class State(TypedDict): 
    messages: Annotated[list, add_messages]
    revised_query: str

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def is_chemistry_related(message: str) -> bool:
    """
    Simple function to determine if a query is about chemistry.
    """
    keywords = ["chemistry", "chemical", "compound", "reaction", "molecule"]
    return any(keyword in message.lower() for keyword in keywords)


def check_query_type(state: State) -> str:
    """
    Conditional function to decide the next node.
    """
    user_message = state["messages"][-1].content
    if is_chemistry_related(user_message):
        return "rewrite_query"
    return "general_response"

def general_response(state: State) -> State:
    """
    Handles general conversation flow by generating a response using the LLM.
    """

    conversation_history = ""
    for message in state["messages"]:
        if isinstance(message, HumanMessage):
            conversation_history += f"User: {message.content}\n"
        elif isinstance(message, AIMessage):
            conversation_history += f"Assistant: {message.content}\n"

    prompt = PromptTemplate(
        input_variables=["conversation_history", "new_user_message"],
        template=(
            "You are a helpful and friendly assistant.\n"
            "Conversation history: {conversation_history}\n\n"
            "User: {new_user_message}\n\n"
            "Assistant response:"
        ),
    )

    user_message = state["messages"][-1].content
    formatted_prompt = prompt.format(
        conversation_history=conversation_history, 
        new_user_message=user_message
    )

    ai_message = llm.invoke([HumanMessage(content=formatted_prompt)])

    state["messages"].append(AIMessage(content=ai_message.content))

    return state

def rewrite_query(state: State) -> State: 
    conversation_history = ""
    for message in state["messages"]:
        if isinstance(message, HumanMessage):
            conversation_history += f"User: {message.content}\n"
        elif isinstance(message, AIMessage):
            conversation_history += f"Assistant: {message.content}\n"
    prompt = PromptTemplate(
        input_variables=["conversation_history", "new_user_message"],
        template=(
            "Based on the provided 'Conversation history' and 'User query', create a 'Revised Query' which rewords the query to be contextually rich, by including only the relevant elements of the Conversation history\n\n"
            "Conversation history: {conversation_history}\n\n"
            "User Query: {new_user_message}\n\n"
            "Revised Query:"
        ),
    )

    # Format the prompt using the conversation history and the latest user message
    user_message = state["messages"][-1].content
    formatted_prompt = prompt.format(
        conversation_history=conversation_history, 
        new_user_message=user_message
    )
    # Generate the AI response
    ai_message = llm.invoke([HumanMessage(content=formatted_prompt)])
    print(f"revised_query: {ai_message.content}")
    state["revised_query"] = ai_message.content

    return state

def retrieve_documents(revised_query: str) -> List[Document]:
    """
    Retrieves mock documents related to chemistry, based on the revised query.
    """
    print(f"Performing retrieval based on this query:\n\n{revised_query}\n\n")

    # Mock documents related to chemistry
    return [
        Document(
            page_content=(
                "Water is often referred to as the universal solvent due to its ability to dissolve more substances "
                "than any other liquid. This is because water molecules are polar, with a partial negative charge on the "
                "oxygen atom and partial positive charges on the hydrogen atoms. This polarity allows water to interact "
                "with and stabilize ions and polar molecules, making it crucial for chemical reactions in biological and "
                "industrial systems."
            ),
            metadata={"source": "https://chemistryfacts.com/water-solvent"}
        ),
        Document(
            page_content=(
                "The periodic table of elements is one of the most significant achievements in science, organizing "
                "all known chemical elements by their atomic number, electron configuration, and recurring chemical properties. "
                "It allows scientists to predict the behavior of elements and their compounds, serving as a framework for "
                "understanding chemical reactivity and trends, such as electronegativity and ionization energy."
            ),
            metadata={"source": "https://chemistryfacts.com/periodic-table"}
        ),
        Document(
            page_content=(
                "Acids and bases play a vital role in chemistry and everyday life. Acids, like hydrochloric acid (HCl), "
                "release hydrogen ions (H+) in solution, while bases, such as sodium hydroxide (NaOH), release hydroxide ions (OH-). "
                "The pH scale, ranging from 0 to 14, measures the acidity or alkalinity of a solution, with 7 being neutral. "
                "These reactions are critical in processes like digestion, industrial manufacturing, and environmental regulation."
            ),
            metadata={"source": "https://chemistryfacts.com/acids-and-bases"}
        ),
        Document(
            page_content=(
                "Catalysts are substances that increase the rate of a chemical reaction without being consumed in the process. "
                "They work by lowering the activation energy required for the reaction to proceed, allowing it to occur more "
                "rapidly. Catalysts are used in a wide range of applications, from the synthesis of ammonia in the Haber process "
                "to catalytic converters in vehicles that reduce harmful emissions."
            ),
            metadata={"source": "https://chemistryfacts.com/catalysts"}
        ),
        Document(
            page_content=(
                "Carbon is a versatile element that forms the backbone of organic chemistry. With its ability to form four covalent bonds, "
                "carbon can create complex structures such as chains, rings, and frameworks found in biomolecules like DNA and proteins. "
                "It also appears in inorganic compounds like carbon dioxide (CO2), playing a key role in processes such as photosynthesis "
                "and the carbon cycle."
            ),
            metadata={"source": "https://chemistryfacts.com/carbon"}
        )
    ]



def generate_answer_from_retrieval(state: State) -> State:
    """
    Generates a final answer using the LLM based on retrieved documents and the revised query.
    """
    # Extract the revised query
    revised_query = state["revised_query"]

    # Retrieve documents using the helper function
    retrieved_documents = retrieve_documents(revised_query)

    # Combine the content of retrieved documents into a single string
    document_context = "\n".join(
        [
            f"Document {i+1}: {doc.page_content} (source: {doc.metadata['source']})"
            for i, doc in enumerate(retrieved_documents)
        ]
    )

    # Create a prompt template for answer generation
    prompt = PromptTemplate(
        input_variables=["revised_query", "document_context"],
        template=(
            "You are a knowledgeable assistant. Using the provided 'Revised Query' and 'Document Context', "
            "generate a clear and concise answer to the query.\n\n"
            "Revised Query:\n{revised_query}\n\n"
            "Document Context:\n{document_context}\n\n"
            "Final Answer:"
        ),
    )

    # Format the prompt using the revised query and document context
    formatted_prompt = prompt.format(
        revised_query=revised_query,
        document_context=document_context
    )

    # Generate the final answer using the LLM
    ai_response = llm.invoke([HumanMessage(content=formatted_prompt)])

    # Append the final answer as an AI message to the conversation history
    state["messages"].append(AIMessage(content=ai_response.content))

    return state
