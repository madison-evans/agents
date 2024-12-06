from langchain.tools import BaseTool
from langchain_core.documents import Document

class RetrieveDocuments(BaseTool):
    """
    A tool to retrieve documents based on a query.
    Simulates a RAG pipeline and returns mock documents related to chemistry.
    """

    name: str = "retrieve_documents"
    description: str = "Retrieves relevant documents for a given query."

    def _run(self, query: str) -> list[Document]:
        """
        Perform the document retrieval.
        """
        print(f"Performing retrieval based on this query:\n\n{query}\n\n")

        documents = [
            Document(
                page_content=(
                    "Water is often referred to as the universal solvent due to its "
                    "ability to dissolve more substances than any other liquid."
                ),
                metadata={"source": "https://chemistryfacts.com/water-solvent"},
            ),
            Document(
                page_content=(
                    "The periodic table organizes all known chemical elements by "
                    "their atomic number, electron configuration, and recurring properties."
                ),
                metadata={"source": "https://chemistryfacts.com/periodic-table"},
            ),
            Document(
                page_content=(
                    "Acids release hydrogen ions (H+) in solution, while bases release "
                    "hydroxide ions (OH-). The pH scale measures acidity or alkalinity."
                ),
                metadata={"source": "https://chemistryfacts.com/acids-and-bases"},
            ),
            Document(
                page_content=(
                    "Catalysts increase the rate of a chemical reaction by lowering the "
                    "activation energy required for the reaction to proceed."
                ),
                metadata={"source": "https://chemistryfacts.com/catalysts"},
            ),
            Document(
                page_content=(
                    "Carbon is a versatile element that forms the backbone of organic chemistry, "
                    "found in compounds like DNA and CO2."
                ),
                metadata={"source": "https://chemistryfacts.com/carbon"},
            ),
        ]
        return documents

    async def _arun(self, query: str) -> list[Document]:
        """
        Asynchronous version of document retrieval.
        """
        return self._run(query)
