# cRAG
Here's a Python implementation of the Corrective RAG system as described, using:

1.A pre-indexed PDF file (i.e., you’ve already split, embedded, and indexed chunks using FAISS).

2.Evaluation of retrieved chunks against a query using custom scoring.

3.Correction logic: if the context is poor or fair, refine the query and re-retrieve.

4.Simple streamlit UI and structured output format.

#pre requisites
Add a file named document.pdf in the same directory

Use the file indexx.py to load the document and create vector stores, that act as a pre- requisite index filefor the cRAG code

Use the code cragimplement.py to execute the correctiveRAG code
