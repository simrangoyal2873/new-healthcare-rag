# healthcare_rag
This project, "healthcare-rag," is a Retrieval-Augmented Generation (RAG) system designed to provide informative responses to healthcare-related queries. It leverages various components to process, store, and retrieve information from a set of documents, combining the retrieved context with a language model to generate accurate and relevant answers.

Key Features:

 **Multi-format document processing**  
  Supports multiple input types like **PDFs, scanned documents, and images (JPG/PNG)**. This makes the system flexible for real-world healthcare files such as prescriptions, lab reports, discharge summaries, and medical images.  

 **OCR integration for scanned documents and images**  
  Uses **Optical Character Recognition (OCR)** to extract text from scanned PDFs and image files. This allows the system to read data even from documents that are not digitally searchable (e.g., scanned hospital reports).  

 **Table & chart data extraction**  
  Extracts structured information from tables (like patient test results, billing info) and charts/graphs (like lab trends, diagnostic graphs). This helps convert visual data into usable structured formats.  

 **Visual element recognition & indexing**  
  Identifies non-textual elements (such as medical diagrams, X-ray images, or icons) and indexes them, enabling retrieval of not just text but also relevant **visual context**.  

 **Local RAG-based search and retrieval (no API cost)**  
  Implements a Retrieval-Augmented Generation (RAG) pipeline using a **local vector database**. This means you can query documents intelligently (semantic search) **without relying on expensive external APIs**, keeping everything private and cost-free.  

 **Interactive Streamlit UI**  
  A simple, lightweight **web interface** built with Streamlit. Users can upload documents, view extracted content, and perform semantic searches—all in an intuitive dashboard.  

Setup and Installation
Prerequisites
Python 3.8+

Create and activate virtual environment:
on window:
python -m venv venv
venv\Scripts\activate

Install dependencies:
The project uses a requirements.txt file to install the dependencies using pi its dependencies. 
pip install -r requirements.txt
Git

Steps
Clone the repository:

git clone https://github.com/simrangoyal2873/new-healthcare-rag.git
cd new-healthcare-rag



Usage
(Further instructions on how to use the system will be added here once the setup is complete.)

Project Structure
vector_db.py: Handles the creation and management of the vector database.

rag_pipeline.py: Contains the core logic for the RAG pipeline, including retrieval and generation.

layout_analyzer.py: A utility script for analyzing document layouts to improve information extraction.

requirements.txt: Lists all Python dependencies.

packages.txt: Lists dependencies to be installed via apt-get (for Linux environments).

README.md: This file.

Contributing
We welcome contributions! If you would like to contribute, please follow these steps:

Fork the repository.

Create a new branch for your feature or bug fix (git checkout -b feature/your-feature-name).

Commit your changes (git commit -m 'feat: Add new feature').

Push to the branch (git push origin feature/your-feature-name).

Open a pull request.
