from langchain_community.document_loaders import PyPDFLoader
import tempfile

def extract_text(uploaded_file):
    """
    Extracts text content from an uploaded PDF file.

    This function takes an uploaded PDF file, saves it temporarily,
    extracts the text content from all pages, and then removes the
    temporary file.

    Args:
        uploaded_file (UploadedFile): The uploaded PDF file object.

    Returns:
        str: A string containing the extracted text from all pages of the PDF,
             with pages separated by double newlines.

    Raises:
        Any exceptions raised by PyPDFLoader or file operations are not explicitly handled.
    """

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
        
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    text = '\n\n'.join([page.page_content for page in pages])
    
    try:
        os.unlink(tmp_path)
    except:
        pass
        
    return text
