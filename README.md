# Simple-Pdf_Query_Bot

Dependencies: It utilizes Streamlit for web interface, PyPDF2 for PDF text extraction, and various NLP tools like Google Generative AI and FAISS for text processing and indexing.
Functionality: The script allows users to upload research papers, extracts text from them, and indexes the content for efficient search.
Question Answering: Users can ask questions based on the uploaded content, and the script provides answers by analyzing the indexed text.
Memory: It remembers previous interactions, storing questions and answers for users to review.
Interface: The script offers an intuitive web interface for easy interaction.
Overall, it's a streamlined tool for extracting insights from research papers and facilitating interactive discussions based on the content.



@ The script has specific error regarding deserialization, ensuring smooth execution. If users encounter this error, they can expect a prompt solution. Your feedback and suggestions for improvement are welcomed. Thank you for your interest!

Error : "ValueError: The de-serialization relies loading a pickle file. Pickle files can be modified to deliver a malicious payload that results in execution of arbitrary code on your machine.You will need to set `allow_dangerous_deserialization` to `True` to enable deserialization. If you do this, make sure that you trust the source of the data. For example, if you are loading a file that you created, and no that no one else has modified the file, then this is safe to do. Do not set this to `True` if you are loading a file from an untrusted source (e.g., some random site on the internet.)."
