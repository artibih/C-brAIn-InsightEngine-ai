import os
import time
from dotenv import load_dotenv
from mistralai import Mistral
from src.document.base import Document
import shutil
import uuid
class MistralOCRPDFLoader:

    def __init__(self):
        load_dotenv()

        self.mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

    def extract_text_from_pdf(self, pdf_path: str) -> Document:
        file_obj = None
        try:
            # Open file with explicit close
            file_obj = open(pdf_path, "rb")
            file_content = file_obj.read()
            file_obj.close()
            
            # uploaded_pdf = self.mistral_client.files.upload(
            #     file={
            #         "file_name": "tmp_upload",
            #         "content": file_content,  # Use the content we read
            #     },
            #     purpose="ocr"
            # )
            with open(pdf_path, "rb") as f:
                uploaded_pdf = self.mistral_client.files.upload(
                    file={
                        "file_name": os.path.basename(pdf_path),
                        "content": f,
                    },
                    purpose="ocr"
                )

            # signed_url = self.mistral_client.files.get_signed_url(file_id=uploaded_pdf.id)
            signed_url = None

            for attempt in range(5):
                try:
                    signed_url = self.mistral_client.files.get_signed_url(
                        file_id=uploaded_pdf.id,
                        expiry=24
                    )
                    break
                except Exception as e:
                    if attempt == 4:
                        raise
                    time.sleep(2)
            retries = 0
            max_retries = 3
            ocr_response = None

            while retries < max_retries:    
                retries += 1
                try:
                    ocr_response = self.mistral_client.ocr.process(
                        model="mistral-ocr-latest",
                        document={
                            "type": "document_url",
                            "document_url": signed_url.url,
                        },
                        include_image_base64=True
                    )
                    break
                except Exception as e:
                    if retries < max_retries:
                        time.sleep(2 ** (retries - 1))
                        continue
                    raise

            self.mistral_client.files.delete(file_id=uploaded_pdf.id)

            text = '\n'.join([page.markdown for page in ocr_response.pages])

            images = []
            for page in ocr_response.pages:
                for image in page.images:
                    images.append(image.image_base64)
            paper_id = str(uuid.uuid4())

            doc = Document(
                    text=text,
                    metadata={
                        "paper_id": paper_id,
                        "doi": None,
                        "title": None,
                        "authors": None,
                        "abstract": None,
                        "original_path": pdf_path,
                    }
                )
            doc.content['images'] = images
            # doc.metadata['original_path'] = pdf_path

            return doc
        
        except Exception as e:
            # Make sure file is closed even on exception
            if file_obj and not file_obj.closed:
                file_obj.close()
            raise