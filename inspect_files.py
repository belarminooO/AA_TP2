import pickle
import os

def inspect_pickle(filepath):
    print(f"--- Inspecting {filepath} ---")
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"Type: {type(data)}")
        if isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")
            for k, v in data.items():
                print(f"Key: {k}, Type: {type(v)}")
                if hasattr(v, 'shape'):
                    print(f"  Shape: {v.shape}")
                elif isinstance(v, list):
                    print(f"  Length: {len(v)}")
                    if len(v) > 0:
                        print(f"  Sample: {v[0]}")
        elif isinstance(data, list):
             print(f"Length: {len(data)}")
             if len(data) > 0:
                 print(f"Sample: {data[0]}")
        else:
            print(f"Content: {data}")
    except Exception as e:
        print(f"Error reading pickle: {e}")

def inspect_pdf(filepath):
    print(f"\n--- Inspecting {filepath} ---")
    try:
        # Try importing pypdf or PyPDF2
        try:
            from pypdf import PdfReader
        except ImportError:
            try:
                from PyPDF2 import PdfReader
            except ImportError:
                print("pypdf/PyPDF2 not installed. Cannot read PDF text directly.")
                return

        reader = PdfReader(filepath)
        print(f"Number of pages: {len(reader.pages)}")
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        with open("pdf_content.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print("PDF content written to pdf_content.txt")
            
    except Exception as e:
        print(f"Error reading PDF: {e}")

if __name__ == "__main__":
    # inspect_pickle("imdbFull.p") # Already done
    inspect_pdf("Enunciado_TP2.pdf")
