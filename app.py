import streamlit as st
import PyPDF2
import fitz  # PyMuPDF
import io
import json
from datetime import datetime
import random
import re
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="PDF Quiz Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .quiz-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .question-block {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .correct-answer {
        color: #28a745;
        font-weight: bold;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #5a6fd8 0%, #6a4190 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pdf_text' not in st.session_state:
    st.session_state.pdf_text = ""
if 'generated_questions' not in st.session_state:
    st.session_state.generated_questions = []
if 'quiz_generated' not in st.session_state:
    st.session_state.quiz_generated = False
if 'llm_provider' not in st.session_state:
    st.session_state.llm_provider = os.getenv('LLM_PROVIDER', 'gemini')
if 'show_answers' not in st.session_state:
    st.session_state.show_answers = True

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="PDF Quiz Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .quiz-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .question-block {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .correct-answer {
        color: #28a745;
        font-weight: bold;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #5a6fd8 0%, #6a4190 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pdf_text' not in st.session_state:
    st.session_state.pdf_text = ""
if 'generated_questions' not in st.session_state:
    st.session_state.generated_questions = []
if 'quiz_generated' not in st.session_state:
    st.session_state.quiz_generated = False
if 'llm_provider' not in st.session_state:
    st.session_state.llm_provider = os.getenv('LLM_PROVIDER', 'openai')

# LLM Configuration
class LLMConfig:
    def __init__(self):
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.gemini_model = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
        
        # Initialize Gemini client
        if self.gemini_api_key and self.gemini_api_key != 'your_gemini_api_key_here':
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_client = genai.GenerativeModel(self.gemini_model)
        else:
            self.gemini_client = None

llm_config = LLMConfig()

def clean_text(text):
    """Clean text with robust character filtering"""
    if not text:
        return ""
    
    cleaned = ""
    for char in text:
        try:
            # Skip surrogate characters
            if 0xD800 <= ord(char) <= 0xDFFF:
                cleaned += " "
            # Skip other problematic Unicode ranges
            elif ord(char) > 0x10FFFF:
                cleaned += " "
            else:
                char.encode('utf-8')
                cleaned += char
        except (UnicodeEncodeError, ValueError):
            cleaned += " "
    
    # Clean up whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()

def clean_text_aggressive(text):
    """More aggressive text cleaning for problematic PDFs"""
    if not text:
        return ""
    
    # Keep only printable ASCII and common Unicode characters
    cleaned = ""
    for char in text:
        if (32 <= ord(char) <= 126) or char in '\n\t\r' or (160 <= ord(char) <= 255):
            cleaned += char
        elif ord(char) > 255:
            # Replace other Unicode with space
            cleaned += " "
    
    # Clean up multiple spaces and line breaks
    cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)  # Normalize line breaks
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)       # Normalize spaces
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)   # Limit consecutive line breaks
    
    return cleaned.strip()

def extract_text_from_pdf(pdf_file):
    """
    Extract text and images from uploaded PDF file using multiple methods.
    Uses PyMuPDF (fitz) first for robust extraction, then falls back to PyPDF2 and others.
    Returns extracted text. Displays images in Streamlit if found.
    """

    # Method 1: PyMuPDF (fitz) for text and images
    def extract_with_pymupdf():
        try:
            pdf_file.seek(0)
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            text = ""
            images = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                # Extract text
                page_text = page.get_text()
                if page_text:
                    text += page_text + "\n\n"
                # Extract images
                img_list = page.get_images(full=True)
                for img_index, img in enumerate(img_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    images.append((f"Page {page_num+1} Image {img_index+1}", image_bytes, image_ext))
            if not text.strip() and not images:
                raise Exception("No text or images found in PDF")
            return text.strip(), images, f"PyMuPDF (fitz) (extracted {len(doc)} pages)"
        except Exception as e:
            raise Exception(f"PyMuPDF failed: {str(e)}")

    # Method 2: Enhanced PyPDF2 with robust error handling
    def extract_with_pypdf2():
        try:
            pdf_file.seek(0)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            successful_pages = 0
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        cleaned_text = clean_text(page_text)
                        if cleaned_text.strip():
                            text += cleaned_text + "\n\n"
                            successful_pages += 1
                except Exception as page_error:
                    continue  # Skip problematic pages silently
            if successful_pages == 0:
                raise Exception("No pages could be successfully extracted")
            return text.strip(), [], f"PyPDF2 (extracted {successful_pages} pages)"
        except Exception as e:
            raise Exception(f"PyPDF2 failed: {str(e)}")

    # Method 3: Alternative extraction with different approach
    def extract_with_alternative():
        try:
            pdf_file.seek(0)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        cleaned = clean_text_aggressive(page_text)
                        if cleaned:
                            text += cleaned + "\n\n"
                except:
                    continue
            if not text.strip():
                raise Exception("No readable content found")
            return text.strip(), [], "Alternative PyPDF2 method"
        except Exception as e:
            raise Exception(f"Alternative method failed: {str(e)}")

    # Method 4: Fallback with basic text extraction
    def extract_basic_fallback():
        try:
            pdf_file.seek(0)
            content = pdf_file.read()
            text_content = str(content, errors='ignore')
            text_matches = re.findall(r'[A-Za-z\s]{10,}', text_content)
            if text_matches:
                extracted_text = ' '.join(text_matches[:50])
                return clean_text(extracted_text), [], "Basic fallback extraction"
            else:
                raise Exception("No readable text patterns found")
        except Exception as e:
            raise Exception(f"Fallback method failed: {str(e)}")

    # Try extraction methods in order
    methods = [
        ("PyMuPDF (fitz)", extract_with_pymupdf),
        ("Enhanced PyPDF2", extract_with_pypdf2),
        ("Alternative extraction", extract_with_alternative),
        ("Basic fallback", extract_basic_fallback)
    ]
    last_error = None
    for method_name, method_func in methods:
        try:
            pdf_file.seek(0)
            result = method_func()
            if len(result) == 3:
                text, images, method_used = result
            else:
                text, images, method_used = result[0], [], result[1]
            if (text and len(text.strip()) > 50) or images:
                return text
            else:
                raise Exception("Extracted text too short or empty and no images found")
        except Exception as e:
            last_error = str(e)
            continue
    
    # If all methods failed, return None to indicate failure
    return None

def generate_llm_questions(text, difficulty, num_questions, provider='gemini', regenerate=False):
    """
    Generate quiz questions using Google Gemini API that are directly relevant to the PDF content
    """
    
    # Add randomness for regeneration
    import random
    import time
    
    # Use different sections of text for variety
    text_sections = [text[i:i+4000] for i in range(0, min(len(text), 12000), 2000)]
    if regenerate and len(text_sections) > 1:
        # Use a different section or combination for regeneration
        text_preview = random.choice(text_sections)
        if len(text_sections) > 2:
            # Sometimes combine sections for variety
            if random.choice([True, False]):
                text_preview = text_sections[0][:2000] + text_sections[-1][:2000]
    else:
        text_preview = text[:4000]  # Default for first generation
    
    # Extract key elements for better prompting
    key_terms = re.findall(r'\b[A-Z][a-zA-Z\s]{2,25}\b', text_preview)[:10]
    dates_years = re.findall(r'\b(?:\d{4}|\d{1,2}/\d{1,2}/\d{4})\b', text_preview)[:5]
    percentages = re.findall(r'\b\d+(?:\.\d+)?%\b', text_preview)[:5]
    
    # Create varied prompts for regeneration
    if regenerate:
        prompt_variations = [
            "Focus on different aspects and create varied questions",
            "Emphasize numerical data, dates, and specific facts",
            "Create questions about relationships and processes described",
            "Focus on key terminology and technical concepts",
            "Generate questions about cause-and-effect relationships"
        ]
        prompt_instruction = random.choice(prompt_variations)
        additional_instruction = f"\n\nSPECIAL INSTRUCTION: {prompt_instruction}. Create questions that are different from what might have been asked before about this content."
    else:
        additional_instruction = ""
    
    # Create the enhanced prompt for question generation
    prompt = f"""
You are creating a quiz based on the following document content. You must read the text carefully and create questions that test understanding of the SPECIFIC information, facts, and concepts presented in this exact document.

DOCUMENT CONTENT:
{text_preview}

EXTRACTED KEY ELEMENTS (use these in your questions):
- Key Terms: {', '.join(key_terms[:5]) if key_terms else 'N/A'}
- Dates/Years: {', '.join(dates_years) if dates_years else 'N/A'}
- Percentages/Numbers: {', '.join(percentages) if percentages else 'N/A'}

TASK: Create {num_questions + 3} quiz questions at {difficulty} difficulty level that are DIRECTLY based on the content above.{additional_instruction}

CONTENT-SPECIFICITY REQUIREMENTS:
1. Questions MUST reference specific information that appears in the document
2. Use actual names, dates, numbers, or facts mentioned in the text
3. For multiple choice: correct answer should be explicitly stated in the document
4. For true/false: base statements on actual claims made in the document
5. Avoid generic questions that could apply to any document on this topic

DIFFICULTY GUIDELINES:
- Easy: Direct recall of explicitly stated facts, definitions, or numbers from the text
- Medium: Understanding relationships between concepts or cause-and-effect mentioned in the document
- Hard: Analysis or synthesis of multiple specific concepts from the document

CRITICAL JSON FORMATTING RULES:
1. Return ONLY valid JSON - no explanatory text before or after
2. Use DOUBLE quotes for all strings - never single quotes
3. For options arrays, each option must be properly quoted: ["Option 1", "Option 2", "Option 3", "Option 4"]
4. If an option contains quotes, escape them with backslash: "He said \"hello\" to me"
5. Do not use nested quotes like "Option with "quotes" inside" - use "Option with \"quotes\" inside"
6. Remove any special characters that break JSON (smart quotes, em dashes, etc.)
7. Do not include trailing commas
8. Ensure proper array structure

OPTION CREATION GUIDELINES (VERY IMPORTANT):
- For multiple choice questions, create 4 options that are ALL plausible and relevant to the question topic
- The CORRECT option should contain the exact answer from the document
- The 3 INCORRECT options should be related to the same topic but clearly wrong based on the document
- Use information from the document to create realistic but incorrect distractors
- Avoid generic options like "Option A", "Option B" - make them specific to the content
- All options should be similar in length and complexity
- Make sure incorrect options test similar knowledge but have clear distinctions

OPTION FORMATTING EXAMPLES:
Correct: "options": ["Workshop held in November 2023", "Conference scheduled for December 2023", "Meeting planned for January 2024", "Seminar organized for February 2024"]
Correct: "options": ["25% increase in efficiency", "15% decrease in performance", "40% improvement in quality", "30% reduction in costs"]
Wrong: "options": ["Option containing exact information from document", "Plausible but incorrect option", "Another plausible but wrong option", "Fourth plausible but wrong option"]
Wrong: "options": ["True answer", "False answer", "Maybe answer", "Possibly answer"]

QUESTION QUALITY EXAMPLES:
Good: "According to the document, what percentage is mentioned regarding [specific statistic from text]?"
Good: "The text states that [specific person/entity] did what in [specific year from document]?"
Poor: "What is the main concept discussed?" (too generic)
Poor: "Which of the following is important?" (not content-specific)

JSON FORMAT (return ONLY this exact structure):
[
  {{
    "question": "According to the document, when was the [specific event/workshop/meeting mentioned in the text] held?",
    "type": "multiple_choice",
    "options": ["November 15, 2023", "December 20, 2023", "January 10, 2024", "February 5, 2024"],
    "correct_answer": "A",
    "difficulty": "{difficulty}",
    "explanation": "The document specifically states that [specific event] was held on November 15, 2023"
  }},
  {{
    "question": "True or False: The document states that [specific claim/fact from the text]",
    "type": "true_false", 
    "options": ["True", "False"],
    "correct_answer": "True",
    "difficulty": "{difficulty}",
    "explanation": "This is true because the document explicitly mentions [specific reference with details]"
  }}
]

IMPORTANT: Every question must be answerable ONLY by someone who has read THIS specific document. Use specific facts, names, numbers, and claims from the text provided above. Make options realistic and content-specific.
"""

    try:
        if provider == 'gemini' and llm_config.gemini_client:
            # Use a more balanced approach for regeneration
            system_prompt = """You are an expert educational content creator. 
            
CRITICAL REQUIREMENTS:
1. Return ONLY valid JSON - no markdown, no explanations, no extra text
2. Use proper double quotes for all strings
3. Escape quotes inside strings with backslash
4. Follow the exact JSON structure provided in the example
5. Keep content professional and avoid irregular punctuation"""
            
            full_prompt = f"{system_prompt}\n\n{prompt}"
            
            # Use more moderate temperature for regeneration to balance variety with stability
            if regenerate:
                # Reduce temperature after multiple regenerations to maintain stability
                regeneration_count = getattr(st.session_state, 'regeneration_count', 1)
                if regeneration_count <= 2:
                    temperature = 0.4  # Moderate creativity for first few regenerations
                elif regeneration_count <= 4:
                    temperature = 0.25  # More conservative after several attempts
                else:
                    temperature = 0.15  # Very conservative for many regenerations
            else:
                temperature = 0.1  # Conservative for first generation
            
            # Add multiple attempts for regeneration to get better results
            max_attempts = 3 if regenerate else 1
            response_text = None
            
            for attempt in range(max_attempts):
                try:
                    response = llm_config.gemini_client.generate_content(
                        full_prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=temperature,
                            max_output_tokens=3000,
                            candidate_count=1,
                            response_mime_type="application/json"
                        )
                    )
                    
                    response_text = response.text.strip()
                    
                    # Quick validation check - if it looks malformed, try again
                    if regenerate and attempt < max_attempts - 1:
                        # Basic validation: check for proper JSON structure
                        if not (response_text.strip().startswith('[') and response_text.strip().endswith(']')):
                            continue
                        # Check for excessive irregular punctuation
                        irregular_chars = len(re.findall(r'[^\w\s\[\]{}":,.\-?!()]', response_text))
                        if irregular_chars > 10:
                            continue
                    
                    # If we get here, the response looks acceptable
                    break
                    
                except Exception as e:
                    if attempt < max_attempts - 1:
                        continue
                    else:
                        raise e
            
            if not response_text:
                raise Exception("Failed to get valid response after multiple attempts")
            
        else:
            raise Exception(f"No valid Gemini API key found")
        
        # Clean the response to extract JSON with enhanced validation
        
        # Enhanced response validation for regeneration attempts
        def validate_response_quality(text):
            """Check if response has excessive irregular patterns"""
            issues = []
            
            # Check for excessive special characters
            special_char_count = len(re.findall(r'[^\w\s\[\]{}":,.\-?!()]', text))
            if special_char_count > 15:
                issues.append(f"Too many special characters ({special_char_count})")
            
            # Check for malformed JSON structure
            if not (text.strip().startswith('[') or text.strip().startswith('{')):
                issues.append("Missing JSON opening bracket")
            
            if not (text.strip().endswith(']') or text.strip().endswith('}')):
                issues.append("Missing JSON closing bracket")
            
            # Check for excessive repeated punctuation
            repeated_punct = len(re.findall(r'([.!?]){3,}|[-]{4,}', text))
            if repeated_punct > 3:
                issues.append(f"Excessive repeated punctuation ({repeated_punct})")
            
            return issues
        
        # Validate response quality
        quality_issues = validate_response_quality(response_text)
        if quality_issues and regenerate:
            pass  # Continue processing even with quality issues
        
        # Remove markdown code blocks and clean whitespace
        response_text = re.sub(r'^```(?:json)?\s*', '', response_text, flags=re.IGNORECASE)
        response_text = re.sub(r'\s*```$', '', response_text)
        response_text = response_text.strip()
        
        # Advanced JSON cleaning and parsing
        def fix_json_string(json_str):
            """Apply comprehensive JSON fixes with enhanced irregular pattern handling"""
            # Remove any non-JSON content at the beginning or end
            json_str = re.sub(r'^[^[{]*', '', json_str)
            json_str = re.sub(r'[^}\]]*$', '', json_str)
            
            # Clean up irregular punctuation and characters first
            # Remove excessive punctuation and weird regex patterns
            json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)  # Remove control characters
            json_str = re.sub(r'[^\w\s\[\]{}":,.\-?!()]+', '', json_str)  # Remove excessive special chars
            
            # Fix multiple consecutive punctuation marks
            json_str = re.sub(r'([.!?]){3,}', r'\1', json_str)  # Reduce multiple punctuation
            json_str = re.sub(r'[-]{3,}', '-', json_str)  # Fix excessive dashes
            
            # Find options arrays and fix quotes inside them
            def fix_options_array(match):
                full_match = match.group(0)
                options_content = match.group(1)
                
                # Clean irregular patterns in options
                options_content = re.sub(r'[^\w\s\[\],"\-?!.()\/:]+', '', options_content)
                
                # Split by commas but be careful about nested quotes
                # Fix unescaped quotes that aren't at array boundaries
                fixed_content = options_content
                
                # Replace standalone quotes that aren't escaped and aren't array delimiters
                fixed_content = re.sub(r'(?<!\\)(?<![\[\,\s])"(?![,\]\}\s])', '\\"', fixed_content)
                
                return f'"options": [{fixed_content}]'
            
            # Apply the fix to options arrays
            json_str = re.sub(r'"options":\s*\[([^\]]+)\]', fix_options_array, json_str, flags=re.DOTALL)
            
            # General quote fixes
            # Fix unescaped quotes in string values (but preserve array/object syntax)
            json_str = re.sub(r'(?<=[:\s])"([^"]*)"([^"]*)"([^"]*)"(?=\s*[,\]\}])', r'"\1\2\3"', json_str)
            
            # Replace problematic characters
            json_str = json_str.replace('"', '"').replace('"', '"')  # Smart quotes
            json_str = json_str.replace(''', "'").replace(''', "'")  # Smart apostrophes
            json_str = json_str.replace('–', '-').replace('—', '-')  # Em/en dashes
            
            # Fix trailing commas
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
            
            # Ensure proper JSON array structure
            json_str = json_str.strip()
            if not json_str.startswith('['):
                json_str = '[' + json_str
            if not json_str.endswith(']'):
                json_str = json_str + ']'
            
            return json_str
        
        # Try multiple parsing strategies
        questions = None
        parsing_errors = []
        
        # Pre-compile regex patterns to avoid f-string backslash issues
        array_pattern = r'\[.*\]'
        object_pattern = r'\{.*\}'
        
        array_match = re.search(array_pattern, response_text, re.DOTALL)
        object_match = re.search(object_pattern, response_text, re.DOTALL)
        
        def aggressive_quote_fix(text):
            """Last resort quote fixing for malformed options"""
            # Specifically target the options arrays that are causing issues
            def fix_option_quotes(match):
                options_part = match.group(1)
                # Replace unescaped quotes in the middle of option strings
                # Look for: "text"text" and replace with "text\"text"
                fixed = re.sub(r'"([^"]*)"([^",\]]*)"', r'"\1\\\"\2"', options_part)
                return f'"options": [{fixed}]'
            
            # Apply the fix
            fixed_text = re.sub(r'"options":\s*\[([^\]]+)\]', fix_option_quotes, text)
            return fixed_text
        
        strategies = [
            ("Direct parsing", response_text),
            ("Basic cleaning", fix_json_string(response_text)),
            ("Extract array", array_match.group(0) if array_match else None),
            ("Extract object as array", f"[{object_match.group(0)}]" if object_match else None),
            ("Aggressive quote fixing", aggressive_quote_fix(response_text))
        ]
        
        for strategy_name, candidate in strategies:
            if candidate is None:
                continue
                
            try:
                # Apply additional cleaning for this candidate
                cleaned_candidate = fix_json_string(candidate)
                
                # Try to parse
                parsed_data = json.loads(cleaned_candidate)
                
                # Validate the structure
                if isinstance(parsed_data, list) and len(parsed_data) > 0:
                    questions = parsed_data
                    break
                elif isinstance(parsed_data, dict):
                    questions = [parsed_data]
                    break
                    
            except json.JSONDecodeError as e:
                error_msg = f"{strategy_name} failed: {str(e)}"
                parsing_errors.append(error_msg)
                continue
            except Exception as e:
                error_msg = f"{strategy_name} error: {str(e)}"
                parsing_errors.append(error_msg)
                continue
        
        # If all parsing failed, try one last manual approach
        if questions is None:
            try:
                # Enhanced manual question extraction using multiple patterns
                questions = []
                
                # Pattern 000: Improved question pattern that handles quotes better
                # Use non-greedy matching and better quote handling
                question_pattern = r'"question":\s*"((?:[^"\\]|\\.)*)"\s*,.*?"type":\s*"([^"]+)"\s*,.*?"options":\s*\[(.*?)\].*?"correct_answer":\s*"([^"]+)"\s*,.*?"explanation":\s*"((?:[^"\\]|\\.)*)"'
                matches = re.findall(question_pattern, response_text, re.DOTALL)
                
                if matches:
                    for match in matches:
                        question_text, q_type, options_str, correct, explanation = match
                        
                        # Properly unescape the question text
                        question_text = question_text.replace('\\"', '"').replace('\\n', ' ').replace('\\t', ' ')
                        explanation = explanation.replace('\\"', '"').replace('\\n', ' ').replace('\\t', ' ')
                        
                        # Clean and parse options more carefully
                        # Remove outer quotes and split properly
                        options_str = options_str.strip()
                        
                        # Try different option parsing methods
                        options = []
                        
                        # Method 1: Split by quotes and commas
                        option_matches = re.findall(r'"([^"]*(?:\\"[^"]*)*)"', options_str)
                        if option_matches:
                            options = [opt.replace('\\"', '"') for opt in option_matches]
                        
                        # Method 2: If method 1 fails, try simpler split
                        if not options:
                            # Split by comma, then clean each option
                            raw_options = [opt.strip() for opt in options_str.split(',')]
                            for opt in raw_options:
                                # Remove surrounding quotes and clean
                                clean_opt = opt.strip(' "\'')
                                if clean_opt:
                                    options.append(clean_opt)
                        
                        # Ensure we have 4 options for multiple choice, 2 for true/false
                        if q_type == "multiple_choice" and len(options) < 4:
                            while len(options) < 4:
                                options.append(f"Option {len(options) + 1}")
                        elif q_type == "true_false" and len(options) < 2:
                            options = ["True", "False"]
                        
                        questions.append({
                            "question": question_text.strip(),
                            "type": q_type.strip(),
                            "options": options[:4] if q_type == "multiple_choice" else options[:2],
                            "correct_answer": correct.strip(),
                            "explanation": explanation.strip(),
                            "difficulty": difficulty
                        })
                    
                    if questions:
                        pass  # Continue with manual extraction questions
                
                # Pattern 2: If pattern 1 fails, try extracting individual components with better quote handling
                if not questions:
                    # Extract all questions with better regex that handles escaped quotes
                    question_texts = re.findall(r'"question":\s*"((?:[^"\\]|\\.)*)"', response_text)
                    question_types = re.findall(r'"type":\s*"([^"]+)"', response_text)
                    correct_answers = re.findall(r'"correct_answer":\s*"([^"]+)"', response_text)
                    explanations = re.findall(r'"explanation":\s*"((?:[^"\\]|\\.)*)"', response_text)
                    
                    # Clean up extracted questions by unescaping quotes
                    question_texts = [q.replace('\\"', '"').replace('\\n', ' ').strip() for q in question_texts]
                    explanations = [e.replace('\\"', '"').replace('\\n', ' ').strip() for e in explanations]
                    
                    # Extract options arrays separately
                    options_arrays = re.findall(r'"options":\s*\[[^\]]+\]', response_text)
                    
                    # Process each question
                    for i in range(min(len(question_texts), len(question_types), len(correct_answers))):
                        options = ["Option A", "Option B", "Option C", "Option D"]
                        
                        # Try to extract options for this question
                        if i < len(options_arrays):
                            options_text = options_arrays[i]
                            # Extract options from this array
                            extracted_options = re.findall(r'"([^"]+)"', options_text)
                            if len(extracted_options) >= 2:
                                options = extracted_options[:4] if question_types[i] == "multiple_choice" else extracted_options[:2]
                        
                        # Use true/false options for true_false questions
                        if question_types[i] == "true_false":
                            options = ["True", "False"]
                        
                        questions.append({
                            "question": question_texts[i],
                            "type": question_types[i],
                            "options": options,
                            "correct_answer": correct_answers[i] if i < len(correct_answers) else "A",
                            "explanation": explanations[i] if i < len(explanations) else "Based on document content",
                            "difficulty": difficulty
                        })
                    
                    if questions:
                        pass  # Continue with alternative manual extraction
                
                # Pattern 3: Last resort - extract questions with even more flexible matching
                if not questions:
                    
                    # Split response into potential question blocks
                    question_blocks = re.split(r'(?=\s*{|\s*"question")', response_text)
                    
                    for block in question_blocks:
                        if '"question"' in block and '"type"' in block:
                            try:
                                # Extract question with very flexible pattern
                                q_match = re.search(r'"question":\s*"([^"]*(?:"[^"]*"[^"]*)*)"', block, re.DOTALL)
                                t_match = re.search(r'"type":\s*"([^"]+)"', block)
                                c_match = re.search(r'"correct_answer":\s*"([^"]+)"', block)
                                
                                if q_match and t_match:
                                    question_text = q_match.group(1)
                                    # Clean up the question text
                                    question_text = re.sub(r'\s+', ' ', question_text).strip()
                                    question_text = question_text.replace('\\"', '"')
                                    
                                    # Skip if question is too short (likely incomplete)
                                    if len(question_text) < 10:
                                        continue
                                    
                                    # Extract options from this block
                                    options = ["Option A", "Option B", "Option C", "Option D"]
                                    options_match = re.search(r'"options":\s*\[([^\]]+)\]', block)
                                    if options_match:
                                        options_text = options_match.group(1)
                                        extracted_options = re.findall(r'"([^"]*(?:\\"[^"]*)*)"', options_text)
                                        if extracted_options:
                                            options = [opt.replace('\\"', '"') for opt in extracted_options]
                                    
                                    # Use true/false options for true_false questions
                                    if t_match.group(1) == "true_false":
                                        options = ["True", "False"]
                                    
                                    questions.append({
                                        "question": question_text,
                                        "type": t_match.group(1),
                                        "options": options[:4] if t_match.group(1) == "multiple_choice" else options[:2],
                                        "correct_answer": c_match.group(1) if c_match else "A",
                                        "explanation": "Based on document content",
                                        "difficulty": difficulty
                                    })
                                    
                            except Exception as e:
                                continue
                    
                    if questions:
                        pass  # Continue with flexible extraction
                        
            except Exception as e:
                pass  # Continue even if manual extraction fails
        
        if questions is None:
            # Final fallback: try to extract any useful text and create basic questions
            try:
                # Extract any text that looks like questions
                question_texts = re.findall(r'["\']?([^"\']*\?)["\']?', response_text)
                if question_texts:
                    questions = []
                    for i, q_text in enumerate(question_texts[:num_questions]):
                        questions.append({
                            "question": q_text.strip(),
                            "type": "multiple_choice",
                            "options": ["Option A", "Option B", "Option C", "Option D"],
                            "correct_answer": "A",
                            "difficulty": difficulty,
                            "explanation": "Generated from partial response"
                        })
            except Exception as e:
                pass  # Continue even if emergency fallback fails
        
        if questions is None:
            error_summary = "; ".join(parsing_errors) if parsing_errors else "Unknown parsing error"
            raise json.JSONDecodeError(f"All JSON parsing attempts failed: {error_summary}", response_text, 0)
        
        # Validate and format questions with completeness checks
        formatted_questions = []
        
        # Ensure questions is a list
        if not isinstance(questions, list):
            questions = [questions] if isinstance(questions, dict) else []
        
        for q in questions:
            if isinstance(q, dict) and 'question' in q and 'type' in q:
                # Clean and validate the question text
                question_text = str(q.get('question', '')).replace('\n', ' ').replace('\r', ' ').strip()
                # Unescape any remaining escaped quotes
                question_text = question_text.replace('\\"', '"').replace('\\/', '/')
                
                # Question completeness validation
                question_valid = True
                
                # Check if question is too short (likely incomplete)
                if len(question_text) < 15:
                    question_valid = False
                
                # Check if question ends properly (should end with ? or :)
                if question_valid and not question_text.endswith(('?', ':', '.')):
                    # Try to find if there's more text that got cut off
                    if not any(char in question_text[-10:] for char in ['?', '.', ':']):
                        # Add a question mark if it seems to be a question
                        if any(start in question_text.lower() for start in ['what', 'which', 'how', 'when', 'where', 'who', 'why']):
                            question_text += "?"
                        else:
                            question_text += "."
                
                # Check for truncated questions (common patterns)
                truncation_indicators = ['...', '..', 'according to the document, what', 'the text states that']
                if question_valid:
                    for indicator in truncation_indicators:
                        if indicator in question_text.lower() and len(question_text) < 50:
                            pass  # Continue processing even with potential truncation
                
                # Skip very obviously incomplete questions (more lenient criteria)
                if question_valid and (len(question_text.split()) < 3 or 
                                     question_text.strip() in ['', 'Question:', 'Q:']):
                    question_valid = False
                
                if question_valid:
                    explanation_text = str(q.get('explanation', '')).replace('\n', ' ').replace('\r', ' ').strip()
                    explanation_text = explanation_text.replace('\\"', '"').replace('\\/', '/')
                    
                    formatted_questions.append({
                        "question": question_text,
                        "type": q.get('type', 'multiple_choice'),
                        "options": q.get('options', []),
                        "correct_answer": q.get('correct_answer', ''),
                        "difficulty": q.get('difficulty', difficulty),
                        "explanation": explanation_text if explanation_text else "Based on document content"
                    })
        
        if not formatted_questions:
            raise Exception("No valid complete questions found in the response")
        
        # Ensure we have exactly the requested number of questions
        if len(formatted_questions) < num_questions:
            # Generate additional questions using fallback method
            additional_needed = num_questions - len(formatted_questions)
            fallback_questions = simulate_llm_question_generation(text, difficulty, additional_needed, regenerate)
            formatted_questions.extend(fallback_questions)
            
        return formatted_questions[:num_questions]  # Ensure we return exactly the requested number
        
    except json.JSONDecodeError as e:
        # Falling back to simulated questions
        return simulate_llm_question_generation(text, difficulty, num_questions, regenerate)
        
    except Exception as e:
        # Falling back to simulated question generation
        return simulate_llm_question_generation(text, difficulty, num_questions, regenerate)

def simulate_llm_question_generation(text, difficulty, num_questions, regenerate=False):
    """
    Highly improved fallback function that creates content-specific questions when LLM APIs are not available
    """
    
    # Add randomness for regeneration
    import random
    
    # Use different approaches for regeneration
    if regenerate:
        # Shuffle the order of processing and use different random seeds
        random.seed(random.randint(1, 10000))  # Different seed each time
    
    # Enhanced text analysis
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 30][:30]
    paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 50][:15]
    
    # Extract key information with better patterns
    # Find dates, years, percentages, monetary amounts
    dates = re.findall(r'\b(?:\d{1,2}/\d{1,2}/\d{4}|\d{4}|January|February|March|April|May|June|July|August|September|October|November|December)\b', text)
    percentages = re.findall(r'\b\d+(?:\.\d+)?%\b', text)
    monetary = re.findall(r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b', text)
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    
    # Extract proper nouns and important terms
    proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    capitalized_phrases = re.findall(r'\b[A-Z][a-zA-Z\s]{2,30}\b', text)
    
    # Extract technical terms and concepts (words ending in common suffixes)
    technical_terms = re.findall(r'\b\w*(?:tion|sion|ment|ness|ity|ism|ology|graphy|metry)\b', text, re.IGNORECASE)
    
    # Remove common words and get unique important terms
    stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'this', 'that', 'these', 'those', 'a', 'an'}
    
    key_terms = list(set([term for term in proper_nouns + technical_terms + capitalized_phrases 
                         if len(term) > 3 and term.lower() not in stopwords]))[:20]
    
    facts = dates + percentages + monetary + [num for num in numbers if len(num) <= 4]
    
    def extract_factual_content():
        """Extract specific facts from sentences"""
        factual_content = []
        
        for sentence in sentences:
            # Look for sentences with specific patterns
            if any(keyword in sentence.lower() for keyword in ['according to', 'states that', 'shows that', 'indicates', 'demonstrates', 'reveals']):
                factual_content.append(sentence)
            elif any(fact in sentence for fact in facts[:5]):
                factual_content.append(sentence)
            elif any(term in sentence for term in key_terms[:10]):
                factual_content.append(sentence)
                
        return factual_content[:10]
    
    factual_sentences = extract_factual_content()
    
    def create_multiple_choice_from_fact(fact_sentence, term_or_fact):
        """Create a multiple choice question from a factual sentence with relevant options"""
        # Create a more specific question based on the content
        if any(word in fact_sentence.lower() for word in ['date', 'year', 'when']) or any(d in fact_sentence for d in dates):
            question_stem = f"When was {term_or_fact} mentioned in the document?"
            # Create date-based options
            if dates:
                base_date = dates[0] if dates else "2023"
                if base_date.isdigit():
                    base_year = int(base_date)
                    options = [
                        f"In {base_date}",
                        f"In {base_year + 1}",
                        f"In {base_year - 1}",
                        "Date not specified"
                    ]
                else:
                    options = [
                        f"In {base_date}",
                        "In 2024",
                        "In 2022",
                        "Date not specified"
                    ]
            else:
                options = ["Recently mentioned", "Several years ago", "In the near future", "Time not specified"]
        elif any(word in fact_sentence.lower() for word in ['percent', '%', 'increase', 'decrease']) or percentages:
            question_stem = f"What percentage or numerical value is associated with {term_or_fact}?"
            if percentages:
                base_percent = percentages[0]
                # Extract numbers from percentage string
                numbers_in_percent = re.findall(r'\d+', base_percent)
                if numbers_in_percent:
                    base_num = int(numbers_in_percent[0])
                    options = [
                        base_percent,
                        f"{base_num + 10}%",
                        f"{base_num - 5}%",
                        "No percentage given"
                    ]
                else:
                    options = [base_percent, "15%", "25%", "No percentage given"]
            else:
                options = ["Significant increase", "Minor decrease", "No change reported", "Data not available"]
        else:
            question_stem = f"According to the document, what is stated about {term_or_fact}?"
            
            # Create contextual options based on other terms in the document
            other_terms = [t for t in key_terms if t != term_or_fact][:3]
            
            # Extract a meaningful portion from the actual sentence for the correct answer
            correct_info = "Mentioned in the document"
            if len(fact_sentence) > 30:
                words = fact_sentence.split()
                if len(words) > 6:
                    # Extract key information while keeping it under 80 chars
                    key_part = ' '.join(words[2:8])
                    if len(key_part) > 20:
                        correct_info = key_part[:75] + ("..." if len(key_part) > 75 else "")
            
            options = [
                correct_info,
                f"It relates to {other_terms[0] if other_terms else 'other concepts'} mentioned",
                f"Associated with {other_terms[1] if len(other_terms) > 1 else 'different topics'}",
                f"Connected to {other_terms[2] if len(other_terms) > 2 else 'various subjects'}"
            ]
        
        # Ensure options are not too similar
        options = list(dict.fromkeys(options))  # Remove duplicates while preserving order
        while len(options) < 4:
            options.append(f"Alternative information about {term_or_fact}")
        
        return {
            "question": question_stem,
            "type": "multiple_choice",
            "options": options[:4],
            "correct_answer": "A",
            "difficulty": difficulty,
            "explanation": f"The document specifically mentions this information about {term_or_fact}."
        }
    
    def create_true_false_from_content(content):
        """Create true/false questions from content"""
        if len(content) > 30:
            # Create a statement based on the content
            words = content.split()
            if len(words) > 8:
                statement = ' '.join(words[:15]) + ("..." if len(words) > 15 else "")
                return {
                    "question": f"True or False: {statement}",
                    "type": "true_false",
                    "options": ["True", "False"],
                    "correct_answer": "True",
                    "difficulty": difficulty,
                    "explanation": "This statement is directly supported by the document content."
                }
        return None
    
    # Generate content-specific questions with variety for regeneration
    questions = []
    
    # Add randomization for regeneration
    if regenerate:
        # Shuffle content sources for variety
        random.shuffle(factual_sentences)
        random.shuffle(key_terms)
        random.shuffle(sentences)
        
        # Use different ratios for variety
        mc_ratio = random.choice([0.5, 0.6, 0.7])  # Vary the mix
    else:
        mc_ratio = 0.6  # Default ratio
    
    # Generate multiple choice questions
    mc_count = max(1, int(num_questions * mc_ratio))
    for i in range(min(mc_count, len(factual_sentences))):
        if i < len(key_terms) and i < len(factual_sentences):
            question = create_multiple_choice_from_fact(factual_sentences[i], key_terms[i])
            questions.append(question)
    
    # Generate true/false questions from various content
    tf_count = num_questions - len(questions)
    content_sources = factual_sentences + sentences[:10]
    
    if regenerate:
        random.shuffle(content_sources)  # Different order for regeneration
    
    for i in range(min(tf_count, len(content_sources))):
        tf_question = create_true_false_from_content(content_sources[i])
        if tf_question:
            questions.append(tf_question)
    
    # Fill remaining slots with topic-based questions if needed
    while len(questions) < num_questions and key_terms:
        term_index = len(questions) % len(key_terms)
        if regenerate:
            # Use random terms instead of sequential for variety
            term_index = random.randint(0, len(key_terms) - 1)
        
        term = key_terms[term_index]
        
        # Vary the question templates for regeneration
        if regenerate:
            question_templates = [
                f"What does the document indicate about {term}?",
                f"According to the text, how is {term} described?",
                f"The document mentions {term} in what context?",
                f"What information is provided about {term}?",
                f"How does the document characterize {term}?"
            ]
            question_text = random.choice(question_templates)
        else:
            question_text = f"What does the document indicate about {term}?"
            
        # Create more relevant options for fallback questions
        other_terms = [t for t in key_terms if t != term][:3]
        
        # Generate contextual options based on document content
        if facts and len(facts) > 0:
            options = [
                f"Described with specific details and {facts[0] if facts else 'data'}",
                f"Mentioned alongside {other_terms[0] if other_terms else 'related concepts'}",
                f"Associated with {other_terms[1] if len(other_terms) > 1 else 'key topics'}",
                f"Referenced in context of {other_terms[2] if len(other_terms) > 2 else 'document themes'}"
            ]
        elif dates:
            options = [
                f"Discussed in relation to {dates[0] if dates else 'recent events'}",
                f"Connected to {other_terms[0] if other_terms else 'related topics'}",
                f"Mentioned with historical context",
                f"Referenced without specific timeframe"
            ]
        else:
            options = [
                f"Provides comprehensive information about {term}",
                f"Relates to {other_terms[0] if other_terms else 'other key concepts'}",
                f"Connected with {other_terms[1] if len(other_terms) > 1 else 'document themes'}",
                f"Briefly referenced in passing"
            ]
            
        fallback_question = {
            "question": question_text,
            "type": "multiple_choice",
            "options": options,
            "correct_answer": "A",
            "difficulty": difficulty,
            "explanation": f"The document contains relevant information about {term}."
        }
        questions.append(fallback_question)
    
    # Shuffle final questions for variety in regeneration
    if regenerate and len(questions) > 1:
        random.shuffle(questions)
    
    return questions[:num_questions]


def display_questions(questions):
    """Display generated questions with optional answers and explanations"""
    st.markdown('<div class="quiz-container">', unsafe_allow_html=True)
    
    # Add toggle for showing/hiding answers
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("📝 Generated Quiz Questions")
    with col2:
        st.session_state.show_answers = st.toggle(
            "Show Answers", 
            value=st.session_state.show_answers,
            help="Toggle to show or hide the correct answers"
        )
    
    for i, q in enumerate(questions, 1):
        st.markdown(f'<div class="question-block">', unsafe_allow_html=True)
        
        st.markdown(f"**Question {i}** ({q['difficulty']} - {q['type'].replace('_', ' ').title()})")
        st.markdown(f"**{q['question']}**")
        
        if q['type'] == 'multiple_choice':
            for j, option in enumerate(q['options']):
                # Create option label (A, B, C, D)
                option_letter = chr(65 + j)  # A, B, C, D
                
                # Check if this is the correct option using multiple methods
                is_correct = False
                if st.session_state.show_answers:
                    correct_answer = q.get('correct_answer', '').strip()
                    
                    # Method 1: Direct letter match (A, B, C, D)
                    if correct_answer == option_letter:
                        is_correct = True
                    
                    # Method 2: Check if it's the first option and correct_answer is "A" or "1" or "0"
                    elif j == 0 and correct_answer in ['A', '1', '0']:
                        is_correct = True
                    elif j == 1 and correct_answer in ['B', '2', '1']:
                        is_correct = True
                    elif j == 2 and correct_answer in ['C', '3', '2']:
                        is_correct = True
                    elif j == 3 and correct_answer in ['D', '4', '3']:
                        is_correct = True
                    
                    # Method 3: Check if correct_answer matches the option text
                    elif correct_answer.lower() == option.lower():
                        is_correct = True
                    
                    # Method 4: Check if correct_answer is contained in option
                    elif correct_answer and correct_answer.lower() in option.lower():
                        is_correct = True
                
                if is_correct:
                    # Highlight the correct answer with light green background and dark green text
                    st.markdown(f"<div style='background-color: #d4edda; color: #155724; padding: 8px; border-radius: 4px; margin: 2px 0; border-left: 4px solid #28a745; margin-bottom: 20px;'>"
                               f"<span style='color: #28a745; font-weight: bold;'>✓</span> {option_letter}. {option}"
                               f"</div>", unsafe_allow_html=True)
                else:
                    # Display regular option
                    st.markdown(f"  {option_letter}. {option}")
        else:  # true_false
            if st.session_state.show_answers:
                if q['correct_answer'] == "True":
                    st.markdown(f"<div style='background-color: #d4edda; color: #155724; padding: 8px; border-radius: 4px; margin: 2px 0; border-left: 4px solid #28a745; margin-bottom: 20px;'>"
                               f"<span style='color: #28a745; font-weight: bold;'>✓</span> True"
                               f"</div>", unsafe_allow_html=True)
                    st.markdown("  False")
                else:
                    st.markdown("  True")
                    st.markdown(f"<div style='background-color: #d4edda; color: #155724; padding: 8px; border-radius: 4px; margin: 2px 0; border-left: 4px solid #28a745; margin-bottom: 20px;'>"
                               f"<span style='color: #28a745; font-weight: bold;'>✓</span> False"
                               f"</div>", unsafe_allow_html=True)
            else:
                st.markdown("  True")
                st.markdown("  False")
        
        # Show explanation if available and answers are shown
        if st.session_state.show_answers and 'explanation' in q and q['explanation']:
            with st.expander("💡 Explanation"):
                st.write(q['explanation'])
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")
    
    st.markdown('</div>', unsafe_allow_html=True)

def export_questions_json(questions):
    """Export questions as JSON"""
    export_data = {
        "generated_at": datetime.now().isoformat(),
        "total_questions": len(questions),
        "questions": questions
    }
    return json.dumps(export_data, indent=2)

def export_questions_text(questions, show_answers=True):
    """Export questions as formatted text with optional answers"""
    text_output = f"Quiz Questions - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    text_output += "=" * 50 + "\n\n"
    
    for i, q in enumerate(questions, 1):
        text_output += f"Question {i} ({q['difficulty']} - {q['type'].replace('_', ' ').title()})\n"
        text_output += f"{q['question']}\n"
        
        if q['type'] == 'multiple_choice':
            for option in q['options']:
                if show_answers:
                    marker = "✓" if option.startswith(f"Option {q['correct_answer']}") else " "
                    text_output += f"{marker} {option}\n"
                else:
                    text_output += f"  {option}\n"
        else:
            if show_answers:
                text_output += f"{'✓' if q['correct_answer'] == 'True' else ' '} True\n"
                text_output += f"{'✓' if q['correct_answer'] == 'False' else ' '} False\n"
            else:
                text_output += f"  True\n"
                text_output += f"  False\n"
        
        if show_answers:
            text_output += f"Correct Answer: {q['correct_answer']}\n"
            if 'explanation' in q and q['explanation']:
                text_output += f"Explanation: {q['explanation']}\n"
        
        text_output += "-" * 30 + "\n\n"
    
    return text_output

# Main app
def main():
    # Clear cache button at the top
    if st.button("🔄 Clear Cache", help="Clear all cached data and restart if downloads aren't working"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.cache_data.clear()
        st.rerun()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📚 PDF Quiz Generator</h1>
        <p>Upload a PDF chapter and generate customized quiz questions for your students</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🎛️ Quiz Settings")
        
        # File upload
        st.subheader("📁 Upload PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Maximum file size: 200MB"
        )
        
        if uploaded_file is not None:
            # Check file size (200MB limit)
            if uploaded_file.size > 200 * 1024 * 1024:
                st.error("File size exceeds 200MB limit. Please upload a smaller file.")
            else:
                st.success(f"File uploaded: {uploaded_file.name}")
                
                # Extract text
                with st.spinner("Extracting text from PDF..."):
                    pdf_text = extract_text_from_pdf(uploaded_file)
                    if pdf_text:
                        st.session_state.pdf_text = pdf_text
                        st.success(f"Extracted {len(pdf_text.split())} words from PDF")
        
        # Quiz configuration
        if st.session_state.pdf_text:
            st.subheader("🤖 Gemini AI Settings")
            
            # Check if Gemini is available
            if llm_config.gemini_client:
                st.success("✅ Google Gemini AI is configured and ready")
                st.info(f"Using model: {llm_config.gemini_model}")
                st.session_state.llm_provider = "gemini"
            else:
                st.warning("⚠️ Gemini API key not configured. Using fallback question generation.")
                st.info("To use AI-powered question generation, add your Gemini API key to the .env file")
                st.session_state.llm_provider = "fallback"
            
            st.subheader("⚙️ Question Settings")
            
            # Difficulty selection
            difficulty = st.selectbox(
                "Question Difficulty",
                ["Easy", "Medium", "Hard"],
                index=1,
                help="Choose the complexity level of questions"
            )
            
            # Number of questions
            st.write("Number of Questions")
            preset_numbers = [10, 20, 30, 40, 50]
            
            # Radio buttons for preset numbers
            num_questions = st.radio(
                "Select preset:",
                preset_numbers,
                index=1,
                horizontal=True
            )
            
            # Generate button
            if st.button("🚀 Generate Quiz from PDF", type="primary"):
                with st.spinner(f"Generating {num_questions} {difficulty.lower()} questions using Gemini AI..."):
                    # Generate questions using Gemini or fallback
                    if st.session_state.llm_provider == "gemini":
                        questions = generate_llm_questions(
                            st.session_state.pdf_text, 
                            difficulty, 
                            num_questions,
                            "gemini"
                        )
                    else:
                        questions = simulate_llm_question_generation(
                            st.session_state.pdf_text, 
                            difficulty, 
                            num_questions
                        )
                    
                    if questions and len(questions) > 0:
                        st.session_state.generated_questions = questions
                        st.session_state.quiz_generated = True
                        st.success(f"Generated {len(questions)} questions successfully!")
                    else:
                        st.error("Failed to generate questions. Please try again.")
                        st.session_state.generated_questions = []
                        st.session_state.quiz_generated = False
            
            # Regenerate button
            if st.session_state.quiz_generated:
                if st.button("🔄 Regenerate Questions", type="secondary"):
                    with st.spinner("Regenerating questions using Gemini AI..."):
                        # Track regeneration attempts for stability
                        if 'regeneration_count' not in st.session_state:
                            st.session_state.regeneration_count = 0
                        st.session_state.regeneration_count += 1
                        
                        # Clear any cached download session to force new files
                        if 'download_session_id' in st.session_state:
                            del st.session_state.download_session_id
                        
                        # Show regeneration attempt info
                        if st.session_state.regeneration_count > 2:
                            st.info(f"💡 Regeneration attempt #{st.session_state.regeneration_count} - Using more stable generation for better JSON quality")
                            
                        # Regenerate questions using Gemini or fallback with regenerate=True
                        if st.session_state.llm_provider == "gemini":
                            questions = generate_llm_questions(
                                st.session_state.pdf_text, 
                                difficulty, 
                                num_questions,
                                "gemini",
                                regenerate=True  # This ensures variety
                            )
                        else:
                            questions = simulate_llm_question_generation(
                                st.session_state.pdf_text, 
                                difficulty, 
                                num_questions,
                                regenerate=True  # This ensures variety
                            )
                        
                        if questions and len(questions) > 0:
                            st.session_state.generated_questions = questions
                            st.success("Questions regenerated successfully!")
                            st.info("💡 New questions generated with different focus and variety!")
                        else:
                            st.error("Failed to regenerate questions. Please try again.")
                            st.session_state.generated_questions = []
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if not st.session_state.pdf_text:
            st.info("👈 Please upload a PDF file to get started")
            
            # Preview area
            st.subheader("📖 How it works")
            st.markdown("""
            1. **Upload PDF**: Choose your chapter or document (max 200MB)
            2. **Configure Quiz**: Select difficulty level and number of questions
            3. **Generate**: Gemini AI creates multiple-choice and true/false questions
            4. **Review**: Check generated questions with answer keys
            5. **Export**: Download questions in your preferred format
            """)
        
        elif not st.session_state.quiz_generated:
            st.subheader("📄 PDF Content Preview")
            with st.expander("View extracted text", expanded=False):
                st.text_area(
                    "Extracted Text",
                    st.session_state.pdf_text[:2000] + ("..." if len(st.session_state.pdf_text) > 2000 else ""),
                    height=300,
                    disabled=True
                )
            st.info("👈 Configure your quiz settings and click 'Generate Quiz from PDF'")
        
        else:
            # Display generated questions
            display_questions(st.session_state.generated_questions)
    
    with col2:
        if st.session_state.quiz_generated:
            st.subheader("📤 Export Options")
            
            # Quiz statistics
            st.metric("Total Questions", len(st.session_state.generated_questions))
            
            difficulty_counts = {}
            type_counts = {}
            for q in st.session_state.generated_questions:
                difficulty_counts[q['difficulty']] = difficulty_counts.get(q['difficulty'], 0) + 1
                type_counts[q['type']] = type_counts.get(q['type'], 0) + 1
            
            st.write("**Difficulty Distribution:**")
            for diff, count in difficulty_counts.items():
                st.write(f"• {diff}: {count}")
            
            st.write("**Question Types:**")
            for qtype, count in type_counts.items():
                st.write(f"• {qtype.replace('_', ' ').title()}: {count}")
            
            st.markdown("---")
            
            # Export buttons
            st.write("**Download Quiz:**")
            
            # Generate unique timestamp and session ID for this session
            if 'download_session_id' not in st.session_state:
                st.session_state.download_session_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:19]
            
            session_id = st.session_state.download_session_id
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Ensure questions exist before creating downloads
            if not st.session_state.generated_questions:
                st.error("No questions to download. Please generate questions first.")
                return
            
            try:
                # JSON export (always includes answers)
                json_data = export_questions_json(st.session_state.generated_questions)
                if json_data:
                    st.download_button(
                        label="📄 Download as JSON",
                        data=json_data,
                        file_name=f"quiz_{timestamp}.json",
                        mime="application/json",
                        help="JSON format includes all data including answers",
                        key=f"json_download_{session_id}",
                        use_container_width=True
                    )
                
                # Text export options
                col1, col2 = st.columns(2)
                
                with col1:
                    # Text export with answers
                    text_data_with_answers = export_questions_text(st.session_state.generated_questions, show_answers=True)
                    if text_data_with_answers:
                        st.download_button(
                            label="📝 Download with Answers",
                            data=text_data_with_answers,
                            file_name=f"quiz_with_answers_{timestamp}.txt",
                            mime="text/plain",
                            help="Text format with correct answers and explanations",
                            key=f"text_with_answers_{session_id}",
                            use_container_width=True
                        )
                
                with col2:
                    # Text export without answers
                    text_data_no_answers = export_questions_text(st.session_state.generated_questions, show_answers=False)
                    if text_data_no_answers:
                        st.download_button(
                            label="📄 Download Quiz Only",
                            data=text_data_no_answers,
                            file_name=f"quiz_questions_{timestamp}.txt",
                            mime="text/plain",
                            help="Text format with questions only (no answers)",
                            key=f"text_no_answers_{session_id}",
                            use_container_width=True
                        )
            
            except Exception as e:
                st.error(f"Error preparing downloads: {str(e)}")
                st.info("Please try regenerating the questions if download issues persist.")
                # Reset download session on error
                if 'download_session_id' in st.session_state:
                    del st.session_state.download_session_id
            
            # Clear quiz button
            st.markdown("---")
            if st.button("🗑️ Clear Quiz", type="secondary"):
                st.session_state.generated_questions = []
                st.session_state.quiz_generated = False
                st.rerun()
        
        else:
            st.info("Generate questions to see export options")

if __name__ == "__main__":
    main()