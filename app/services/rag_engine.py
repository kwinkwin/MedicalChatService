import pickle
import os
import json
import re
import time
import numpy as np
import faiss
import google.generativeai as genai
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from neo4j import GraphDatabase
from huggingface_hub import InferenceClient
from google.api_core import retry, exceptions
import logging
from app.config import settings

logger = logging.getLogger(__name__)

# --- 1. PROMPTS (Copy từ Notebook) ---
PROMPT_LLM_CANONICAL = (
    "Bạn là mô-đun Entity Linking & Canonicalization cho hệ thống Y tế.\n"
    "Nhiệm vụ: Đọc câu hỏi và danh sách các thực thể (candidates) được tìm thấy từ Database. "
    "Hãy chọn lọc và sắp xếp lại các thực thể phù hợp nhất để trả lời câu hỏi.\n\n"
    
    "QUAN TRỌNG: Hãy chú ý đến trường 'label' (Loại thực thể) để đảm bảo đúng ngữ cảnh:\n"
    "- Nếu người dùng hỏi về 'triệu chứng', ưu tiên chọn candidate có label 'TrieuChung'.\n"
    "- Nếu người dùng hỏi về 'thuốc trị bệnh', ưu tiên chọn candidate có label 'Thuoc' hoặc 'Benh'.\n"
    "- Loại bỏ các candidate không liên quan hoặc trùng lặp về ý nghĩa.\n\n"
    
    "Input:\n"
    "User Question: <QUESTION>\n"
    "Raw Candidates (JSON): <CANDIDATES>\n\n"
    
    "Output:\n"
    "Trả về một JSON *duy nhất* chứa danh sách các thực thể đã chọn lọc (giữ nguyên cấu trúc name, label, id). "
    "Định dạng bắt buộc:\n"
    "{\n"
    "  \"selected_entities\": [\n"
    "    {\"name\": \"Tên thực thể\", \"label\": \"Nhãn\", \"id\": \"...\", \"reason\": \"Lý do chọn ngắn gọn\"}\n"
    "  ]\n"
    "}\n\n"
    "Quy định: Không thêm văn bản giải thích ngoài JSON. Nếu không tìm thấy gì phù hợp, trả về mảng rỗng."
)

# PROMPT_LLM_CYPHER = (
#     "Bạn là chuyên gia Neo4j (Graph Database) và Y tế.\n"
#     "Nhiệm vụ: Phân tích câu hỏi và sinh Cypher Query tối ưu nhất để lấy dữ liệu từ Knowledge Graph.\n\n"
    
#     "Input:\n"
#     "1. User Question: <<<QUESTION>>>\n"
#     "2. Selected Entities: <<<CANDIDATES>>> (Các thực thể đã được nhận diện)\n"
#     "3. Schema: <<<SCHEMA>>>\n\n"
    
#     "CHIẾN THUẬT QUERY (Hãy chọn 1 trong 3 dựa trên ý định người dùng):\n\n"
    
#     "--- CHIẾN THUẬT 1: TRA CỨU TRỰC TIẾP (Direct Lookup) ---\n"
#     "Áp dụng khi: Người dùng hỏi về thuộc tính hoặc quan hệ của 1 thực thể cụ thể.\n"
#     "Ví dụ: 'Biến chứng của bệnh X?', 'Thuốc Y trị bệnh gì?'\n"
#     "Pattern: MATCH (n:Label {ten: 'Tên'})-[r]->(target) RETURN ...\n\n"
    
#     "--- CHIẾN THUẬT 2: TÌM KIẾM TỔ HỢP / CHẨN ĐOÁN (Intersection/Scoring) ---\n"
#     "Áp dụng khi: Người dùng đưa ra danh sách các yếu tố (triệu chứng, nguyên nhân...) và hỏi về đối tượng chung.\n"
#     "Ví dụ: 'Sốt, ho, đau đầu là bệnh gì?', 'Thuốc nào rẻ mà tốt?'\n"
#     "Pattern:\n"
#     "MATCH (target)-[:REL]->(source)\n"
#     "WHERE toLower(source.ten) CONTAINS 'yếu tố A' OR toLower(source.ten) CONTAINS 'yếu tố B' ...\n"
#     "WITH target, count(distinct source) AS matches, collect(source.ten) AS evidences\n"
#     "RETURN target.ten, matches, evidences ORDER BY matches DESC LIMIT 5\n\n"
    
#     "--- CHIẾN THUẬT 3: TÌM MỐI LIÊN HỆ (Path Finding) ---\n"
#     "Áp dụng khi: Người dùng hỏi về quan hệ giữa 2 thực thể cụ thể.\n"
#     "Ví dụ: 'Tiểu đường ăn sầu riêng được không?', 'Thuốc A uống chung thuốc B được không?'\n"
#     "Pattern:\n"
#     "MATCH (a {ten: 'A'}), (b {ten: 'B'})\n"
#     "MATCH p = shortestPath((a)-[*]-(b)) \n"
#     "RETURN p\n\n"
    
#     "QUY ĐỊNH BẮT BUỘC:\n"
#     "- CHỈ dùng Label và Relationship Type có trong Schema.\n"
#     "- Luôn dùng `toLower(...) CONTAINS ...` để tìm tên node (để tránh lỗi case-sensitive).\n"
#     "- Trả về đủ thông tin (tên, mô tả, quan hệ) để bot trả lời.\n\n"
    
#     "Output:\n"
#     "CHỈ trả về JSON: {\"cypher\": \"...\"}"
# )

PROMPT_LLM_CYPHER = (
    "Bạn là chuyên gia Neo4j và Phân tích Ý định Y khoa (Medical Intent Analyst).\n"
    "Nhiệm vụ: Phân tích Input người dùng và danh sách Thực thể đã chọn (Selected Entities) và sinh câu lệnh Cypher phù hợp nhất để lấy dữ liệu từ Knowledge Graph.\n\n"
    
    "Input:\n"
    "1. User Input: <<<QUESTION>>>\n"
    "2. Selected Entities: <<<CANDIDATES>>>\n"
    "3. Schema: <<<SCHEMA>>>\n\n"
    
    "QUY TRÌNH SUY LUẬN (CHAIN OF THOUGHT):\n"
    "Bước 1: Xác định Ý ĐỊNH (Intent) của người dùng (kể cả khi họ không dùng câu hỏi):\n"
    "   - Nhóm KỂ BỆNH/MÔ TẢ: 'Tôi bị đau đầu, mệt mỏi', 'Có triệu chứng sốt, ho'. -> Cần tìm nguyên nhân/bệnh (Chiến thuật 2).\n"
    "   - Nhóm RA LỆNH/TÌM HIỂU: 'Nói về bệnh X', 'Thông tin thuốc Y', 'Viêm gan B'. -> Cần xem thông tin tổng quan và hàng xóm của node (Chiến thuật 1).\n"
    "   - Nhóm KIỂM TRA QUAN HỆ: 'A dùng chung với B được không?', 'Mối liên hệ giữa X và Y'. -> Cần tìm đường đi (Chiến thuật 3).\n\n"
    
    "Bước 2: Chọn CHIẾN THUẬT QUERY tương ứng:\n\n"
    
    "--- CHIẾN THUẬT 1: KHÁM PHÁ THỰC THỂ (Entity Exploration) ---\n"
    "Áp dụng cho: Tra cứu thông tin, định nghĩa, hoặc những câu lệnh có nghĩa tương tự với 'nói rõ hơn về...', 'chi tiết về...'.\n"
    "Mục tiêu: Lấy thông tin node đó VÀ các node xung quanh (1-hop) để có ngữ cảnh.\n"
    "Pattern:\n"
    "MATCH (n:Label)-[r]-(m)\n"
    "WHERE toLower(n.ten) CONTAINS 'tên entity'\n"
    "RETURN n, r, m\n\n"
    
    "--- CHIẾN THUẬT 2: CHẨN ĐOÁN ĐA YẾU TỐ (Multi-Factor Diagnosis) ---\n"
    "Áp dụng cho: Người dùng liệt kê triệu chứng (kể khổ), yếu tố nguy cơ, hoặc hỏi 'là bệnh gì?', hoặc hỏi về đối tượng chung.\n"
    "Mục tiêu: Tìm node trung tâm (thường là Bệnh) có kết nối với nhiều entity trong danh sách nhất.\n"
    "Pattern:\n"
    "MATCH (target:Benh)-[:REL]->(source)\n"
    "WHERE toLower(source.ten) CONTAINS 'entity 1' OR toLower(source.ten) CONTAINS 'entity 2' ...\n"
    "WITH target, count(distinct source) AS matches, collect(distinct source.ten) AS evidences\n"
    "RETURN target.ten, target.mo_ta, matches, evidences ORDER BY matches DESC LIMIT 5\n\n"
    
    "--- CHIẾN THUẬT 3: PHÂN TÍCH LIÊN KẾT (Path Analysis) ---\n"
    "Áp dụng cho: Câu hỏi về tương tác thuốc, kiêng kỵ, hoặc quan hệ giữa 2 thực thể cụ thể.\n"
    "Pattern:\n"
    "MATCH (a), (b)\n"
    "WHERE toLower(a.ten) CONTAINS 'entity 1' AND toLower(b.ten) CONTAINS 'entity 2'\n"
    "MATCH p = shortestPath((a)-[*]-(b))\n"
    "RETURN p\n\n"
    
    "Bước 3: Sinh Cypher (Final Output).\n"
    "Yêu cầu:\n"
    "- Chỉ được dùng đúng Label/RelType trong Schema.\n"
    "- Luôn dùng `toLower(...) CONTAINS ...` để tìm tên node (để tránh lỗi case-sensitive).\n"
    "- Trả về đủ thông tin (tên, mô tả, quan hệ) để bot trả lời.\n"
    "- Khi RETURN node Bệnh, BẮT BUỘC phải lấy kèm thuộc tính 'nguon' (nếu có)."
    "- Output là JSON duy nhất: {\"cypher\": \"...\"}"
)

PROMPT_LLM_ANSWER = """
Bạn là 'Hệ thống Hỗ trợ Quyết định Lâm sàng' (CDSS) chuyên sâu.
Nhiệm vụ: Phân tích dữ liệu Knowledge Graph (KG) và tạo báo cáo tóm tắt lâm sàng súc tích, hỗ trợ bác sĩ ra quyết định nhanh chóng.

Input:
- Câu hỏi: <<<QUESTION>>>
- Dữ liệu KG: <<<FACTS>>>

Tư duy xử lý (Chain of Thought):
1. LỌC: Chỉ chọn các facts trong KG trực tiếp trả lời cho "Câu hỏi". Bỏ qua dữ liệu nhiễu.
2. TỔNG HỢP: Viết lại thành văn phong y khoa tự nhiên, không liệt kê máy móc dạng A-relation-B.

Quy tắc Output (Nghiêm ngặt):
- **Conciseness (Súc tích)**: Đi thẳng vào vấn đề. Sử dụng gạch đầu dòng ngắn gọn.
- **Evidence-Only**: Chỉ dùng thông tin từ Dữ liệu KG. Nếu thiếu thông tin quan trọng, hãy nói rõ là dữ liệu chưa cập nhật.
- **No Fluff**: Bỏ qua các câu dẫn dắt rườm rà (ví dụ: "Dựa trên dữ liệu được cung cấp...").

Cấu trúc báo cáo:
### 1. Kết luận Lâm sàng (Direct Answer)
- Trả lời trực diện câu hỏi dựa trên bằng chứng mạnh nhất.

### 2. Phân tích Chi tiết (Grouped Evidence)
*Thay vì liệt kê từng quan hệ, hãy nhóm thông tin:*
- **Về Bệnh học/Triệu chứng**: [Tổng hợp các node liên quan]
- **Về Điều trị/Thuốc**: [Tổng hợp các node liên quan]
- **Cơ chế/Lý do (Nếu có)**: [Giải thích ngắn gọn mối liên kết giữa các node]

### 3. Cảnh báo (Alerts)
- Nêu bật các chống chỉ định hoặc rủi ro tìm thấy (Nếu không có, bỏ qua mục này).
- Thông báo rằng các thông tin từ hệ thống Knowledge Graph mang tính tham khảo. Vui lòng tham vấn bác sĩ chuyên khoa.

### 4. Nguồn tham khảo (References)
- Kiểm tra trong dữ liệu KG, nếu node có thuộc tính `nguon` (thường là danh sách link url), hãy trích xuất và hiển thị dưới dạng link Markdown.
- Định dạng: "- [Xem chi tiết tại Nhà thuốc Long Châu](url)" hoặc "- [Nguồn tham khảo](url)".
- Chỉ hiển thị nếu có link thực sự.

Output Rules:
- Nếu Dữ liệu KG rỗng hoặc không liên quan: "Hệ thống tri thức hiện tại của tôi chỉ tập trung vào các chủ đề y tế phổ biến, nên tôi chưa đủ thông tin để trả lời câu hỏi này. Bạn có câu hỏi nào khác liên quan đến sức khỏe không?"
- Định dạng Markdown.
"""

PROMPT_QUERY_REWRITE = """
Bạn là mô-đun 'Coreference Resolution' (Giải quyết đồng tham chiếu) cho AI.
Nhiệm vụ: Viết lại câu hỏi mới nhất của người dùng sao cho nó ĐẦY ĐỦ NGỮ NGHĨA, dựa trên Lịch sử hội thoại.

Input:
- Lịch sử chat:
<<<HISTORY>>>
- Câu hỏi hiện tại: <<<CURRENT_QUESTION>>>

Quy tắc:
1. Thay thế các đại từ (nó, bệnh đó, thuốc này...) bằng tên thực thể cụ thể được nhắc đến trước đó.
2. Nếu câu hỏi đã rõ ràng hoặc không liên quan đến lịch sử, hãy GIỮ NGUYÊN.
3. KHÔNG trả lời câu hỏi, chỉ viết lại nó.
4. Output chỉ là một dòng văn bản duy nhất (câu hỏi đã viết lại).

Ví dụ:
History: "User: Triệu chứng sốt xuất huyết? AI: Sốt cao, đau đầu..."
Current: "Cách chữa nó?"
Output: "Cách chữa bệnh sốt xuất huyết?"
"""

# --- 2. CONFIGS & UTILS ---
DISALLOWED_RE = re.compile(r"(?i)\b(DELETE|REMOVE|DETACH|CREATE|MERGE|SET|DROP)\b")
HF_EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"
TOP_K = 100
MAX_CYPHER_RETRIES = 3

@dataclass
class NodeRecord:
    node_id: str
    label: str
    name: str
    text: str
    properties: Dict[str,Any]

class MedicalGraphRAG:
    INDEX_FILE = "vector_store.faiss"
    META_FILE = "metadata.pkl"

    def __init__(self):
        # Setup Connections
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI, 
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )
        
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        # Lưu ý: Nếu key free tier quá tải với flash 2.0, hãy đổi thành "gemini-1.5-flash"
        self.gemini_model = genai.GenerativeModel("gemini-2.0-flash") 

        self.hf_client = InferenceClient(
            token=settings.HF_API_KEY,
            provider="hf-inference"
        )
        
        # In-memory Vector Store
        self.index = None
        self.metadatas = []
        self.is_ready = False

    def close(self):
        if self.driver:
            self.driver.close()

    # --- INTERNAL HELPERS ---
    def _embed_texts(self, texts: List[str], batch_size: int = 50) -> np.ndarray:
        """
        Gửi request theo từng batch nhỏ để tránh lỗi 504 Timeout.
        """
        all_embeddings = []
        total = len(texts)
        
        logger.info(f"Bắt đầu embedding {total} văn bản (Batch size: {batch_size})...")
        print(f"Bắt đầu embedding {total} văn bản (Batch size: {batch_size})...")

        for i in range(0, total, batch_size):
            # Cắt lấy 1 nhóm 50 câu
            batch_texts = texts[i : i + batch_size]
            
            try:
                # Gọi API chỉ cho 50 câu này
                outputs = self.hf_client.feature_extraction(
                    model=HF_EMBEDDING_MODEL,
                    text=batch_texts
                )
                
                # Chuyển kết quả về numpy
                batch_result = np.array(outputs)
                
                # Xử lý mean-pooling nếu kết quả trả về là 3 chiều (Batch, Token, Vector)
                if batch_result.ndim == 3:
                    batch_result = np.mean(batch_result, axis=1)
                    
                all_embeddings.append(batch_result)
                
            except Exception as e:
                logger.warning(f"Lỗi tại batch {i}-{i+batch_size}: {e}. Đang thử lại...")
                # print(f"Lỗi tại batch {i}-{i+batch_size}: {e}")
                # Nếu lỗi, thử đợi 2s rồi chạy lại batch này (cơ chế retry đơn giản)
                time.sleep(2)
                try:
                    outputs = self.hf_client.feature_extraction(model=HF_EMBEDDING_MODEL, text=batch_texts)
                    batch_result = np.array(outputs)
                    if batch_result.ndim == 3:
                        batch_result = np.mean(batch_result, axis=1)
                    all_embeddings.append(batch_result)
                except Exception as final_e:
                    # raise e # Nếu thử lại vẫn lỗi thì dừng
                    logger.error(f"Thử lại thất bại batch {i}: {final_e}") # Error nếu chết hẳn
                    raise final_e
            
            # In tiến trình
            if (i // batch_size) % 5 == 0:
                # print(f"Đã xử lý {min(i + batch_size, total)}/{total} câu...")
                logger.info(f"Tiến độ embedding: {min(i + batch_size, total)}/{total}")

        # Nối lại thành 1 mảng lớn (4017, 768)
        return np.concatenate(all_embeddings, axis=0)
        
    def _get_graph_schema(self, driver) -> str:
        """
        Truy vấn Neo4j để lấy schema thực tế của đồ thị.
        Trả về chuỗi mô tả dạng: (:LabelA)-[:REL_TYPE]->(:LabelB)
        """
        # Query này lấy tất cả các cặp quan hệ đang tồn tại trong DB
        query = """
        MATCH (a)-[r]->(b)
        WITH labels(a) AS source_labels, type(r) AS rel_type, labels(b) AS target_labels
        UNWIND source_labels AS sl
        UNWIND target_labels AS tl
        RETURN DISTINCT sl AS source, rel_type, tl AS target
        ORDER BY source, rel_type
        """
        
        schema_lines = []
        try:
            with driver.session() as session:
                result = session.run(query)
                for record in result:
                    line = f"(:{record['source']})-[:{record['rel_type']}]->(:{record['target']})"
                    schema_lines.append(line)
        except Exception as e:
            # print(f"Error fetching schema: {e}")
            logger.error(f"Error fetching schema: {e}")
            return "Không thể lấy schema. Hãy giả định các quan hệ y tế phổ biến."

        # Lấy thêm thông tin về properties của các node quan trọng (tuỳ chọn nhưng nên có)
        # Ở đây tôi thêm text mẫu để LLM biết các trường quan trọng
        schema_text = "Graph Schema Patterns:\n" + "\n".join(schema_lines)
        schema_text += "\n\nNode Properties Assumptions:\n- Hầu hết các Node đều có thuộc tính: 'ten' (tên hiển thị), 'mo_ta' (mô tả chi tiết), 'ma' (mã định danh)."
        
        return schema_text

    def _run_cypher(self, cypher: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        params = params or {}
        with self.driver.session() as session:
            result = session.run(cypher, **params)
            return [rec.data() for rec in result]

    def _safe_parse_json(self, text: str) -> Dict[str, Any]:
        if not isinstance(text, str): return None
        cleaned = re.sub(r"```[a-zA-Z]*\n", "", text)
        cleaned = cleaned.replace("```", "")
        m = re.search(r"\{[\s\S]*\}", cleaned)
        if not m: return None
        js = m.group(0)
        try:
            return json.loads(js)
        except:
            try:
                js2 = js.replace("'", '"')
                return json.loads(js2)
            except:
                return None

    def _validate_cypher(self, cypher: str):
        if not cypher or not isinstance(cypher, str):
            raise ValueError("Empty cypher")
        if DISALLOWED_RE.search(cypher):
            raise ValueError("Cypher contains disallowed keywords")
        if "RETURN" not in cypher.upper():
            raise ValueError("Cypher must contain RETURN clause")
        return True
    
    @retry.Retry(predicate=retry.if_exception_type(exceptions.ResourceExhausted))
    def _call_gemini(self, prompt: str, model: str = "gemini-2.0-flash", max_output_tokens: int = 1024, temperature: float = 0.0) -> str:
        """
        Call Gemini with a single plain-string prompt (no system role).
        Returns text content.
        """
        # generate_content may accept 'prompt' as plain string depending on SDK version
        # Use a simple wrapper and try common attributes for response extraction
        resp = genai.GenerativeModel(model).generate_content(
            prompt,
            generation_config={
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
            },
        )
        return resp.text
    
    def _format_history(self, history: List[dict], limit: int = 6) -> str:
        """
        Chuyển đổi List[Message] thành chuỗi văn bản để đưa vào Prompt.
        Chỉ lấy 'limit' tin nhắn gần nhất để tiết kiệm token.
        """
        if not history:
            return ""
        
        # Lấy N tin nhắn cuối cùng
        recent_msgs = history[-limit:]
        formatted = []
        for msg in recent_msgs:
            # Xử lý linh hoạt: msg có thể là dict hoặc object Pydantic (như bạn định nghĩa)
            role = getattr(msg, 'role', None) or msg.get('role', 'user')
            content = getattr(msg, 'content', None) or msg.get('content', '')
            
            role_name = "User" if role == "user" else "ai"
            formatted.append(f"{role_name}: {content}")
            
        return "\n".join(formatted)

    def _rewrite_question(self, user_question: str, history_str: str) -> str:
        """
        Gọi LLM để viết lại câu hỏi dựa trên ngữ cảnh.
        """
        if not history_str:
            return user_question # Không có lịch sử thì không cần viết lại
            
        prompt = PROMPT_QUERY_REWRITE \
            .replace("<<<HISTORY>>>", history_str) \
            .replace("<<<CURRENT_QUESTION>>>", user_question)
            
        try:
            # Dùng model rẻ/nhanh nhất để rewrite (Flash 2.0 rất tốt việc này)
            rewritten = self._call_gemini(prompt, model="gemini-2.0-flash", max_output_tokens=256)
            return rewritten.strip()
        except Exception as e:
            logger.error(f"Error rewriting question: {e}")
            return user_question # Fallback về câu gốc nếu lỗi

    # --- CORE FUNCTIONS ---

    def load_data_and_build_index(self):
        # --- CÁCH 1: THỬ LOAD TỪ FILE CACHE TRƯỚC (SIÊU NHANH) ---
        if os.path.exists(self.INDEX_FILE) and os.path.exists(self.META_FILE):
            print(f">>> Found cached index files. Loading from disk...")
            logger.info(f"Found cached index files. Loading from disk...")
            try:
                # Load FAISS Index
                self.index = faiss.read_index(self.INDEX_FILE)
                
                # Load Metadata
                with open(self.META_FILE, "rb") as f:
                    self.metadatas = pickle.load(f)
                
                print(f">>> Loaded successfully! {self.index.ntotal} vectors ready in < 1s.")
                logger.info(f"Loaded successfully! {self.index.ntotal} vectors ready.")
                self.is_ready = True
                return
            except Exception as e:
                print(f">>> Cache corrupted ({e}). Re-building from scratch...")
                logger.error(f"Cache corrupted ({e}). Re-building from scratch...")
        
        # --- CÁCH 2: NẾU KHÔNG CÓ CACHE THÌ TẢI TỪ NEO4J (CHẠY LẦN ĐẦU) ---
        print(">>> Cache miss. Starting to load nodes from Neo4j...")
        logger.info("Cache miss. Starting to load nodes from Neo4j...")
        q = """
        MATCH (n)
        WHERE n.ten IS NOT NULL OR n.name IS NOT NULL
        RETURN id(n) AS id, labels(n) AS labels, coalesce(n.ten, n.name) AS ten, properties(n) AS props 
        """
        # Lưu ý: Tôi đã tăng LIMIT lên 5000 hoặc bỏ hẳn LIMIT nếu muốn lấy hết
        
        try:
            with self.driver.session() as session:
                res = session.run(q)
                nodes = []
                for r in res:
                    nid = str(r["id"])
                    labels = r["labels"]
                    label = labels[0] if labels else "Entity"
                    name = r["ten"] or "NoName"
                    props = dict(r["props"]) if r["props"] else {}
                    
                    desc_parts = []
                    for k in ["mo_ta", "mo_ta_ngan", "desc", "description"]:
                        if k in props and props[k]:
                            desc_parts.append(str(props[k]))
                    
                    text = name + (". " + " ".join(desc_parts) if desc_parts else "")
                    nodes.append(NodeRecord(node_id=nid, label=label, name=name, text=text, properties=props))
            
            if not nodes:
                print(">>> No nodes found in Neo4j.")
                logger.warning("No nodes found in Neo4j.")
                self.is_ready = True 
                return

            print(f">>> Embedding {len(nodes)} nodes ...")
            logger.info(f"Embedding {len(nodes)} nodes ...")
            texts = [n.text for n in nodes]
            
            # # Batch embedding
            # batch_size = 32
            # all_vecs = []
            # for i in range(0, len(texts), batch_size):
            #     batch = texts[i:i+batch_size]
            #     vecs = self._embed_texts(batch)
            #     all_vecs.append(vecs)
            #     # In log tiến độ để đỡ sốt ruột
            #     if i % 100 == 0:
            #         print(f"   Processed {i}/{len(texts)}...")

            # if all_vecs:
            #     final_vecs = np.concatenate(all_vecs, axis=0)
            #     norms = np.linalg.norm(final_vecs, axis=1, keepdims=True)
            #     norms[norms == 0] = 1.0
            #     final_vecs = final_vecs / norms

            #     self.index = faiss.IndexFlatIP(final_vecs.shape[1])
            #     self.index.add(final_vecs.astype('float32'))
                
            #     self.metadatas = [
            #         {
            #             "node_id": n.node_id, 
            #             "label": n.label, 
            #             "name": n.name, 
            #             "text": n.text, 
            #             "properties": n.properties
            #         }
            #         for n in nodes
            #     ]
            
            final_vecs = self._embed_texts(texts, batch_size=50)
            
            norms = np.linalg.norm(final_vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            final_vecs = final_vecs / norms

            # Build FAISS Index
            self.index = faiss.IndexFlatIP(final_vecs.shape[1])
            self.index.add(final_vecs.astype('float32'))
            
            self.metadatas = [
                {
                    "node_id": n.node_id, 
                    "label": n.label, 
                    "name": n.name, 
                    "text": n.text, 
                    "properties": n.properties
                }
                for n in nodes
            ]
                
                # --- LƯU CACHE XUỐNG ĐĨA ---
            print(">>> Saving index to disk for next run...")
            logger.info("Saving index to disk for next run...")
            faiss.write_index(self.index, self.INDEX_FILE)
            with open(self.META_FILE, "wb") as f:
                pickle.dump(self.metadatas, f)
            
            print(f">>> FAISS Index built & saved with {self.index.ntotal} vectors.")
            logger.info(f"FAISS Index built & saved with {self.index.ntotal} vectors.")
            self.is_ready = True
                
        except Exception as e:
            print(f">>> CRITICAL ERROR loading data: {e}")
            logger.critical(f"CRITICAL ERROR loading data: {e}", exc_info=True)

    async def process_question(self, user_question: str, history: List[dict]) -> Dict[str, Any]:
        """
        Main pipeline: Retrieve -> Canonicalize -> Cypher Gen -> Execute -> Answer
        """
        # history
        print(f"++++++++++++++++++++++++++++++++++{len(history)}")
        try:
            print(f">>> START processing question: {user_question}")
            logger.info(f"START processing question: {user_question}")
            
            if not self.is_ready:
                return {"answer": "Hệ thống đang khởi động hoặc gặp lỗi kết nối CSDL."}
            
            history_str = self._format_history(history)
            standalone_question = user_question
            
            if history_str:
                print(">>> Rewriting question based on history...")
                standalone_question = self._rewrite_question(user_question, history_str)
                print(f">>> Rewritten Question: {standalone_question}")
                logger.info(f"Rewritten Question: {standalone_question}")
                
            # 1. Retrieve Candidates
            # qvec = self._embed_texts([user_question])[0]
            qvec = self._embed_texts([standalone_question])[0]
            qvec = qvec / (np.linalg.norm(qvec) + 1e-12)
            
            # Search FAISS
            D, I = self.index.search(np.array([qvec], dtype=np.float32), TOP_K)
            
            retrieved_candidates = []
            for score, idx in zip(D[0], I[0]):
                if 0 <= idx < len(self.metadatas):
                    meta = self.metadatas[idx]
                    retrieved_candidates.append((meta, float(score)))
            
            # candidate_names = [m['name'] for m in retrieved_candidates]
            rich_candidates = []
            for meta, score in retrieved_candidates:
                rich_candidates.append({
                    "name": meta['name'],
                    "label": meta['label'],
                    "id": meta['node_id']
                })
                
            logger.info(f"Retrieved {len(rich_candidates)} candidates via Vector Search")

            # 2. Canonicalize (Optional step to re-rank via LLM)
            # Giảm số lượng candidate truyền vào prompt để tiết kiệm token
            # top_candidates_str = "\n".join(candidate_names[:50]) 
            # prompt_can = PROMPT_LLM_CANONICAL.replace("<QUESTION>", user_question) + "\nCandidates:\n" + top_candidates_str
            
            prompt_can = PROMPT_LLM_CANONICAL \
            .replace("<QUESTION>", standalone_question) \
            .replace("<CANDIDATES>", json.dumps(rich_candidates[:100], ensure_ascii=False))

            canon_raw = self._call_gemini(prompt_can, model="gemini-2.0-flash", max_output_tokens=1024)
            canon_json = self._safe_parse_json(canon_raw)
            
            # try:
            #     canon_raw = self.gemini_model.generate_content(prompt_can).text
            #     canon_json = self._safe_parse_json(canon_raw)
            #     if canon_json and isinstance(canon_json.get("candidates"), list):
            #         ordered_candidates = canon_json["candidates"]
            #     else:
            #         ordered_candidates = candidate_names
            # except Exception as e:
            #     print(f"Canonicalize error: {e}")
            #     ordered_candidates = candidate_names
            
            selected_entities = []
            # Ưu tiên lấy từ JSON trả về, nếu lỗi thì lấy top 3 từ vector search
            if canon_json and "selected_entities" in canon_json:
                selected_entities = canon_json["selected_entities"]
            else:
                selected_entities = rich_candidates[:3]
                
            logger.info(f"Canonicalized to: {json.dumps(selected_entities[:2], ensure_ascii=False)}...")

            # if debug:
            #     print(f"Selected Entities: {json.dumps(selected_entities, ensure_ascii=False)}")

            # 3. Generate Cypher Loop
            # executed_cypher = None
            # records = []
            # last_error = None
            
            # # Chỉ thử 3 candidate tốt nhất để tiết kiệm thời gian
            # for attempt_idx, cand in enumerate(ordered_candidates[:MAX_CYPHER_RETRIES]):
            #     prompt_cy = PROMPT_LLM_CYPHER.replace("<<<QUESTION>>>", user_question).replace("<<<CANDIDATES>>>", json.dumps([cand]))
                
            #     try:
            #         cy_resp = self.gemini_model.generate_content(prompt_cy)
            #         parsed = self._safe_parse_json(cy_resp.text)
                    
            #         if parsed and "cypher" in parsed:
            #             cypher = parsed["cypher"]
            #         else:
            #             # Fallback cypher đơn giản
            #             cand_esc = cand.replace("'", "\\'")
            #             cypher = f"MATCH (b) WHERE toLower(b.ten) CONTAINS '{cand_esc.lower()}' RETURN b.ten AS name, properties(b) AS props LIMIT 5"

            #         # Replace placeholder if any (from prompt instruction legacy)
            #         cypher = cypher.replace("<candidate>", cand)
                    
            #         self._validate_cypher(cypher)
            #         records = self._run_cypher(cypher)
                    
            #         if records:
            #             executed_cypher = cypher
            #             break # Tìm thấy dữ liệu thì dừng loop

            #     except Exception as e:
            #         last_error = str(e)
            #         continue
            REAL_SCHEMA_TEXT = self._get_graph_schema(self.driver)
            
            prompt_cy = PROMPT_LLM_CYPHER \
            .replace("<<<QUESTION>>>", standalone_question) \
            .replace("<<<CANDIDATES>>>", json.dumps(selected_entities, ensure_ascii=False)) \
            .replace("<<<SCHEMA>>>", REAL_SCHEMA_TEXT)

            cy_raw = self._call_gemini(prompt_cy, model="gemini-2.0-flash", max_output_tokens=1024)
            
            parsed = self._safe_parse_json(cy_raw)
            cypher = ""
            logger.info("Generating Cypher query...")
            if parsed and "cypher" in parsed:
                cypher = parsed["cypher"]
            else:
                # --- FIX LỖI Ở ĐÂY: Logic Fallback khi LLM không sinh ra json ---
                if selected_entities:
                    # Lấy entity đầu tiên làm vật tế thần để query đơn giản
                    first_entity = selected_entities[0]
                    c_name = first_entity.get('name', '')
                    c_label = first_entity.get('label', 'Benh')
                    cand_esc = c_name.replace("'", "\\'")
                    
                    cypher = f"MATCH (n:{c_label}) WHERE toLower(n.ten) CONTAINS '{cand_esc.lower()}' RETURN n.ten, n.mo_ta"
                else:
                    # Trường hợp xấu nhất: Không tìm thấy entity nào -> Query đại 1 bệnh
                    cypher = "MATCH (n:Benh) RETURN n.ten, n.mo_ta LIMIT 3"

            # Validate cú pháp Cypher (để tránh injection hoặc lỗi cú pháp cơ bản)
            last_error = None
            try:
                self._validate_cypher(cypher)
            except Exception as e:
                last_error = f"Cypher validation failed: {e}"
                # Nếu validate fail, ép về fallback an toàn nhất
                cypher = "MATCH (n:Benh) RETURN n.ten, n.mo_ta LIMIT 1"

            # 4. Fallback Global Search nếu loop trên thất bại toàn tập
            # if not records:
            #     fallback_cypher = "MATCH (b:Benh) RETURN b.ten AS benh LIMIT 5" # Ví dụ lấy random bệnh
            #     try:
            #         # Tìm text match đơn giản nếu RAG fail
            #         simple_cypher = f"MATCH (n) WHERE toLower(n.ten) CONTAINS '{user_question.split()[-1].lower()}' RETURN n.ten, properties(n) LIMIT 5"
            #         records = self._run_cypher(simple_cypher)
            #         executed_cypher = simple_cypher
            #     except:
            #         pass
            
            records = []
            executed_cypher = None

            try:
                if not last_error: # Chỉ chạy nếu validate qua
                    records = self._run_cypher(cypher)
                    executed_cypher = cypher
            except Exception as e:
                last_error = f"Neo4j execution error: {e}"
                records = []

            # 5. Final Answer
            # facts_json = json.dumps(records, ensure_ascii=False)
            # prompt_answer = PROMPT_LLM_ANSWER.replace("<<<QUESTION>>>", user_question).replace("<<<FACTS>>>", facts_json)
            
            # try:
            #     final_resp = self.gemini_model.generate_content(prompt_answer)
            #     final_answer = final_resp.text
            # except Exception as e:
            #     final_answer = f"Xin lỗi, tôi gặp sự cố khi tổng hợp câu trả lời. Lỗi: {str(e)}"
            facts_json = json.dumps(records, ensure_ascii=False)
        
            prompt_answer = PROMPT_LLM_ANSWER \
                .replace("<<<QUESTION>>>", standalone_question) \
                .replace("<<<FACTS>>>", facts_json)
                
            logger.info("Generating final answer with Gemini...")
            final_answer = self._call_gemini(prompt_answer, model="gemini-2.0-flash", max_output_tokens=4096)

            return {
                "answer": final_answer,
                "debug_info": {
                    "question": user_question,
                    "rewritten_question": standalone_question,
                    # "ordered_candidates": ordered_candidates[:5],
                    "selected_entities": selected_entities,
                    "executed_cypher": executed_cypher,
                    "record_count": len(records),
                    "last_error": last_error
                }
            }
            
        except Exception as e:
            logger.error("CRITICAL ERROR IN PROCESS_QUESTION", exc_info=True)
            return {"answer": "Lỗi hệ thống", "error": str(e)}