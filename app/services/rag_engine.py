import pickle
import os
import json
import re
import time
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from neo4j import GraphDatabase
from huggingface_hub import InferenceClient
from google.api_core import retry, exceptions
import logging
from app.config import settings
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# --- 1. PROMPTS (Copy từ Notebook) ---
PROMPT_LLM_CANONICAL = (
    "Bạn là mô-đun Entity Linking & Canonicalization cho hệ thống Y tế.\n"
    "Nhiệm vụ: Đọc câu hỏi và danh sách các thực thể (candidates) được tìm thấy từ Database. "
    "Hãy chọn lọc và sắp xếp lại các thực thể phù hợp nhất để trả lời câu hỏi. (Nếu thực thể là Bệnh thì ưu tiên xếp đầu tiên, nếu không có bệnh phù hợp thì bắt buộc phải lấy bệnh liên quan nhất)\n\n"
    
    "QUAN TRỌNG: Hãy chú ý đến trường 'label' (Loại thực thể) để đảm bảo đúng ngữ cảnh:\n"
    "- Nếu người dùng hỏi về 'triệu chứng', ưu tiên chọn candidate có label 'TrieuChung'.\n"
    "- Nếu người dùng hỏi về 'thuốc trị bệnh', ưu tiên chọn candidate có label 'Thuoc' hoặc 'Benh'.\n"
    "- Loại bỏ các candidate không liên quan.\n\n"
    
    "Input:\n"
    "User Question: <QUESTION>\n"
    "Raw Candidates (Tìm theo ngữ cảnh) (JSON): <CANDIDATES>\n"
    "Disease Candidates (Tìm theo tên bệnh chính xác) (JSON): <BENH_CANDIDATES>\n\n"
    
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

PROMPT_LLM_CYPHER = (
    "Bạn là chuyên gia Neo4j và Phân tích Ý định Y khoa (Medical Intent Analyst).\n"
    "Nhiệm vụ: Phân tích Input người dùng và danh sách Thực thể đã chọn (Selected Entities) và sinh câu lệnh Cypher phù hợp nhất.\n\n"
    
    "Input:\n"
    "1. User Input: <<<QUESTION>>>\n"
    "2. Selected Entities: <<<CANDIDATES>>>\n"
    "3. Schema: <<<SCHEMA>>>\n\n"
    
    "QUY TẮC BẤT DI BẤT DỊCH (HARD RULES):\n"
    "1. KHÔNG BAO GIỜ dùng `{id: ...}` hoặc `elementId(...)` trong câu lệnh MATCH. ID từ Vector Search có thể không khớp với cấu trúc Graph.\n"
    "2. LUÔN dùng `WHERE toLower(n.ten) CONTAINS toLower('từ khóa')` để tìm node.\n"
    "3. Khi kiểm tra quan hệ giữa A và B, đừng tìm chính xác node B. Hãy tìm node A, bung mở quan hệ, và lọc kết quả chứa những từ khóa giống ngữ nghĩa với nghĩa của B.\n"
    "4. Khi dùng nhiều quan hệ (OR logic), dùng cú pháp `[:REL_A|REL_B]` (chỉ một dấu hai chấm đầu tiên). KHÔNG DÙNG `[:REL_A|:REL_B]`.\n"
    "5. QUAN TRỌNG: Khi RETURN loại quan hệ, CHỈ được dùng `type(r)`. TUYỆT ĐỐI KHÔNG dùng `type((a)-[r]->(b))`. Luôn đảm bảo biến `r` đã được định nghĩa trong MATCH (ví dụ: `MATCH (a)-[r:REL]->(b)` thay vì `MATCH (a)-[:REL]->(b)`).\n"
    "6. QUAN TRỌNG VỀ QUAN HỆ (RELATIONSHIP VARIABLE):\n"
    "   - SAI: MATCH (a)-[:REL]->(b) RETURN type(r) (Lỗi: biến 'r' chưa được định nghĩa)\n"
    "   - SAI: MATCH (a)-[:REL]->(b) RETURN type((a)-[r]->(b)) (Lỗi: cú pháp không tồn tại)\n"
    "   - ĐÚNG: MATCH (a)-[r:REL]->(b) RETURN type(r) (Phải gán biến 'r' ngay trong MATCH)\n"
    
    "QUY TRÌNH SUY LUẬN (CHAIN OF THOUGHT):\n"
    "Bước 1: Xác định Ý ĐỊNH (Intent) & QUAN HỆ CỤ THỂ:\n"
    "   - ví dụ Hỏi 'Biến chứng của X': Cần tìm quan hệ [:GAY_BIEN_CHUNG] hoặc [:CO_BIEN_CHUNG].\n"
    "   - ví dụ Hỏi 'Triệu chứng của X': Cần tìm quan hệ [:CO_TRIEU_CHUNG].\n"
    "   - ví dụ Hỏi 'Nguyên nhân của X': Cần tìm quan hệ [:DO_NGUYEN_NHAN] hoặc ngược lại.\n"
    "   - Nếu không rõ quan hệ: Dùng quan hệ chung (-[]-).\n\n"
    
    "Bước 2: Chọn CHIẾN THUẬT QUERY tối ưu (1, 2, 3 hoặc 4):\n\n"
    
    "--- CHIẾN THUẬT 1: TRUY VẤN HƯỚNG ĐÍCH (Targeted Expansion) ---\n"
    "Áp dụng khi: Hỏi về thuộc tính của 1 thực thể (VD: Biến chứng của X? Phòng ngừa X thế nào? X do đâu?).\n"
    "Yêu cầu 1: KHÔNG ĐƯỢC LỌC target node. Hãy để Graph trả về toàn bộ kết quả.\n"
    "Yêu cầu 2: Sinh 2 queries chạy song song.\n"
    "   - Query 1 (Gốc): KHÔNG ĐƯỢC LỌC target node. Hãy để Graph trả về toàn bộ kết quả.\n"
    "   - Query 2 (Bổ sung): Tìm TẤT CẢ quan hệ xung quanh root (1-hop) để lấy ngữ cảnh rộng.\n"
    "Pattern:\n"
    "   Query 1: MATCH (root:Benh|TrieuChung|Thuoc)\n"
    "            WHERE toLower(root.ten) CONTAINS 'tên entity A'\n"
    "            MATCH (root)-[r:LOAI_QUAN_HE_CU_THE]->(target)\n"
    "            RETURN root.ten, root.nguon, type(r), target.ten, target.nguon, target.mo_ta\n"
    "\n"
    "   Query 2: MATCH (root:Benh|TrieuChung|Thuoc)-[r]-(m)\n"
    "            WHERE toLower(root.ten) CONTAINS 'tên entity A'\n"
    "            RETURN root.ten, root.nguon, type(r), m.ten, m.nguon, m.mo_ta\n\n"
    
    "--- CHIẾN THUẬT 2: XÁC MINH QUAN HỆ (Verification) ---\n"
    "Áp dụng khi: Hỏi xác nhận mối quan hệ cụ thể (VD: Béo phì có phải do ít vận động không?).\n"
    "Yêu cầu 1: Phải LỌC target node dựa trên từ khóa trong câu hỏi.\n"
    "Yêu cầu 2: Sinh 2 queries chạy song song.\n"
    "   - Query 1 (Gốc): Phải LỌC target node dựa trên từ khóa trong câu hỏi (Sử dụng logic OR cho các từ đồng nghĩa).\n"
    "   - Query 2 (Bổ sung): Tìm TẤT CẢ quan hệ xung quanh root (1-hop) đề phòng Query 1 bị lọt lưới.\n"
    "Pattern:\n"
    "   Query 1: MATCH (root:Benh|TrieuChung|Thuoc)\n"
    "            WHERE toLower(root.ten) CONTAINS 'tên entity A'\n"
    "            MATCH (root)-[r:LOAI_QUAN_HE_CU_THE]-(target)\n"
    "            // Logic lọc cốt lõi (OR):\n"
    "            WHERE toLower(target.ten) CONTAINS 'từ khóa cốt lõi B1' OR toLower(target.ten) CONTAINS 'từ khóa cốt lõi B2' ... (OR đến khi hết Selected Entities)\n"
    "            RETURN root.ten, root.nguon, type(r), target.ten, target.nguon, target.mo_ta\n"
    "\n"
    "   Query 2: MATCH (root:Benh|TrieuChung|Thuoc)-[r]-(m)\n"
    "            WHERE toLower(root.ten) CONTAINS 'tên entity A'\n"
    "            RETURN root.ten, root.nguon, type(r), m.ten, m.nguon, m.mo_ta\n\n"
    
    "--- CHIẾN THUẬT 3: CHẨN ĐOÁN & TÌM KIẾM TỔ HỢP (Diagnosis) ---\n"
    "Áp dụng khi: User liệt kê nhiều triệu chứng hoặc yếu tố.\n"
    "Pattern:\n"
    "MATCH (t:Benh)-[:CO_TRIEU_CHUNG|DO_NGUYEN_NHAN]-(s)\n"
    "WHERE toLower(s.ten) CONTAINS 'yếu tố 1' OR toLower(s.ten) CONTAINS 'yếu tố 2' ... (OR đến khi hết Selected Entities)\n"
    "WITH t, count(distinct s) AS matches, collect(distinct s.ten) AS evidence\n"
    "RETURN t.ten, t.nguon, t.mo_ta, matches, evidence ORDER BY matches DESC LIMIT 20\n\n"
    
    "--- CHIẾN THUẬT 4: TÌM ĐƯỜNG NGẮN NHẤT AN TOÀN (Safe Shortest Path) ---\n"
    "Áp dụng khi: Tìm quan hệ giữa 2 thực thể A và B (Tương tác thuốc, Bệnh A có gây B không?).\n"
    "Quy tắc an toàn:\n"
    "1. Phải gán Label cho node đầu/cuối (thường là Benh, Thuoc, TrieuChung).\n"
    "2. Giới hạn hops tối đa là 3 (range *..3).\n"
    "Pattern:\n"
    "MATCH (a:Benh|Thuoc|TrieuChung), (b:Benh|Thuoc|TrieuChung)\n"
    "WHERE toLower(a.ten) CONTAINS 'entity 1' AND toLower(b.ten) CONTAINS 'entity 2'\n"
    "MATCH p = shortestPath((a)-[*..3]-(b))\n"
    "RETURN [n in nodes(p) | n.ten] AS path_names, [n in nodes(p) | n.nguon] AS path_sources, length(p) AS hops\n\n"
    
    "Bước 3: Sinh Cypher (Final Output).\n"
    "Yêu cầu:\n"
    "- Ưu tiên dùng `Label` chính xác nếu biết (ví dụ tìm thuốc thì target phải là :Thuoc).\n"
    "- Khi RETURN node Bệnh, cố gắng lấy thêm thuộc tính `.nguon` (nếu có).\n"
    "- Output JSON duy nhất với cấu trúc sau (Đặc biệt Trường 'cypher' PHẢI LÀ MỘT DANH SÁCH (List of Strings)): \n"
    "{\n"
    "  \"thought_process\": \"Giải thích ngắn gọn tại sao chọn chiến thuật này (dựa trên intent người dùng)\",\n"
    "  \"strategy_id\": \"1 (Khám phá) hoặc 2 (Xác minh) hoặc 3 (Chẩn đoán) hoặc 4 (Tìm đường)\",\n"
    "  \"cypher\": [\n"
    "      \"MATCH ... (Query 1) ...\",\n"
    "      \"MATCH ... (Query 2) ...\"\n"
    "  ]\n"
    "}"
)

PROMPT_LLM_ANSWER = """
Bạn là **MediBot**, một trợ lý AI y tế thông minh đóng vai trò là 'Hệ thống Hỗ trợ Quyết định Lâm sàng' (CDSS).
Mục tiêu: Hỗ trợ người dùng/bác sĩ bằng các thông tin chính xác từ dữ liệu Knowledge Graph (KG) hoặc giao tiếp thân thiện khi được chào hỏi.

Input:
- Câu hỏi: <<<QUESTION>>>
- Dữ liệu KG: <<<FACTS>>>

Quy trình xử lý (Chain of Thought):
1. **PHÂN LOẠI Ý ĐỊNH**:
   - *Trường hợp 1 (Giao tiếp xã hội/Hỏi danh tính)*: Nếu câu hỏi là "xin chào", "hi", "bạn là ai", "bạn tên gì"... -> Chuyển sang chế độ GIAO TIẾP.
   - *Trường hợp 2 (Truy vấn y tế)*: Nếu câu hỏi về bệnh, thuốc, triệu chứng, biến chứng, cách điều trị, đối tượng nguy cơ, nguyên nhân, phòng ngừa, yếu tố nguy cơ... -> Chuyển sang chế độ PHÂN TÍCH KG.

2. **CHẾ ĐỘ GIAO TIẾP (Nếu là Trường hợp 1)**:
   - Trả lời thân thiện, xưng là "MediBot".
   - Giới thiệu ngắn gọn chức năng: Tra cứu thông tin y tế dựa trên cơ sở dữ liệu tin cậy.
   - Kết thúc bằng câu gợi mở hỏi về sức khỏe.

3. **CHẾ ĐỘ PHÂN TÍCH KG (Nếu là Trường hợp 2)**:
   - Bước 1 (Lọc): Chỉ chọn các facts trong KG trực tiếp trả lời cho "Câu hỏi". Bỏ qua dữ liệu nhiễu.
   - Bước 2 (Tổng hợp): Viết lại thành văn phong y khoa tự nhiên, không liệt kê máy móc dạng A-relation-B.
   - Bước 3 (Kiểm tra rỗng): Nếu Dữ liệu KG rỗng hoặc không tìm thấy thông tin liên quan -> Trả lời bằng câu Fallback quy định bên dưới.

---
CẤU TRÚC OUTPUT (Chỉ áp dụng cho CHẾ ĐỘ PHÂN TÍCH KG):

Nếu có dữ liệu KG liên quan, hãy trình bày theo format sau:

### 1. Kết luận Lâm sàng (Direct Answer)
- Trả lời trực diện câu hỏi dựa trên bằng chứng mạnh nhất từ KG.

### 2. Phân tích Chi tiết
*(Nhóm các thông tin tương đồng, không liệt kê máy móc)*
- **Về Bệnh học/Triệu chứng/Biến chứng/Điều trị/Đối tượng nguy cơ/Nguyên nhân/Phòng ngừa/Yếu tố nguy cơ**: [Tổng hợp các node liên quan.]
- **Về Điều trị/Thuốc**: [Tổng hợp các node liên quan]
- **Cơ chế/Lý do (Nếu có)**: [Giải thích ngắn gọn mối liên kết giữa các node]

### 3. Cảnh báo (Alerts)
- Nêu bật các chống chỉ định hoặc rủi ro tìm thấy (Nếu không có, bỏ qua mục này).
- *Lưu ý: Thông tin từ MediBot chỉ mang tính tham khảo. Vui lòng tham vấn bác sĩ chuyên khoa.*

### 4. Nguồn tham khảo
- [Nếu node có thuộc tính `nguon`, hiển thị dạng: "- [Tên bệnh tương ứng với 'nguon'](url)"]

Quy tắc Output (Nghiêm ngặt):
- **Conciseness (Súc tích)**: Đi thẳng vào vấn đề. Sử dụng gạch đầu dòng ngắn gọn.
- **Evidence-Only**: Chỉ dùng thông tin từ Dữ liệu KG. Nếu thiếu thông tin quan trọng, hãy nói rõ là dữ liệu chưa cập nhật.
- **No Fluff**: Bỏ qua các câu dẫn dắt rườm rà (ví dụ: "Dựa trên dữ liệu được cung cấp...").

---
QUY TẮC PHẢN HỒI (Nghiêm ngặt):
- **Tên gọi**: Luôn xưng là **MediBot** (không dùng cụm từ "Hệ thống CDSS" khi xưng hô).
- **Trường hợp Fallback (Dành cho câu hỏi y tế nhưng KG rỗng)**:
  "Hiện tại MediBot chỉ tập trung vào các chủ đề y tế phổ biến có trong cơ sở dữ liệu, nên tôi chưa đủ thông tin để trả lời câu hỏi này. Bạn có câu hỏi nào khác liên quan đến sức khỏe không?"
- **Format**: Markdown.
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
TOP_K = 500
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
    
    BENH_INDEX_FILE = "benh_vector_store.faiss" 
    BENH_META_FILE = "benh_metadata.pkl"

    def __init__(self):
        # Setup Connections
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI, 
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )
        
        self.hf_client = InferenceClient(
            token=settings.HF_API_KEY,
            provider="hf-inference"
        )
        
        # In-memory Vector Store
        self.index = None
        self.metadatas = []
        
        self.benh_index = None
        self.benh_metadatas = []
        
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
        schema_text += "\n\nNode Properties Assumptions:\n- Hầu hết các Node đều có thuộc tính: 'ten' (tên hiển thị), 'mo_ta' (mô tả chi tiết), 'id' (mã định danh)."
        
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
    
    # @retry.Retry(predicate=retry.if_exception_type(exceptions.ResourceExhausted))
    def _call_genma(self, prompt: str, model: str = "gemma-3-27b-it", max_output_tokens: int = 1024, temperature: float = 0.0, api_key=None) -> str:
        """
        Call Genma with a single plain-string prompt (no system role).
        Returns text content.
        """
        max_retries = len(settings.GOOGLE_KEYS) if settings.GOOGLE_KEYS else 1
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # 1. Nếu không truyền api_key cụ thể, lấy key từ vòng xoay
                current_api_key = api_key if api_key else settings.get_next_google_key()
                
                # 2. Khởi tạo client với key này
                self.client = genai.Client(api_key=current_api_key)
                
                # 3. Gọi API
                resp = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        safety_settings=[
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                threshold=types.HarmBlockThreshold.OFF,
                            ),
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                                threshold=types.HarmBlockThreshold.OFF,
                            ),
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                                threshold=types.HarmBlockThreshold.OFF,
                            ),
                        ],
                    )
                )
                
                # Nếu thành công, trả về kết quả ngay
                return resp.text

            except exceptions.ResourceExhausted as e:
                # 429: Hết Quota -> Log warning và tiếp tục vòng lặp (lấy key tiếp theo)
                logger.warning(f"Key {current_api_key[:10]}... hết Quota. Đang đổi sang key khác... (Lần thử {attempt + 1}/{max_retries})")
                last_error = e
                # Set api_key về None để vòng lặp sau tự lấy key mới từ settings
                api_key = None 
                time.sleep(1) # Nghỉ 1 xíu trước khi đổi
                continue

            except exceptions.PermissionDenied as e:
                # 403: Key sai hoặc bị khóa -> Log và đổi key
                logger.warning(f"Key {current_api_key[:10]}... bị từ chối quyền. Đang đổi... (Lần thử {attempt + 1}/{max_retries})")
                last_error = e
                api_key = None
                continue

            except Exception as e:
                # Các lỗi khác (500, Bad Request...) thì throw luôn, không đổi key làm gì
                logger.error(f"Lỗi không liên quan đến Key: {e}")
                raise e

        # Nếu chạy hết vòng lặp (hết sạch key) mà vẫn lỗi
        logger.error("Đã thử tất cả các Key nhưng đều thất bại.")
        raise last_error
    
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
            rewritten = self._call_genma(prompt, max_output_tokens=256)
            return rewritten.strip()
        except Exception as e:
            logger.error(f"Error rewriting question: {e}")
            return user_question # Fallback về câu gốc nếu lỗi

    # --- CORE FUNCTIONS ---

    def load_data_and_build_index(self):
        # --- CÁCH 1: THỬ LOAD TỪ FILE CACHE TRƯỚC (SIÊU NHANH) ---
        if os.path.exists(self.INDEX_FILE) and os.path.exists(self.BENH_INDEX_FILE):
            print(">>> Loading cached indices...")
            try:
                self.index = faiss.read_index(self.INDEX_FILE)
                with open(self.META_FILE, "rb") as f: self.metadatas = pickle.load(f)
                
                # Load Benh Index
                self.benh_index = faiss.read_index(self.BENH_INDEX_FILE)
                with open(self.BENH_META_FILE, "rb") as f: self.benh_metadatas = pickle.load(f)
                
                logger.info(f"Loaded: General ({self.index.ntotal}), Benh ({self.benh_index.ntotal})")
                self.is_ready = True
                return
            except Exception as e:
                logger.error(f"Cache corrupted: {e}. Rebuilding...")
        
        # --- CÁCH 2: NẾU KHÔNG CÓ CACHE THÌ TẢI TỪ NEO4J (CHẠY LẦN ĐẦU) ---
        print(">>> Building indices from Neo4j...")
        
        # Query lấy TOÀN BỘ node (cho General Index)
        q = """
        MATCH (n)
        WHERE n.ten IS NOT NULL OR n.name IS NOT NULL
        RETURN id(n) AS id, labels(n) AS labels, coalesce(n.ten, n.name) AS ten, properties(n) AS props 
        """
        nodes = []
        
        # Query CHỈ lấy node Bệnh (cho Disease Index)
        q_benh = """MATCH (n:Benh) 
        WHERE n.ten IS NOT NULL 
        RETURN id(n) as id, labels(n) as labels, n.ten as ten, properties(n) as props"""
        
        try:
            with self.driver.session() as session:
                res = session.run(q)
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
                    
                    # Logic cũ: Text = Name + Description
                    text = name + (". " + " ".join(desc_parts) if desc_parts else "")
                    nodes.append(NodeRecord(node_id=nid, label=label, name=name, text=text, properties=props))
            
            if not nodes:
                print(">>> No nodes found in Neo4j.")
                return

            print(f">>> Embedding {len(nodes)} nodes (General)...")
            texts = [n.text for n in nodes]
            final_vecs = self._embed_texts(texts, batch_size=50)
            
            norms = np.linalg.norm(final_vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            final_vecs = final_vecs / norms

            # Build FAISS Index (General)
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
            
            print(">>> Building separate Disease Index (Only Name)...")
        
            # Query chỉ lấy node Benh
            q_benh = """
            MATCH (n:Benh)
            WHERE n.ten IS NOT NULL
            RETURN id(n) AS id, labels(n) AS labels, n.ten AS ten, properties(n) AS props
            """
            
            nodes_benh = []
            with self.driver.session() as session:
                res_benh = session.run(q_benh)
                for r in res_benh:
                    nid = str(r["id"])
                    label = "Benh"
                    name = r["ten"]
                    props = dict(r["props"]) if r["props"] else {}
                    
                    # Logic MỚI: Text = ONLY Name (Để bắt chính xác tên bệnh)
                    text = name 
                    nodes_benh.append(NodeRecord(node_id=nid, label=label, name=name, text=text, properties=props))

            print(f">>> Embedding {len(nodes_benh)} disease nodes...")
            if nodes_benh:
                texts_benh = [n.text for n in nodes_benh]
                vecs_benh = self._embed_texts(texts_benh, batch_size=50)
                
                norms_b = np.linalg.norm(vecs_benh, axis=1, keepdims=True)
                norms_b[norms_b == 0] = 1.0
                vecs_benh = vecs_benh / norms_b
                
                # Build FAISS Index (Disease)
                self.benh_index = faiss.IndexFlatIP(vecs_benh.shape[1])
                self.benh_index.add(vecs_benh.astype('float32'))
                
                self.benh_metadatas = [
                    {
                        "node_id": n.node_id, 
                        "label": n.label, 
                        "name": n.name, 
                        "text": n.text, # Lưu ý: ở đây text chỉ là tên
                        "properties": n.properties
                    }
                    for n in nodes_benh
                ]
            else:
                # Fallback nếu không có bệnh nào (để tránh lỗi code sau này)
                self.benh_index = faiss.IndexFlatIP(768) 
                self.benh_metadatas = []
            
            print(">>> Saving all indices to disk...")
        
            # Save General
            faiss.write_index(self.index, self.INDEX_FILE)
            with open(self.META_FILE, "wb") as f:
                pickle.dump(self.metadatas, f)
                
            # Save Disease
            faiss.write_index(self.benh_index, self.BENH_INDEX_FILE)
            with open(self.BENH_META_FILE, "wb") as f:
                pickle.dump(self.benh_metadatas, f)
            
            print(">>> All indices built & saved successfully.")
            self.is_ready = True
            
        except Exception as e:
            print(f">>> CRITICAL ERROR loading data: {e}")
            logger.critical(f"CRITICAL ERROR loading data: {e}", exc_info=True)

    async def process_question(self, user_question: str, history: List[dict]) -> Dict[str, Any]:
        """
        Main pipeline: Retrieve -> Canonicalize -> Cypher Gen -> Execute -> Answer
        """
        # history
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
            D1, I1 = self.index.search(np.array([qvec], dtype=np.float32), TOP_K)
            
            retrieved_candidates = []
            for score, idx in zip(D1[0], I1[0]):
                if 0 <= idx < len(self.metadatas):
                    meta = self.metadatas[idx]
                    retrieved_candidates.append((meta, float(score)))
                    
            D2, I2 = self.benh_index.search(np.array([qvec], dtype=np.float32), 100)
            disease_candidates_full = []
            for idx in I2[0]:
                if 0 <= idx < len(self.benh_metadatas):
                    disease_candidates_full.append(self.benh_metadatas[idx])
                    
            disease_names_only = [item['name'] for item in disease_candidates_full]
            
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
            prompt_can = PROMPT_LLM_CANONICAL \
            .replace("<QUESTION>", standalone_question) \
            .replace("<CANDIDATES>", json.dumps(rich_candidates[:500], ensure_ascii=False)) \
            .replace("<BENH_CANDIDATES>", json.dumps(disease_names_only, ensure_ascii=False))

            canon_raw = self._call_genma(prompt_can, max_output_tokens=1024)
            canon_json = self._safe_parse_json(canon_raw)
            
            selected_entities = []
            # Ưu tiên lấy từ JSON trả về, nếu lỗi thì lấy top 3 từ vector search
            if canon_json and "selected_entities" in canon_json:
                selected_entities = canon_json["selected_entities"]
            else:
                selected_entities = rich_candidates[:3]
                
            logger.info(f"Canonicalized to: {json.dumps(selected_entities, ensure_ascii=False)}...")

            REAL_SCHEMA_TEXT = self._get_graph_schema(self.driver)
            
            prompt_cy = PROMPT_LLM_CYPHER \
            .replace("<<<QUESTION>>>", standalone_question) \
            .replace("<<<CANDIDATES>>>", json.dumps(selected_entities, ensure_ascii=False)) \
            .replace("<<<SCHEMA>>>", REAL_SCHEMA_TEXT)

            cy_raw = self._call_genma(prompt_cy, max_output_tokens=1024)
            parsed = self._safe_parse_json(cy_raw)
            
            # cypher = ""
            cypher_queries = []
            strategy_log = "Unknown"
            reasoning_log = "No reasoning provided"
            
            logger.info("Generating Cypher query...")
            if parsed and "cypher" in parsed:
                # cypher = parsed["cypher"]
                raw_cypher = parsed["cypher"]
                # --- PHẦN MỚI: TRÍCH XUẤT LOG CHIẾN THUẬT ---
                strategy_log = parsed.get("strategy_id", "Unknown")
                reasoning_log = parsed.get("thought_process", "No reasoning")
                
                if isinstance(raw_cypher, list):
                    cypher_queries = raw_cypher
                elif isinstance(raw_cypher, str):
                    cypher_queries = [raw_cypher]
                
                logger.info(f"AI Strategy: {strategy_log} | Queries: {len(cypher_queries)}")
                logger.info(f"CYPHER QUERIES: {cypher_queries}")
            else:
                # --- FIX LỖI Ở ĐÂY: Logic Fallback khi LLM không sinh ra json ---
                strategy_log = "Fallback"
                if selected_entities:
                    # Lấy entity đầu tiên làm vật tế thần để query đơn giản
                    first_entity = selected_entities[0]
                    c_name = first_entity.get('name', '')
                    c_label = first_entity.get('label', 'Benh')
                    cand_esc = c_name.replace("'", "\\'")

                    fallback_cypher = f"""
                    MATCH (n:{c_label})-[r]-(m) 
                    WHERE toLower(n.ten) CONTAINS '{cand_esc.lower()}' 
                    RETURN n.ten, type(r), m.ten, m.mo_ta 
                    LIMIT 20"""
                    cypher_queries = [fallback_cypher]
                else:
                    # Trường hợp xấu nhất: Không tìm thấy entity nào -> Query đại 1 bệnh
                    # cypher = "MATCH (n:Benh) RETURN n.ten, n.mo_ta LIMIT 3"
                    cypher_queries = ["MATCH (n:Benh) RETURN n.ten, n.mo_ta LIMIT 3"]
            
            # records = []
            all_records = []
            seen_records = set()
            executed_cyphers_log = []
            executed_cypher = None
            
            logger.info("Executing Cypher queries...")
            
            for idx, query in enumerate(cypher_queries):
                try:
                    self._validate_cypher(query)
                    records = self._run_cypher(query)
                    
                    for rec in records:
                        rec_clean = {k: v for k, v in rec.items() if not k.startswith('_')}
                        rec_hash = json.dumps(rec_clean, sort_keys=True)
                        
                        if rec_hash not in seen_records:
                            seen_records.add(rec_hash)
                            all_records.append(rec)
                    
                    executed_cyphers_log.append(f"Q{idx+1}: Success ({len(records)} records)")
                except Exception as e:
                    logger.error(f"Cypher Q{idx+1} failed: {query} | Error: {e}")
                    executed_cyphers_log.append(f"Q{idx+1}: Failed")

            # 5. Final Answer
            facts_json = json.dumps(all_records, ensure_ascii=False)
                    
            prompt_answer = PROMPT_LLM_ANSWER \
                .replace("<<<QUESTION>>>", standalone_question) \
                .replace("<<<FACTS>>>", facts_json)
                
            logger.info("Generating final answer with Genma...")
            final_answer = self._call_genma(prompt_answer, max_output_tokens=8192)

            return {
                "answer": final_answer,
                "debug_info": {
                    "question": user_question,
                    "rewritten_question": standalone_question,
                    "strategy": strategy_log,
                    "reasoning": reasoning_log,
                    "executed_cyphers": cypher_queries,
                    "execution_log": executed_cyphers_log,
                    "total_records_found": len(all_records),
                }
            }
            
        except Exception as e:
            logger.error("CRITICAL ERROR IN PROCESS_QUESTION", exc_info=True)
            return {"answer": "Lỗi hệ thống", "error": str(e)}