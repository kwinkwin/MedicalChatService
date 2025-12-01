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
from app.config import settings

# --- 1. PROMPTS (Copy từ Notebook) ---
PROMPT_LLM_CANONICAL = (
    "Bạn là mô-đun CANONICALIZATION cho hệ thống hỏi đáp y tế dựa trên Knowledge Graph Neo4j.\n\n"
    "Nhiệm vụ:\n"
    "- Chuẩn hóa và xếp thứ tự danh sách tên node candidates theo mức độ liên quan với câu hỏi của người dùng.\n"
    "- Tập trung vào NGỮ NGHĨA, từ khóa y tế, và sự phù hợp với các nhóm node phổ biến như: \n"
    "  Benh, TrieuChung, BienChung, DieuTri, NguyenNhan, PhongNgua, DoiTuongNguyCo, YeuToNguyCo.\n\n"
    "Input:\n"
    "User question: <QUESTION>\n"
    "Candidates list (mỗi dòng một candidate tên node)\n\n"
    "Output:\n"
    "Một JSON *duy nhất* dạng:\n"
    "{\"candidates\": [\"name1\", \"name2\", ...]}\n"
    "Sắp xếp theo mức liên quan giảm dần.\n\n"
    "Quy định:\n"
    "- Không giải thích.\n"
    "- Không thêm văn bản ngoài JSON.\n"
    "- Nếu không xác định được thứ tự, giữ nguyên danh sách đầu vào."
)

PROMPT_LLM_CYPHER = (
    "Bạn là chuyên gia Neo4j và chuyên gia y tế. Nhiệm vụ:\n"
    "Sinh một Cypher query hợp lệ để trả lời câu hỏi của người dùng bằng cách sử dụng các node và quan hệ trong Knowledge Graph y tế.\n\n"
    "Input:\n"
    "User question: <<<QUESTION>>>\n"
    "Canonical candidates (JSON array hoặc list string): <<<CANDIDATES>>>\n\n"
    "Các loại node có thể gồm:\n"
    "Benh, TrieuChung, BienChung, DieuTri, NguyenNhan, PhongNgua,\n"
    "DoiTuongNguyCo, YeuToNguyCo, ...\n\n"
    "Quan hệ phổ biến (nhưng không giới hạn):\n"
    "CO_TRIEU_CHUNG,\n"
    "CO_BIEN_CHUNG,\n"
    "DO_NGUYEN_NHAN,\n"
    "DUOC_DIEU_TRI_BANG,\n"
    "DUOC_PHONG_NGUA_BANG,\n"
    "CO_DOI_TUONG_NGUY_CO,\n"
    "CO_YEU_TO_NGUY_CO,\n"
    "GAY_BIEN_CHUNG,\n"
    "ANH_HUONG_DEN,\n"
    "PHONG_NGUA_BANG,\n"
    "...\n\n"
    "Hướng dẫn ánh xạ:\n"
    "- Hãy tự suy luận từ câu hỏi nên cần loại quan hệ nào (ví dụ: triệu chứng -> CO_TRIEU_CHUNG).\n"
    "- Nếu không chắc chắn, ưu tiên MATCH hai chiều (b)--[]--(x) để tăng recall.\n"
    "- Luôn tìm node xuất phát bằng CONTAINS: toLower(n.ten) CONTAINS toLower('<candidate>').\n"
    "- Query trả về dữ liệu cần thiết để trả lời câu hỏi (tên node chính và các node liên quan).\n\n"
    "Output:\n"
    "CHỈ trả về JSON dạng:\n"
    "{\"cypher\": \"<CYPHER_QUERY>\"}\n\n"
    "Quy định:\n"
    "- Không thêm mô tả.\n"
    "- Không đưa ví dụ.\n"
)

PROMPT_LLM_ANSWER = (
    "Bạn là 'Hệ thống Hỗ trợ Quyết định Lâm sàng' (Clinical Decision Support System - CDSS).\n"
    "Nhiệm vụ của bạn là tổng hợp dữ liệu thô từ Knowledge Graph thành một báo cáo y khoa có cấu trúc, logic và có cơ sở lý luận.\n\n"
    "Input:\n"
    "User Question: <<<QUESTION>>>\n"
    "Graph Facts (Tri thức từ Neo4j): <<<FACTS>>>\n\n"
    "Nguyên tắc cốt lõi:\n"
    "1. **Evidence-Based (Dựa trên bằng chứng)**: Mọi câu khẳng định phải được truy xuất trực tiếp từ `Graph Facts`. Tuyệt đối không bịa đặt.\n"
    "2. **Explainability (Tính giải thích)**: Đừng chỉ liệt kê. Hãy giải thích mối liên hệ. \n"
    "   - Kém: 'Bệnh A có triệu chứng B.'\n"
    "   - Tốt: 'Theo dữ liệu hệ thống, Bệnh A **biểu hiện qua** triệu chứng B, điều này gợi ý mối liên hệ về [thuộc tính mô tả nếu có]...'\n"
    "3. **Authority (Tính chuyên gia)**: Sử dụng thuật ngữ chính xác trong Facts. Trình bày mạch lạc.\n\n"
    "Cấu trúc câu trả lời bắt buộc:\n"
    "**1. Phân tích tổng quan:**\n"
    "   - Tóm tắt trực tiếp câu trả lời cho vấn đề người dùng hỏi dựa trên các Entity chính tìm thấy.\n\n"
    "**2. Cơ sở tri thức & Chi tiết (The 'Why' & 'How'):**\n"
    "   - Trình bày chi tiết các mối quan hệ tìm thấy. Nhóm thông tin theo logic (ví dụ: Nhóm Triệu chứng, Nhóm Nguyên nhân, Nhóm Điều trị).\n"
    "   - *Quan trọng*: Nếu trong Facts có thông tin về cơ chế, mô tả, hoặc nguyên lý, hãy lồng ghép vào để giải thích tại sao có mối liên hệ này.\n"
    "   - Sử dụng định dạng: '- **[Tên Node A]** có quan hệ *[Quan hệ]* với **[Tên Node B]**...'\n\n"
    "**3. Lưu ý quan trọng:**\n"
    "   - Cảnh báo hoặc chống chỉ định nếu dữ liệu có đề cập.\n\n"
    "----------------\n"
    "Disclaimer: \"Câu trả lời được tổng hợp tự động từ cơ sở tri thức y khoa. Thông tin chỉ mang tính tham khảo hỗ trợ chẩn đoán. Vui lòng tham vấn bác sĩ chuyên khoa cho các quyết định điều trị.\"\n\n"
    "Output:\n"
    "Trả về định dạng Markdown chuyên nghiệp. Nếu Facts rỗng, trả lời chân thành: \"Hệ thống tri thức hiện tại chưa ghi nhận đủ dữ liệu về vấn đề này để đưa ra lập luận chính xác.\""
)

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
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        try:
            outputs = self.hf_client.feature_extraction(
                model=HF_EMBEDDING_MODEL,
                text=texts
            )
            # Xử lý output format của HF API
            if isinstance(outputs, list):
                # Trường hợp trả về list of list (token embeddings) -> Average Pooling
                if len(outputs) > 0 and (isinstance(outputs[0], list) or isinstance(outputs[0], np.ndarray)):
                     # Nếu là token embedding [seq_len, hidden_dim] -> lấy mean
                     # Code notebook xử lý: outputs = [np.mean(tokens, axis=0) for tokens in outputs]
                     # Nhưng API inference đôi khi trả về trực tiếp pooled embedding.
                     # Ta kiểm tra an toàn:
                     if isinstance(outputs[0][0], list) or isinstance(outputs[0][0], float): 
                         # Đây có vẻ là token embeddings hoặc vector trực tiếp
                         pass 
            
            # Để đơn giản và an toàn với numpy:
            arr = np.array(outputs)
            if arr.ndim == 3: # [batch, seq, dim]
                arr = np.mean(arr, axis=1)
            
            return np.asarray(arr, dtype=np.float32)

        except Exception as e:
            print(f"Embedding Error: {e}")
            # Fallback dimension cho sbert (384) hoặc model khác
            return np.zeros((len(texts), 384), dtype=np.float32)

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

    # --- CORE FUNCTIONS ---

    def load_data_and_build_index(self):
        # --- CÁCH 1: THỬ LOAD TỪ FILE CACHE TRƯỚC (SIÊU NHANH) ---
        if os.path.exists(self.INDEX_FILE) and os.path.exists(self.META_FILE):
            print(f">>> Found cached index files. Loading from disk...")
            try:
                # Load FAISS Index
                self.index = faiss.read_index(self.INDEX_FILE)
                
                # Load Metadata
                with open(self.META_FILE, "rb") as f:
                    self.metadatas = pickle.load(f)
                
                print(f">>> Loaded successfully! {self.index.ntotal} vectors ready in < 1s.")
                self.is_ready = True
                return
            except Exception as e:
                print(f">>> Cache corrupted ({e}). Re-building from scratch...")
        
        # --- CÁCH 2: NẾU KHÔNG CÓ CACHE THÌ TẢI TỪ NEO4J (CHẠY LẦN ĐẦU) ---
        print(">>> Cache miss. Starting to load nodes from Neo4j...")
        q = """
        MATCH (n)
        WHERE n.ten IS NOT NULL OR n.name IS NOT NULL
        RETURN id(n) AS id, labels(n) AS labels, coalesce(n.ten, n.name) AS ten, properties(n) AS props
        LIMIT 5000 
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
                self.is_ready = True 
                return

            print(f">>> Embedding {len(nodes)} nodes (Batch processing)...")
            texts = [n.text for n in nodes]
            
            # Batch embedding
            batch_size = 32
            all_vecs = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                vecs = self._embed_texts(batch)
                all_vecs.append(vecs)
                # In log tiến độ để đỡ sốt ruột
                if i % 100 == 0:
                    print(f"   Processed {i}/{len(texts)}...")

            if all_vecs:
                final_vecs = np.concatenate(all_vecs, axis=0)
                norms = np.linalg.norm(final_vecs, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                final_vecs = final_vecs / norms

                self.index = faiss.IndexFlatIP(final_vecs.shape[1])
                self.index.add(final_vecs.astype('float32'))
                
                self.metadatas = [
                    {"node_id": n.node_id, "label": n.label, "name": n.name, "text": n.text, "properties": n.properties}
                    for n in nodes
                ]
                
                # --- LƯU CACHE XUỐNG ĐĨA ---
                print(">>> Saving index to disk for next run...")
                faiss.write_index(self.index, self.INDEX_FILE)
                with open(self.META_FILE, "wb") as f:
                    pickle.dump(self.metadatas, f)
                
                print(f">>> FAISS Index built & saved with {self.index.ntotal} vectors.")
                self.is_ready = True
                
        except Exception as e:
            print(f">>> CRITICAL ERROR loading data: {e}")

    async def process_question(self, user_question: str) -> Dict[str, Any]:
        """
        Main pipeline: Retrieve -> Canonicalize -> Cypher Gen -> Execute -> Answer
        """
        if not self.is_ready:
            return {"answer": "Hệ thống đang khởi động hoặc gặp lỗi kết nối CSDL."}

        # 1. Retrieve Candidates
        qvec = self._embed_texts([user_question])[0]
        qvec = qvec / (np.linalg.norm(qvec) + 1e-12)
        
        # Search FAISS
        D, I = self.index.search(np.array([qvec], dtype=np.float32), TOP_K)
        
        retrieved_candidates = []
        for score, idx in zip(D[0], I[0]):
            if 0 <= idx < len(self.metadatas):
                meta = self.metadatas[idx]
                retrieved_candidates.append(meta)
        
        candidate_names = [m['name'] for m in retrieved_candidates]

        # 2. Canonicalize (Optional step to re-rank via LLM)
        # Giảm số lượng candidate truyền vào prompt để tiết kiệm token
        top_candidates_str = "\n".join(candidate_names[:50]) 
        prompt_can = PROMPT_LLM_CANONICAL.replace("<QUESTION>", user_question) + "\nCandidates:\n" + top_candidates_str
        
        try:
            canon_raw = self.gemini_model.generate_content(prompt_can).text
            canon_json = self._safe_parse_json(canon_raw)
            if canon_json and isinstance(canon_json.get("candidates"), list):
                ordered_candidates = canon_json["candidates"]
            else:
                ordered_candidates = candidate_names
        except Exception as e:
            print(f"Canonicalize error: {e}")
            ordered_candidates = candidate_names

        # 3. Generate Cypher Loop
        executed_cypher = None
        records = []
        last_error = None
        
        # Chỉ thử 3 candidate tốt nhất để tiết kiệm thời gian
        for attempt_idx, cand in enumerate(ordered_candidates[:MAX_CYPHER_RETRIES]):
            prompt_cy = PROMPT_LLM_CYPHER.replace("<<<QUESTION>>>", user_question).replace("<<<CANDIDATES>>>", json.dumps([cand]))
            
            try:
                cy_resp = self.gemini_model.generate_content(prompt_cy)
                parsed = self._safe_parse_json(cy_resp.text)
                
                if parsed and "cypher" in parsed:
                    cypher = parsed["cypher"]
                else:
                    # Fallback cypher đơn giản
                    cand_esc = cand.replace("'", "\\'")
                    cypher = f"MATCH (b) WHERE toLower(b.ten) CONTAINS '{cand_esc.lower()}' RETURN b.ten AS name, properties(b) AS props LIMIT 5"

                # Replace placeholder if any (from prompt instruction legacy)
                cypher = cypher.replace("<candidate>", cand)
                
                self._validate_cypher(cypher)
                records = self._run_cypher(cypher)
                
                if records:
                    executed_cypher = cypher
                    break # Tìm thấy dữ liệu thì dừng loop

            except Exception as e:
                last_error = str(e)
                continue

        # 4. Fallback Global Search nếu loop trên thất bại toàn tập
        if not records:
            fallback_cypher = "MATCH (b:Benh) RETURN b.ten AS benh LIMIT 5" # Ví dụ lấy random bệnh
            try:
                # Tìm text match đơn giản nếu RAG fail
                simple_cypher = f"MATCH (n) WHERE toLower(n.ten) CONTAINS '{user_question.split()[-1].lower()}' RETURN n.ten, properties(n) LIMIT 5"
                records = self._run_cypher(simple_cypher)
                executed_cypher = simple_cypher
            except:
                pass

        # 5. Final Answer
        facts_json = json.dumps(records, ensure_ascii=False)
        prompt_answer = PROMPT_LLM_ANSWER.replace("<<<QUESTION>>>", user_question).replace("<<<FACTS>>>", facts_json)
        
        try:
            final_resp = self.gemini_model.generate_content(prompt_answer)
            final_answer = final_resp.text
        except Exception as e:
            final_answer = f"Xin lỗi, tôi gặp sự cố khi tổng hợp câu trả lời. Lỗi: {str(e)}"

        return {
            "answer": final_answer,
            "debug_info": {
                "question": user_question,
                "ordered_candidates": ordered_candidates[:5],
                "executed_cypher": executed_cypher,
                "record_count": len(records),
                "last_error": last_error
            }
        }