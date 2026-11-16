# Báo cáo Day 7 — Agentic RAG trên Bộ luật Lao động 2019

Học viên: Phan Trung Hiếu (03763). Toàn bộ số liệu dưới đây lấy từ file trong [results/](../../results/); lệnh tái lập ở cuối mỗi phần.

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:** Bộ luật Lao động số 45/2019/QH14, file `bo_luat_lao_dong_45_2019_QH14.pdf` (sha256 `3c865f3d…b5b6c`), 94 trang PDF (93 trang nội dung). Đây là văn bản quy phạm pháp luật, không thuộc phạm vi bảo hộ quyền tác giả (Điều 15 Luật Sở hữu trí tuệ), nên được dùng để index và trích dẫn. Dùng lại index Day 6, chiến lược **B — by_article: 220 chunk** (mỗi chunk là một Điều; trung vị 718 ký tự, dài nhất 7248 ký tự là Điều 219; 25 chunk dài hơn 1500 ký tự). Hai chiến lược còn lại của Day 6 (A fixed 500/50: 421 chunk; C by_clause: 554 chunk) không dùng. Xuất từ Day 6: [data/corpus/index_manifest.json](../../data/corpus/index_manifest.json).
- **Vector store, embedding model, phiên bản và số chiều:** PostgreSQL 16.16 + pgvector 0.8.7 (Docker Day 6, `127.0.0.1:55433`), khoảng cách cosine, score = 1 − distance. Embedding `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers, CPU), **384 chiều**, max_seq_length 128 token, đã normalize, cùng encoder cho query và tài liệu. Kiểm tra index ([results/step1/embedding_check.json](../../results/step1/embedding_check.json)): embed lại 20 chunk mẫu thì cosine nhỏ nhất là 1.0; chạy lại 25 query của Day 6 thì top-3 trùng 25/25.
- **Agent model, cách chạy và phân chia dev/test:** `gemini-3.5-flash-lite` (Gemini API, free tier 5 request/phút), temperature 0. Benchmark `BLLD_45_2019_QH14_day7_v1` gồm 42 câu, đã đóng băng ([data/benchmark/manifest.json](../../data/benchmark/manifest.json)):
  - **dev 30 câu** = 27 có đáp án + 3 thiếu evidence, trong đó 2 multi-hop và 4 có filter. Chỉ dùng dev để chọn cấu hình.
  - **test 12 câu** = 9 có đáp án + 3 thiếu evidence, trong đó 2 multi-hop và 1 có filter. Đây là bộ mới soạn ở Day 7 từ các Điều Day 6 chưa hỏi. 5 câu test cũ của Day 6 đã lộ nên chuyển sang dev. Dev và test không trùng Điều gold nào.
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - *Hit:* chunk thuộc đúng Điều gold **và** chứa trọn chuỗi evidence (sau NFC + lowercase + gộp khoảng trắng). Câu multi-hop phải phủ **mọi** nhóm gold. Context Evidence Rate được kiểm trên **đúng text đã gửi model**, nên nếu excerpt cắt mất evidence thì không tính.
  - *Câu thiếu evidence:* có `absence_check` gồm các mẫu regex quét toàn corpus và các Điều lân cận cần đọc tay; người soạn đã duyệt (`absence_checks_reviewed: true`). Các câu này không vào mẫu số của retrieval metrics.
  - *Chấm tay trên test:* theo [BENCHMARK.md](../../BENCHMARK.md).
    - Context: 0 = không có evidence, 1 = có nhưng thiếu, 2 = đủ.
    - Answer: 0 = sai, thiếu hoặc từ chối; 1 = đúng một phần; 2 = đúng và đủ.
    - Citation: mỗi citation một nhãn đúng/sai, đọc nội dung đoạn đã gửi model chứ không chỉ kiểm tra ID.
  - Agent và retriever chỉ nhận `question` và `filter`, không bao giờ nhận gold, đáp án mẫu hay loại câu.

## Retrieval và thử nghiệm

Điều kiện chung:
- Cùng index B, cùng 27 câu dev có đáp án.
- **Hit@1 và Evidence Hit@3** đo bằng retrieval một lượt với **câu hỏi gốc**, không qua model ([src/eval_retrieval.py](../../src/eval_retrieval.py)).
- **Context Evidence Rate (CER)** báo hai số:
  - *agent:* context cuối cùng agent gửi model, gồm tối đa 4 lượt search do model tự viết query;
  - *1 lượt:* chỉ lượt search đầu, đóng gói bằng starter `pack_context`.
- Mọi cấu hình đều dùng top-k = 3, budget 4000 ký tự, tối đa 4 search, không rerank. Filter là filter của người dùng có sẵn trong câu hỏi (chương La Mã hoặc số Điều).

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | Dense pgvector (MiniLM 384d) + starter `pack_context` (đến trước vào trước, cả chunk hoặc bỏ); top-3; filter người dùng; không rerank; 4000 ký tự | Dùng lại Day 6 nguyên trạng | dev | 12/27 (44%) | 22/27 (81%) | agent 21/27 (78%); 1 lượt 21/27 | retrieval 61.7 ms; agent 83 lượt gọi model, 3579 token/câu, 2.75 s/câu; $0 |
| Phương pháp 2 | BM25 Okapi (k1 = 1.2, b = 0.75, token theo âm tiết) trên cùng 220 chunk; còn lại như baseline | Luật hỏi bằng đúng thuật ngữ ("thời hiệu", "đơn phương", số Điều) nên so khớp từ khóa sẽ thắng dense ở chỗ dense hiểu nghĩa chung chung | dev | 21/27 (78%) | 25/27 (93%) | agent 25/27 (93%); 1 lượt 25/27 | retrieval 1.6 ms; agent 68 lượt gọi, 2858 token/câu, 2.45 s/câu; $0 |
| Phương pháp 3 | Hybrid RRF (k = 60) trên dense top-20 + BM25 top-20 | Gộp hai thứ hạng để bù điểm yếu của nhau | dev | 19/27 (70%) | 25/27 (93%) | 1 lượt 24/27 (89%); **không chạy agent** | retrieval 60.2 ms; $0 |
| Cải thiện 1 | = Phương pháp 2 trong agent: đổi retrieval dense → BM25, giữ nguyên context, prompt và model | Sửa lỗi dev 1 (bên dưới) | dev | 21/27 | 25/27 | agent **25/27**; từ chối nhầm 6/27 → **2/27** | như Phương pháp 2 |
| Cải thiện 2 | BM25 + `excerpt_rr` ([src/context_pack.py](../../src/context_pack.py)): round robin theo hạng qua các lượt search; mỗi chunk tối đa 1500 ký tự, tối thiểu 300; chunk dài được cắt thành excerpt theo query (điểm idf trên từng khối khoản) | Sửa lỗi dev 2 (bên dưới); retrieval không đổi | dev | 21/27 | 25/27 | agent **27/27 (100%)**; từ chối nhầm **1/27** | agent 64 lượt gọi, 2739 token/câu, 2.3 s/câu; $0 |
| Cấu hình đã chọn | = Cải thiện 2 (`bm25_excerpt_B`), đã khóa | Chạy 1 lần sau khi khóa | test | 4/9 (44%) | 8/9 (89%) | agent **8/9 (89%)**; 1 lượt 6/9 | retrieval 2.1 ms; agent 32 lượt gọi, 3573 token/câu, 2.85 s/câu; $0 |
| Baseline | = Baseline (`baseline_dense_B`) | Chạy 1 lần sau khi khóa | test | 3/9 (33%) | 4/9 (44%) | agent 6/9 (67%); 1 lượt 4/9 | retrieval 68.1 ms; agent 37 lượt gọi, 4197 token/câu, 3.9 s/câu; $0 |

Cách đọc bảng:
- *Thời gian:* retrieval = trung vị ms mỗi query; agent = trung vị giây/câu **không tính** thời gian chờ rate limit. Tổng thời gian chờ: dev 869 / 692 / 645 s; test 372 / 327 s.
- *Chi phí:* free tier nên $0; token được báo thay cho tiền.
- *Cải thiện 1 và Phương pháp 2:* là cùng một thay đổi (một phương pháp retrieval mới, đồng thời là cải thiện của agent).
- *Hybrid:* không đưa vào agent vì Hit@1 thấp hơn BM25 (19 so với 21), Evidence Hit@3 bằng nhau, CER 1 lượt thấp hơn (24 so với 25), chậm hơn khoảng 40 lần, và còn phải dùng chung quota free tier.

### Hai lỗi dev, nguyên nhân, cách thử và lý do chọn cấu hình

**Lỗi dev 1 — retrieval (dense bỏ sót Điều đúng):** q17, q19, q24, cùng q03 và q27.
- *Triệu chứng:* baseline tìm đủ 4 lượt rồi từ chối cả 5 câu. Ví dụ q19 (gold Điều 125): context cuối là Điều [47, 46, 40, 204, 199], không có Điều 125.
- *Giả thuyết:*
  - Câu hỏi luật dùng đúng thuật ngữ và số Điều, nhưng MiniLM chỉ đọc 128 token đầu của mỗi chunk B (một Điều dài hơn thế), nên điểm dense bị chi phối bởi nghĩa chung chung.
  - BM25 so khớp đúng từ, nên đưa Điều gold lên hạng 1.
- *Cách thử:* đổi retrieval sang BM25, giữ nguyên mọi thứ khác ([results/step4/agent_dev_summary.json](../../results/step4/agent_dev_summary.json)).
- *Trước → sau:*
  - q17 / q19 / q24: từ chối → trả lời đúng sau 1 search; Điều gold nằm ở hạng 1 hoặc 2.
  - q03 / q27 cũng được sửa, nhưng vì Điều gold có ngay ở hạng 1 của lượt đầu (thứ hạng tốt hơn), không phải nhờ sửa cách đóng gói.
  - Chỉ số: CER 21/27 → 25/27; từ chối nhầm 6/27 → 2/27; lượt gọi model 83 → 68.
- *Còn sót:* q26 và dm02, do lỗi đóng gói (lỗi dev 2).

**Lỗi dev 2 — đóng gói context (đúng Điều đã lấy được nhưng bị loại trước khi tới model):**
- *P1 — chunk dài bị bỏ cả:* starter chỉ nhận cả chunk hoặc bỏ. Ở q26, Điều 188 (3197 ký tự) không còn đủ chỗ trong 4000 ký tự nên bị bỏ cả Điều; BM25 vẫn từ chối sau 4 lượt. Điều 219 (7248 ký tự) thì không bao giờ vào được context.
- *P2 — chiếm chỗ theo thứ tự đến:* ở dm02 (multi-hop: Điều 123 + 128), các chunk của lượt search đầu (gồm Điều 122, 1566 ký tự) chiếm budget, nên Điều 128 tìm được ở lượt sau không vào được. Kết quả: BM25 từ chối, trong khi dense vẫn trả lời được câu này. Đây là một lần BM25 làm hỏng câu mà baseline đúng.
- *Giả thuyết:* lỗi không nằm ở retrieval (Ever retrieved = 27/27) mà ở cách đóng gói.
- *Cách thử:* viết `excerpt_rr`, cùng chữ ký với `pack_context` nên `run_agent` và cách kiểm citation không đổi.
  - Lấy hạng 1 của mọi lượt search trước, rồi mới tới hạng 2…
  - Chunk dài được cắt thành excerpt giữ tiêu đề Điều và các khoản khớp query nhất; chỗ bị bỏ đánh dấu `[…]`.
  - Excerpt chỉ dựa vào query của model và idf của corpus, không dùng gold.
- *Trước → sau:*
  - q26: Điều 188 vào context dưới dạng excerpt 1472 ký tự (khoản 1, điểm a–e) → trả lời đúng.
  - dm02: Điều 122 bị cắt còn 714 ký tự, Điều 128 vào được → trả lời đúng, trích cả [art123] và [art128].
  - Chỉ số: CER 25/27 → 27/27; từ chối nhầm 2/27 → 1/27; token/câu 2858 → 2739.
- *Hồi quy:* q16 chuyển từ trả lời sang từ chối (xem lỗi dev 3).

**Lỗi dev 3 — định dạng citation (để nguyên, có ghi lại):**
- *Triệu chứng:* ở q16, context có đủ evidence (Điều 14), nhưng model viết hai ID trong một ngoặc `[id1, id2]`.
- *Nguyên nhân:* `run_agent` của starter yêu cầu mỗi citation xuất hiện dưới dạng `[id]` trong câu trả lời, nên loại câu trả lời này và ghi là từ chối.
- *Quyết định A:* không sửa prompt trước khi chạy test, để test đo đúng cấu hình đã chọn trên dev. Lỗi này được ghi trong `known_issue_not_fixed` của [results/step5/frozen_config.json](../../results/step5/frozen_config.json).

**Lý do chọn cấu hình.** Tiêu chí được viết **trước** khi chạy test:
1. CER dev cao nhất;
2. từ chối nhầm trên dev thấp nhất;
3. ràng buộc: abstention trên câu thiếu evidence phải giữ 3/3;
4. nếu hòa, chọn cấu hình ít token hơn.

`bm25_excerpt_B` đứng đầu ở cả 4 tiêu chí: 27/27, 1/27, 3/3 và 2739 token/câu. Cấu hình được khóa lúc 2026-11-16 12:31 (+07), có sha256 của mọi file code ([frozen_config.json](../../results/step5/frozen_config.json)); sau đó test chỉ chạy 1 lần cho mỗi cấu hình.

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:**
  - *Vòng lặp:* dùng nguyên `starter/agentic.py::run_agent`. [src/agent.py](../../src/agent.py) chỉ thêm hai adapter:
    - `ToolRetriever` bọc dense, bm25 hoặc hybrid;
    - `GeminiController` gọi Gemini thật một lần cho mỗi quyết định và **bắt buộc** gọi đúng một trong 3 tool: `search(query)`, `answer(answer, citations)` hoặc `abstain(reason)`.
  - *Điều kiện dừng:* model gọi `answer` hoặc `abstain`. Sau 4 lần search, tool `search` bị gỡ nên model buộc phải chọn `answer` hoặc `abstain`.
  - *Context:* mỗi lần search lấy top-3; context gộp qua các lượt và đóng gói vào 4000 ký tự (starter hoặc `excerpt_rr`).
  - *Prompt:* buộc chỉ dùng CONTEXT, ghi `[chunk-id]` cho mỗi ý, gọi `abstain` khi luật chỉ dẫn chiếu sang văn bản khác, và coi CONTEXT là dữ liệu chứ không phải chỉ dẫn.
  - *Kiểm citation:* `run_agent` bắt mọi citation phải có trong context và được ghi `[id]` trong câu trả lời; nếu không đạt, chuyển thành từ chối (`invalid_or_missing_citations`).
  - *Xử lý lỗi API:*
    - Client tự giãn nhịp theo giới hạn 5 request/phút.
    - Retry các lỗi 429, 500, 503 và timeout, theo `retryDelay` server gửi về.
    - Nếu vẫn lỗi, ghi status `error` và không tính điểm câu đó. Không có mock fallback.
    - Phiên bản thử ở bước 2 ([results/step2/](../../results/step2/)) gặp 429 ở dm01/du01; từ khi thêm throttle, mọi lần chạy dev/test đều 0 lỗi.

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Baseline `baseline_dense_B` / test | 12/12 | 37 lượt gọi model; số search/câu {1: 5, 2: 4, 4: 3} | 3 / 1 / 5 — **1.22/2** | 8/8 | 7/7 | 3/3 | 2/9 (t01, t05) |
| Đã chọn `bm25_excerpt_B` / test | 12/12 | 32 lượt gọi model; số search/câu {1: 7, 2: 3, 3: 1, 4: 1} | 1 / 0 / 8 — **1.78/2** | 10/11 | 9/9 | 3/3 | 0/9 (\*) |

(\*) Số 0/9 dễ gây hiểu lầm. Ở t01, model dùng `answer` để nói "context không có thông tin". Status vì vậy là *answered* chứ không phải *abstain*, và chấm tay cho câu này answer = 0.

Context chấm tay (0/1/2), báo cùng các mức như BENCHMARK yêu cầu:
- baseline: 2 / 1 / 6;
- đã chọn: 1 / 0 / 8.

Không có lỗi API hay lỗi vector store ở cả hai lần chạy test.

**Cấu hình đã chọn — test, từng câu** (nhãn: context / answer / citation; ✓ = hỗ trợ, ✗ = không hỗ trợ):

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| t01 | conditional | 4 search: (1) "đơn phương chấm dứt hợp đồng lao động không xác định thời hạn báo trước không hoàn thành công việc" → Điều 36 ở hạng 1 (2677 ký tự, excerpt còn 1495, **mất khoản 2**); (2)–(4) **lặp 3 lần** cùng query "Điều 37 … thời hạn báo trước người sử dụng lao động". Context cuối [36, 35, 38], 3421 ký tự | "CONTEXT hiện tại không chứa thông tin về thời hạn báo trước…" [art036] | **0 / 0 / ✗** — khối khoản 1 (điểm a–d) chiếm hết 1500 ký tự nên excerpt bỏ khoản 2 "Ít nhất 45 ngày"; model từ chối nhưng bằng tool `answer` |
| t02 | paraphrase | 1 search → [47, 46, 41] | "một nửa (1/2) tháng tiền lương… từ đủ 12 tháng" [art046] | 2 / 2 / ✓ — chi tiết "tự ý bỏ việc 05 ngày" model tự suy từ "điểm e khoản 1 Điều 36" (Điều 36 không có trong context) |
| t03 | conditional | 1 search "nghỉ giữa giờ làm việc ban đêm" → [109, 98, 106] | "ít nhất 45 phút liên tục" [art109] | 2 / 2 / ✓ |
| t04 | direct | 1 search → [146, 145, 143] | "không quá 04 giờ/01 ngày và 20 giờ/01 tuần; không làm thêm giờ, ban đêm" [art146] | 2 / 2 / ✓ |
| t05 | paraphrase | 1 search "thời hiệu khởi kiện tranh chấp lao động cá nhân Tòa án" → [190, 187, 191] | "01 năm kể từ ngày phát hiện…" [art190] | 2 / 2 / ✓ |
| t06 | metadata_filter | 1 search, filter `chapter_roman = XIV` → [209, 208, 201] | quốc phòng, an ninh, trật tự công cộng, sức khỏe; danh mục do Chính phủ quy định [art209] ×2 | 2 / 2 / ✓ |
| t07 | cross_reference | 1 search → [169, 219, 149]; Điều 219 (7248 ký tự) thành excerpt 1480 ký tự còn giữ "đủ 20 năm đóng bảo hiểm xã hội" | "ít nhất 20 năm" [art219] | 2 / 2 / ✓ — ghi nhầm vị trí là "điểm a khoản 1 Điều 219" (đúng ra là Điều 54 Luật BHXH được Điều 219 sửa đổi); đáp án vẫn đúng |
| tm01 | multi_hop | 2 search: câu hỏi gộp → [25, 91, 101]; "tiền lương thử việc theo Điều 26" → [26, 24, 41]. Context [25, 26, 91, 24, 101, 41], 3995 ký tự | "không quá 60 ngày" [art025]; "ít nhất 85%" [art026] | 2 / 2 / ✓✓ |
| tm02 | multi_hop | 2 search: "trợ cấp mất việc làm" → Điều 47; "thanh toán các khoản tiền khi chấm dứt hợp đồng lao động thời hạn" → Điều 48. Context [47, 48, 46, 41] | 01 tháng lương/năm, ít nhất 02 tháng [art047]; 14 ngày làm việc, kéo dài tối đa 30 ngày với các trường hợp a–d [art048] | 2 / 2 / ✓✓ |
| tu01 | unanswerable | 3 search về thuế TNCN → [102, 172, 96, 9, 212] | abstain: "BLLĐ 45/2019 không quy định mức thu nhập chịu thuế TNCN" | từ chối đúng — không thêm con số |
| tu02 | unanswerable | 1 search → [17, 14, 18] | abstain: "không quy định mức phạt tiền…" | từ chối đúng — không bịa mức phạt |
| tu03 | unanswerable | 2 search → [139, 141, 46, 168] | abstain: "chế độ thai sản thực hiện theo pháp luật về BHXH, không nêu tỷ lệ %" | từ chối đúng — nêu đúng chỗ dẫn chiếu |

**Baseline — test, từng câu:**

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| t01 | conditional | 4 search (có cả "Điều 36 …") nhưng dense không bao giờ trả về Điều 36 → [45, 31, 35, 39, 50] | abstain | 0 / 0 / — — lỗi retrieval |
| t02 | paraphrase | 1 search → [97, 47, 46] | "một nửa tháng tiền lương" [art046] | 2 / 2 / ✓ (cùng lưu ý "05 ngày" như cấu hình chọn) |
| t03 | conditional | 1 search → [106, 98, 109] | "45 phút liên tục" [art109] | 2 / 2 / ✓ |
| t04 | direct | 1 search → [146, 145, 143] | đúng [art146] | 2 / 2 / ✓ |
| t05 | paraphrase | 1 search → [190, 194, 189] | Model trả lời đúng "01 năm…" nhưng **không ghi `[id]` trong text** (chỉ có trong danh sách citations) → `invalid_or_missing_citations` → từ chối | **2 / 0 / —** — lỗi định dạng citation, giống q16 ở dev |
| t06 | metadata_filter | 1 search, filter XIV → [209, 206, 208] | đúng [art209] ×2 | 2 / 2 / ✓ |
| t07 | cross_reference | 2 search → [169, 149, 168]; không có Điều 219 | "số năm cụ thể theo pháp luật về BHXH" [art169], không nêu số năm | 0 / 0 / ✓ — Điều 169 hỗ trợ đúng những gì model viết, nhưng không trả lời câu hỏi |
| tm01 | multi_hop | 2 search → [155, 26, 97, 25, 123] | 60 ngày [art025]; 85% [art026] | 2 / 2 / ✓✓ |
| tm02 | multi_hop | 4 search (hai lần có "Điều 48"), Điều 48 không bao giờ được trả về → [46, 39, 40, 47, 102] | vế 1 đúng [art047]; vế 2 "CONTEXT không chứa thông tin" | 1 / 1 / ✓ — đúng một phần |
| tu01 | unanswerable | 4 search → abstain | "BLLĐ không quy định mức thu nhập phải nộp thuế TNCN" | từ chối đúng |
| tu02 | unanswerable | 2 search → abstain | "không quy định mức phạt tiền" | từ chối đúng |
| tu03 | unanswerable | 2 search → abstain | "dẫn chiếu sang pháp luật về BHXH" | từ chối đúng |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:**
  - Trace dev: [results/step4/](../../results/step4/) (`*_dev_all.json`, tổng hợp ở `agent_dev_summary.json`).
  - Trace test: [results/step5/baseline_dense_B_test_all.json](../../results/step5/baseline_dense_B_test_all.json) và [results/step5/bm25_excerpt_B_test_all.json](../../results/step5/bm25_excerpt_B_test_all.json).
    - Mỗi trace có: tool call cùng tham số, kết quả search, context sau mỗi lượt, `pack_log`, câu trả lời, citation, token, số lần retry, thời gian chờ và lỗi.
  - Nhãn chấm tay kèm lý do, đã được học viên duyệt: [reviews_baseline_dense_B.json](../../results/step5/reviews_baseline_dense_B.json) và [reviews_bm25_excerpt_B.json](../../results/step5/reviews_bm25_excerpt_B.json).
  - Kết quả tổng hợp: [agent_metrics_baseline_dense_B.json](../../results/step5/agent_metrics_baseline_dense_B.json), [agent_metrics_bm25_excerpt_B.json](../../results/step5/agent_metrics_bm25_excerpt_B.json); chỉ số tự động ở [agent_test_auto_summary.json](../../results/step5/agent_test_auto_summary.json); retrieval test ở [retrieval_test_summary.json](../../results/step5/retrieval_test_summary.json).
  - *Chi phí:* free tier nên $0. Token test: baseline 48 131 prompt + 2 238 output; cấu hình chọn 40 495 prompt + 2 377 output; không có thinking token.
  - *Lệnh chạy* (từ thư mục gốc repo; dense cần Docker Postgres của Day 6 đang chạy; đặt `HF_HUB_OFFLINE=1` để dùng embedder đã cache):
    ```
    python -m src.export_day6 && python -m src.check_embedding           # bước 1: dữ liệu + kiểm tra index
    python -m src.build_benchmark                                          # kiểm tra benchmark (đã đóng băng)
    python -m src.eval_retrieval                                           # bước 3: dense / bm25 / hybrid trên dev
    python -m src.agent --split dev --retrieval dense --context starter    --name baseline_dense_B --out results/step4
    python -m src.agent --split dev --retrieval bm25  --context starter    --name bm25_B           --out results/step4
    python -m src.agent --split dev --retrieval bm25  --context excerpt_rr --name bm25_excerpt_B   --out results/step4
    python -m src.eval_agent results/step4/baseline_dense_B_dev_all.json results/step4/bm25_B_dev_all.json results/step4/bm25_excerpt_B_dev_all.json
    # sau khi khóa (frozen_config.json):
    python -m src.agent --split test --allow-test --retrieval dense --context starter    --name baseline_dense_B --out results/step5
    python -m src.agent --split test --allow-test --retrieval bm25  --context excerpt_rr --name bm25_excerpt_B   --out results/step5
    python -m src.eval_agent --split test results/step5/baseline_dense_B_test_all.json results/step5/bm25_excerpt_B_test_all.json
    python -m src.eval_retrieval --split test                              # chỉ chạy được sau khi khóa; chỉ dense + bm25
    python starter/agent_metrics.py --cases data/benchmark/cases.json --traces results/step5/bm25_excerpt_B_traces.json --reviews results/step5/reviews_bm25_excerpt_B.json --split test --output results/step5/agent_metrics_bm25_excerpt_B.json
    ```
  - *Demo trên máy:* `python -m src.demo_server`, mở http://127.0.0.1:8000. Demo chỉ có câu hỏi dev, key giữ ở phía server.
- **Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:**
  - *Trước → sau trên test* (baseline → cấu hình chọn):
    - grounded answer trung bình 1.22 → 1.78 / 2;
    - CER agent 6/9 → 8/9;
    - Evidence Hit@3 (câu hỏi gốc) 4/9 → 8/9;
    - từ chối nhầm 2/9 → 0/9 (chi tiết ở chú thích \*);
    - token/câu 4197 → 3573, lượt gọi model 37 → 32;
    - abstention trên câu thiếu evidence giữ 3/3.
  - *Từng câu thay đổi:*
    - Ba câu được sửa: t05, t07 (Điều 219 dài 7248 ký tự chỉ vào được context nhờ excerpt — đúng lỗi P1 đã thấy trên dev) và tm02 (BM25 tìm được Điều 48).
    - Không có câu nào bị làm hỏng; t01 sai ở cả hai cấu hình, nhưng vì hai lý do khác nhau.
  - *Lỗi mới do excerpt (t01):*
    - BM25 đã đưa Điều 36 lên hạng 1, nhưng khối khoản 1 (điểm a–d, khoảng 1500 ký tự) chiếm hết phần cap. Excerpt lấy các điểm theo thứ tự văn bản, nên bỏ cả khoản 2 chứa đáp án, dù context lúc đó mới dùng 3421/4000 ký tự.
    - Hướng sửa (**chưa áp dụng vì đã khóa**; phải thử lại trên dev trước):
      - trong một khối quá dài, chọn điểm theo điểm số thay vì theo thứ tự;
      - không áp cap cho chunk hạng 1 khi budget còn trống.
  - *Hai điểm yếu của vòng lặp agent* (cũng thấy ở t01):
    - model lặp lại cùng một query 3 lần dù prompt cấm;
    - model dùng `answer` để từ chối. Bộ lọc `REFUSAL_PREFIXES` chỉ bắt câu mở đầu bằng "không đủ bằng chứng…", nên không bắt được câu "CONTEXT hiện tại không chứa…" và trả về status answered.
  - *Đánh đổi:*
    - Excerpt cho nhiều Điều vừa context hơn và tốn ít token hơn, đổi lại có nguy cơ cắt mất evidence (t01). Chỉ số CER kiểm trên text đã gửi nên vẫn bắt được trường hợp này.
    - BM25 nhanh hơn khoảng 30 lần và không cần DB, nhưng phụ thuộc vào trùng từ. Câu diễn đạt khác thuật ngữ luật thì đẩy evidence xuống thấp: t02 ở hạng 3, cả hai câu multi-hop phải tới hạng 3–4 mới đủ nhóm.
  - *Giới hạn:*
    - **Mẫu nhỏ:** test có 9 câu có đáp án, 1 câu ≈ 11 điểm %; mỗi cấu hình chạy test 1 lần nên không ước lượng được độ dao động.
    - **Không tất định:** temperature 0 vẫn có lúc cho query đầu khác nhau với cùng prompt (thấy trên dev).
    - **Một người chấm:** nhãn do AI soạn, học viên duyệt; không đo độ đồng thuận giữa người chấm.
    - **Kiểm citation quá cứng:** câu trả lời đúng vẫn bị loại vì định dạng (q16 trên dev, t05 baseline trên test); chưa sửa để giữ nguyên cấu hình đã khóa.
    - **Còn thiếu:** chỉ thử chiến lược chunk B; không có rerank; chi phí bản trả phí chưa tính vì không bịa giá.

## Thử thách nâng cao (nếu thực hiện)

Không thực hiện. Bài này chỉ dùng corpus văn bản Bộ luật Lao động 2019 (chỉ có text, không có bảng, ảnh hay biểu đồ cần xử lý riêng). Vì vậy các mục và bảng của thử thách nâng cao được để trống có chủ ý.
