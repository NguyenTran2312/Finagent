# Finagent Query Paraphrasing — thiết kế tích hợp

Ngày: 2026-11-09. Trạng thái: người dùng đã duyệt ngày 2026-11-09.

Người dùng đã chọn phương án A: module điều phối độc lập trước Hybrid Retrieval.
Các chính sách chi tiết ghi dưới đây được đề xuất bởi người triển khai và đã được
người dùng duyệt cùng toàn bộ tài liệu sau bước chọn phương án A.

## 1. Mục tiêu và giới hạn

Chuyển câu hỏi tài chính tiếng Việt và lịch sử tùy chọn thành các câu hỏi độc lập,
phân rã vừa đủ, kiểm tra output rồi truy xuất evidence theo từng subquery.
Bảo toàn doanh nghiệp, metric, thời gian, phạm vi báo cáo, đơn vị và ý định tính toán.
Không tự thêm doanh nghiệp, năm hiện tại hoặc thông tin từ câu trả lời cũ.

Không xây Answer Generation, Calculation Solver, kho lưu hội thoại, parser ngôn ngữ
tổng quát hoặc hệ thống agents chạy lúc phục vụ. Không viết lại BM25, Dense, RRF,
cross-encoder; không thay corpus, database schema hay indexing pipeline.
Không bỏ scope filter để tăng recall. Không coi tests dùng mock là kiểm chứng API LLM.

## 2. Kiến trúc hiện tại đã khảo sát

- `src/query_paraphrase/` là tên thư mục thực tế. `schemas.py` chứa JSON minh họa,
  không phải Pydantic model. `paraphraser.py`, `validator.py`, `__init__.py` trống.
- `prompts.py` là prompt FinSage, còn mặc định Lotus, năm 2024, output ba dòng và
  viết lại tiếng Anh; chưa có call site trong repo ngoài chính module.
- `src/main/query_pipeline.py`, `src/retrieval/multi_query.py` và hai file test
  paraphrasing/multi-query đang trống. Các file khung nói trên đang untracked.
- `refine_question()` trích ticker, năm, scope; mặc định consolidated.
  `retrieve_refined()` lọc BM25 trước khi scoring, giữ IDF/index alignment hiện tại.
- Hybrid nhận string hoặc mapping `question`/`filter_question`; chuyển cùng bộ lọc
  sang Dense rồi RRF. Reranker chuyển nguyên mapping xuống Hybrid.
- Metadata corpus có ticker, report_year, scope. Scope chứa consolidated, separate,
  explanations và letters; parent là alias legacy cho separate trong retrieval.
  Chưa có metadata riêng cho quý/bán niên hoặc report_scope của explanations/letters.
- `data/code_stock.csv` có 100 ticker/tên đầy đủ, không có cột aliases.
- Tests hiện có dùng unittest và mocks; một số PostgreSQL integration tests cần
  cấu hình riêng. Một số import dùng `main`/`retrieval`, một số dùng `src.*`.
  Không mở đợt refactor imports ngoài phạm vi để thống nhất chúng.

## 3. Phương án đã cân nhắc

**A — được người dùng chọn:** Paraphraser và validator tạo kế hoạch; multi_query
điều phối từng subquery qua retriever hiện tại với filter riêng. Ưu điểm: tách ranh
giới, mock dễ, bảo toàn output của Hybrid và reranker. Đổi lại có thêm lớp điều phối.

**B — không chọn:** cho Hybrid nhận cả kế hoạch, tự phân nhánh và trả evidence nhóm.
Ít lớp hơn nhưng làm Hybrid kiêm validation/orchestration, phức tạp compatibility.

## 4. Luồng và public interfaces

Luồng mới:

Original question + qa_history → prompt + injected LLM → Pydantic output → code
validation [+ optional semantic reviewer] → validated plan → per-subquery filters
→ BM25 + Dense → RRF → reranking → evidence grouped by subquery.

- `QueryParaphraser(llm, ..., semantic_reviewer=None).paraphrase(question,
  qa_history=None)` trả `ParaphraseResult`.
- LLM được inject dưới dạng callable nhận prompt và trả JSON string, mapping hoặc
  Pydantic plan. Chỉ một lần gọi mỗi request, không retry/repair tự động.
  Một adapter nhỏ cho LangChain `with_structured_output(QueryPlan)` có thể đặt trong
  `paraphraser.py`; caller chịu trách nhiệm tạo client, timeout và retry hữu hạn.
  Unit tests inject callable thuần, không import/load model nặng.
- `retrieve_subqueries(plan, retriever, ..., policy=...)` trả danh sách evidence
  groups; chỉ nhận kế hoạch đã được validator chấp nhận trong request hiện tại.
- `build_query_pipeline(paraphraser, retriever, ...)` trong `query_pipeline.py`
  là entrypoint tiện dụng: invoke string hoặc mapping `question`/`qa_history`.
  Caller truyền retriever đã bọc reranker nếu cần; không tự load embeddings/reranker.
- Pipeline luôn gọi validator trong request hiện tại trước orchestration. Helper
  retrieve_subqueries không nhận raw LLM output: nhận ParaphraseResult đã được code
  tạo, kiểm tra status/plan trước mọi call. Đây là precondition API nội bộ, không
  phải cơ chế chứng thực output qua mạng hay quyền bỏ qua validation của caller.
- Orchestration chạy tuần tự để giới hạn tài nguyên, giữ thứ tự subquery. Không có
  parallel execution/caching mới trong tính năng này.

## 5. Schema Pydantic v2

Giữ tên `original_question`, `operation`, `subqueries`, `id`, `search_query`,
`ticker`, `year`, `scope`, `metric` từ JSON minh họa. Chưa có callable schema API
được các module khác sử dụng; bổ sung model không phải migration của một API chạy được.

Các model dùng extra=forbid, kiểu dữ liệu chặt chẽ, cấm chuỗi trắng; không coercion
bool/string thành year. Trường nullable phải xuất hiện trong JSON; giá trị chưa biết
là null. Những danh sách tùy chọn có default rỗng rõ ràng.

`QueryPlan`:

- original_question: chuỗi gốc, phải khớp input.
- operation: lookup | compare | aggregate | calculate; chỉ thể hiện ý định.
- operation_description: chuỗi nullable giữ phép toán/điều kiện bằng ngôn ngữ tự nhiên.
- subqueries: list[Subquery], không rỗng, IDs duy nhất.
- relationships: list quan hệ có kind, ordered subquery_ids và description nullable.
  IDs phải tồn tại; giữ thứ tự operands khi phép tính không giao hoán.

`Subquery`:

- id, question (câu hỏi độc lập tiếng Việt), search_query (tiếng Việt).
- ticker, company_name: nullable; đối chiếu registry CSV, không tự tạo ticker.
- year: int nullable, khoảng 1900–2199, cùng miền extractor hiện tại.
- period: nullable string giữ mô tả quý/bán niên/lũy kế/kỳ tài chính; không tự mặc định FY.
- scope: consolidated | separate | null, chỉ là phạm vi báo cáo được hỏi.
- metric: canonical ID nullable; metric_text giữ diễn đạt gốc cho metric ngoài catalog.
- unit: nullable string giữ đơn vị yêu cầu, không quy đổi số liệu.
- document_types: list[financial_statements | explanations | letters].
  Rỗng nghĩa là áp dụng retrieval policy, không phải tìm mọi loại tài liệu.
- context_refs: mapping tên trường được kế thừa → chỉ số history turn (zero-based).
  Chỉ các trường ticker/company_name/year/period/scope/metric/unit được tham chiếu.
  Mapping rỗng khi không dùng history.

`ParaphraseResult` do code tạo, không cho LLM tự xác nhận:

- status: valid | needs_clarification | invalid | unavailable.
- plan: nullable; plan không hợp lệ không được chuyển sang retrieval.
- issues: list(code, message, subquery_id nullable).
- semantic_status: not_run | passed | failed | unknown.

`valid` chỉ nghĩa là đã qua kiểm tra code đã định nghĩa, không chứng minh ngữ nghĩa
toàn diện. semantic_status=passed vẫn chỉ là đánh giá của reviewer, không phải bằng
chứng kế toán. Không dùng status do LLM sinh làm quyền bỏ qua validator.

## 6. Prompt, history và metric catalog

Thay prompt FinSage bằng prompt tài chính Việt Nam: giữ tiếng Việt, không mặc định
doanh nghiệp/năm, không over-decompose một nhu cầu thống nhất, giữ constraints và
phép toán. Dùng JSON/Pydantic schema, bỏ parse ba dòng và các prompt Answer Generation
Lotus không có call site. Không đưa thời gian hiện tại vào prompt như năm tham chiếu.

History là list `QATurn(question: str, answer: str | None)`, tùy chọn; mặc định tối đa
5 turn gần nhất, cắt toàn bộ turn cũ hơn, không cắt giữa chuỗi. Context refs sử dụng
index gốc trong list. Mặc định question tối đa 8.000 ký tự, mỗi trường history tối đa
8.000 ký tự; quá giới hạn trả lỗi input, không âm thầm cắt mất ràng buộc.

Chỉ nội dung câu hỏi của người dùng được dùng làm căn cứ kế thừa; answer được đánh
dấu là nội dung chưa xác minh và không dùng làm nguồn kiểm chứng metadata. Prompt
phân biệt chỉ dẫn với question/history là dữ liệu. Không nhận chỉ dẫn nằm trong history.
Chỉ kế thừa trường bị thiếu khi có tham chiếu tiếp nối rõ và context_refs có nguồn
khớp; constraints hiện tại thắng history. Nhiều antecedent hợp lý hoặc không xác định
được quan hệ thì needs_clarification; không quét toàn history để chọn tên/năm bất kỳ.
Phiên bản đầu hỗ trợ các mẫu tiếp nối phổ biến ("còn năm 2023?", "công ty đó",
"chỉ tiêu này"); không tuyên bố giải quyết được mọi tham chiếu tự nhiên.

Thêm `metric_catalog.py`: bảng canonical ID, nhãn tiếng Việt, aliases khai báo rõ.
Khởi đầu gồm revenue, net_revenue, profit_before_tax, profit_after_tax,
consolidated_profit_after_tax, profit_after_tax_parent_share; bổ sung metric phổ biến
nếu tests/yêu cầu có căn cứ. Không gộp doanh thu với doanh thu thuần, LNTT với LNST,
LNST hợp nhất với LNST thuộc cổ đông công ty mẹ. Match cụm dài/cụ thể trước cụm ngắn,
không coi cụm "cổ đông công ty mẹ" là yêu cầu scope separate.
Metric ngoài catalog giữ metric=None và metric_text, không ép về metric gần giống.

## 7. Validation và failure handling

Structural: parse JSON trực tiếp/Pydantic; không cắt markdown để cứu output. Cấm
subquery trống, ID lặp, cùng query/constraints lặp vô nghĩa, vượt max_subqueries
(mặc định 8, cấu hình được). Kiểm tra relations/operation và kiểu các trường.

Entities: đọc CSV UTF-8, kiểm tra header/ticker trùng. Nhận ticker và tên đầy đủ sau
chuẩn hóa case/whitespace/Unicode. Cho phép alias registry được inject, alias phải
trỏ ticker có trong CSV; alias collision giữ danh sách candidates và yêu cầu làm rõ.
Không tự suy ra tên thương mại, không fuzzy-match, không dùng việc ticker không
tồn tại để chọn ticker khác. Ticker có trong CSV cũng phải có căn cứ từ câu hỏi hoặc
history hợp lệ, không đủ chỉ vì LLM chọn một ticker có thật. Không có doanh nghiệp
trong câu hỏi là trường chưa xác định; doanh nghiệp được nêu nhưng mơ hồ/không hợp lệ
là needs_clarification và chặn retrieval, tránh tìm toàn corpus do mất filter.

Financial checks bằng code:

- Đối chiếu recognized metric aliases, ticker, explicit years/year ranges, period,
  scope, unit và dấu hiệu phép toán giữa input, subqueries, metadata và search text.
- Kiểm tra coverage của các nhu cầu/ràng buộc nhận diện được; bắt mất năm/metric/
  doanh nghiệp, đổi nghĩa metric, thêm constraint không có căn cứ, sai cặp ràng buộc
  khi cấu trúc cặp trong câu hỏi có thể kiểm chứng bằng các mẫu hỗ trợ.
- Không áp toàn bộ years/tickers lên mỗi subquery. So sánh VJC 2022/2023 phải có hai
  subqueries tương ứng; câu hỏi VJC 2022 và HPG 2023 không được đổi thành tích Descartes.
- Không suy scope separate chỉ vì metric thuộc cổ đông công ty mẹ. Phân biệt null
  do không được hỏi với null làm rơi constraint explicit: trường hợp sau là lỗi.
- Với cách diễn đạt/quan hệ không kiểm chứng được, ghi rõ giới hạn. Nếu việc chưa
  xác minh khiến không thể chọn metadata an toàn, trả needs_clarification.

Code không chứng minh được toàn bộ tính đầy đủ/binding của decomposition tự nhiên.
Optional semantic_reviewer được inject riêng, nhận original/history/plan và trả
passed/failed/unknown cùng lý do. Failed hoặc unknown khi reviewer đã được cấu hình
thì chặn retrieval. Không cấu hình reviewer thì semantic_status=not_run và công khai
giới hạn; không thêm một cuộc gọi LLM thứ hai mặc định.

JSON/schema/metadata mâu thuẫn: invalid, không invoke retrieval. Entity/context mơ hồ:
needs_clarification. LLM timeout/lỗi provider: unavailable với message đã lọc, không
đưa API key/raw provider payload vào kết quả. Không fallback sang kế hoạch bị lỗi hoặc
gọi retrieval trực tiếp trên output chưa kiểm tra. Lỗi lập trình/configuration không
bị catch-all thành kết quả có vẻ thành công. Các failure này là fallback an toàn;
không cần xây deterministic paraphraser thứ hai.

## 8. Structured filters và backward compatibility

Bổ sung model `MetadataFilters` độc lập với LLM schema tại
`src/retrieval/filters.py`: tickers, years, scopes là các list được kiểm tra kiểu và
giá trị. Đây là filter vật lý; scopes chỉ chứa consolidated/separate/explanations/
letters (parent được chuẩn hóa thành separate). Không chứa metric/unit/period vì
corpus chưa có metadata tương ứng. List rỗng cho field chưa biết giữ nghĩa unrestricted
chỉ ở field đó; policy luôn tạo scopes không rỗng cho luồng mới.

Interface mới:

    retriever.invoke({"question": subquery.search_query,
                      "metadata_filters": {"tickers": ["VJC"],
                                           "years": [2022],
                                           "scopes": ["consolidated"]}})

`retrieve_refined()` có keyword metadata_filters; Hybrid chuyển nó xuống và dùng
cùng normalized filters dựng Chroma where. Nếu có metadata_filters thì không gọi
refine_question để suy lại constraints từ search text. Cấm truyền đồng thời
metadata_filters và filter_question để tránh precedence mơ hồ; mapping sai/unknown
field bị từ chối trước khi truy xuất. Không mutate input hoặc source Documents.

Giữ nguyên invoke(string), invoke({question, filter_question}), output list[Document],
default consolidated, parent alias, BM25 scoring, RRF identity và reranker behavior.
Reranker vốn pass mapping xuống dưới; chỉ cần sửa nếu tests chứng minh thiếu hỗ trợ.
Không áp catalog 100 ticker lên API legacy đang hỗ trợ corpus/test tickers khác.

## 9. Report scope, document types và thuyết minh

Policy được inject/cấu hình, mặc định:

- Câu hỏi không yêu cầu scope: subquery.scope vẫn null; primary retrieval dùng
  default_report_scope=consolidated. Result ghi effective scope và nguồn policy,
  không nói người dùng đã yêu cầu hợp nhất. Không mặc định cả hai phạm vi.
- Scope explicit: primary chỉ dùng đúng consolidated hoặc separate.
- document_types rỗng: primary financial_statements; fallback explanations một lần
  khi primary không có evidence. Câu hỏi yêu cầu thuyết minh: tìm explanations ngay,
  không chờ primary rỗng; câu hỏi yêu cầu cả hai thì chạy cả hai.
- letters chỉ truy xuất khi được yêu cầu rõ; không tự coi là thuyết minh BCTC.
- Mọi lần truy xuất đều giữ ticker/year riêng; notes/letters dùng filter category
  riêng, không OR chung vào primary và không bỏ scope filter.
- explanations/letters hiện không xác minh được consolidated/separate từ metadata.
  Trả trong supplemental_evidence, scope_verified=false, report_scope=null;
  không trộn vào primary evidence hoặc gắn nhãn phạm vi theo yêu cầu người dùng.
  Nếu người dùng yêu cầu thuyết minh hợp nhất/riêng lẻ, result là partial dù có notes.
  Không đoán scope từ filename, ticker hay LLM. Xác minh scope notes về sau là việc
  riêng nếu có nguồn dữ liệu tin cậy; không nằm trong thay đổi indexing lần này.
- Khi fallback hoặc notes không tìm được evidence, trả nhóm rỗng và status empty,
  không tự mở rộng ticker/year/scope. Fallback không kích hoạt khi retrieval lỗi.

## 10. Kết quả điều phối và lỗi truy xuất

Pipeline result gồm paraphrase result và evidence_groups theo thứ tự plan.
Mỗi group giữ subquery đầy đủ, filters thực tế, primary evidence, supplemental
evidence, warnings và status ok | partial | empty | error. Documents giữ nguyên
metadata và scores hiện có; wrapper giữ nhãn scope verification, không sửa corpus.

Rerank riêng từng lần truy xuất bằng search_query tương ứng. Không RRF giữa các
subqueries, không tính comparison/aggregation và không dedup mất quan hệ giữa nhóm.
Notes là nhóm phụ độc lập, không so sánh điểm reranker giữa primary và supplemental.

Period được giữ trong câu hỏi tìm kiếm và metadata subquery nhưng không được corpus
filter xác nhận. Với yêu cầu quý/bán niên/lũy kế, group ghi period_verified=false
và warning; report_year trùng không được coi là đã chứng minh đúng kỳ. Tương tự,
unit được bảo toàn như yêu cầu, chưa xác nhận đơn vị của từng giá trị evidence.
Đây là giới hạn retrieval, không phải giấy phép làm mất period/unit trong rewrite.

Retrieval lỗi ở một nhóm: group=error và overall=partial hoặc error nếu toàn bộ
nhóm lỗi; các nhóm khác vẫn được chạy. Không trả lỗi thành empty hoặc tự nới filter.
Overall successful retrieval có status ok/partial/empty, phân biệt với trạng thái
paraphrase valid. Không trả overall=ok nếu có error/partial group. Không phát sinh
calls retrieval nào khi paraphrasing không valid.

## 11. Cấu hình, files và validation strategy

`config/settings.py`: cấu hình paraphrasing được đọc khi tạo module, hỗ trợ override
constructor; max_subqueries=8, max_history_turns=5, input limits ở trên,
default_report_scope=consolidated, notes_fallback=true. Reject config không hợp lệ.
Không hardcode credentials, provider endpoint hoặc bắt người chạy tests có API key.
Client/model/provider do caller inject; không tự chọn provider cho dự án.

Files dự kiến bổ sung/hoàn thiện:

- `src/query_paraphrase/{schemas,prompts,paraphraser,validator,__init__}.py`.
- `src/query_paraphrase/metric_catalog.py`, `src/query_paraphrase/entities.py`.
- `src/retrieval/filters.py`, `src/retrieval/multi_query.py`.
- `src/main/query_pipeline.py`, thay đổi nhỏ `src/main/refine_questions.py`,
  `src/retrieval/hybrid.py`, `config/settings.py`; reranker chỉ khi cần.
- `tests/test_query_paraphrasing.py`, `tests/test_multi_query.py` cùng tests filters/
  catalog/entities nếu tách giúp giữ test rõ; mở rộng regression tests hiện có.
- Hướng dẫn cấu hình, ví dụ DI và giới hạn ở tài liệu tính năng; không đổi requirements
  nếu dependencies hiện tại đã đủ.

Acceptance tests không cần GPU/API/database: simple lookup; cùng metric nhiều năm;
nhiều metric; nhiều doanh nghiệp; cặp ticker-year không được nhân chéo; scopes explicit;
profit parent-share không thành separate; notes và letters không trộn primary;
ambiguous/invalid entities; context tiếp nối và unrelated/incorrect answer history;
JSON/schema lỗi; metadata/search text mâu thuẫn; omission/addition of constraints;
duplicate/limit/relations/unit/period/operation validation; LLM và retrieval failures.

Integration dùng BM25 in-memory thật, Dense store và cross-encoder giả có recording
calls; assert bộ lọc hai nhánh, RRF/rerank, evidence groups và không có retrieval cho
output lỗi. Regression giữ string/filter_question API, default consolidated, parent
alias, index alignment, RRF/source immutability và reranker input forwarding.

Chạy baseline bằng Python .venv hiện có, với PYTHONPATH hỗ trợ root và src trong
worktree. Sau đó RED → GREEN → REFACTOR từng task, review spec/quality trước task kế.
PostgreSQL/API/GPU checks được đánh dấu rõ nếu không có môi trường; chỉ báo cáo các
kết quả thực chạy. Có failure dùng systematic-debugging trước khi sửa.

## 12. Quy trình tiếp theo và phần giao người triển khai quyết định

Sau duyệt thiết kế: writing-plans tạo plan có paths/dependencies/tests; tiếp theo
using-git-worktrees tạo nhánh codex/ riêng, bảo toàn khung untracked bằng bản sao
có kiểm tra nội dung, không stash/xóa file gốc; baseline trước implementation.
Triển khai bằng subagent-driven-development và TDD, review từng task, review toàn bộ
theo yêu cầu người dùng và AGENTS.md, cuối cùng verification-before-completion.
Reviewer chỉ đọc, không sửa code trong review.

Người triển khai được quyết định tên helper/private classes, chia nhỏ test files,
thông điệp lỗi, tổ chức fixtures và mẫu regex cụ thể trong phạm vi đã ghi; không tự
thêm semantic service, database fields hoặc thay policy scope đã duyệt.

Tài liệu này là thiết kế, chưa phải phần đã triển khai. Chưa chạy baseline/tests,
chưa kiểm thử API LLM và chưa cam kết chất lượng retrieval trên corpus thực tế.

## 13. Kiểm tra khả năng lập kế hoạch

Một subagent mới đã đọc thiết kế và đối chiếu code ở chế độ chỉ đọc. Kết quả:
"nothing important left" — không còn câu hỏi quan trọng bắt buộc phải trả lời để
lập implementation plan. Đây là kiểm tra độ đầy đủ của thiết kế, không thay thế
code review hoặc kết quả kiểm thử phần triển khai.
