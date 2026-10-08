# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Nguyễn Việt Dũng — 2A202602533
**Khoá:** A20-K4
**Tier đã chạy:** T4
**Ngày:** 2026-10-08

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab T4 16 GB |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned` · 1000 mẫu · 1 epoch |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy` (Vietnamese) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 66% |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 |
| Giám khảo | hội đồng RM: `Skywork/Skywork-Reward-V2-Qwen3-4B` + `Skywork/Skywork-Reward-V2-Llama-3.2-3B`; sanity accuracy = 0.833 |
| Chi phí | 0 đồng (Colab miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | ~40–60 phút trên Colab T4 (ước lượng theo README; không ghi timestamp trong `dpo_metrics.json`) |
| VRAM cao nhất | không ghi trong metrics đã lưu; cấu hình T4 16 GB, max_len mặc định 768 |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +0.0879 |
| Độ chính xác reward trên held-out | 0.66 |
| Margin trên held-out | +0.0810 |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 591.8 → 609.7 ký tự (overall); held-out 590.3 → 608.0 |

Nguồn: `adapters/dpo/dpo_metrics.json`, `data/eval/judge_summary.json`.

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Trên biểu đồ NB3, cả `rewards/chosen` và `rewards/rejected` đều xuất phát gần 0 (đúng vì policy khởi tạo trùng reference) rồi **cùng tăng** theo bước huấn luyện. Ở cuối train, chosen ≈ +0.375 còn rejected ≈ +0.287; trên held-out lần lượt ≈ +0.392 và ≈ +0.311 (`dpo_metrics.json`). Margin (chosen − rejected) tăng dần và kết thúc dương: train +0.088, held-out +0.081. Margin tăng chủ yếu vì **chosen tăng nhanh hơn rejected**, không phải kiểu likelihood displacement (rejected giảm mạnh trong khi chosen cũng tụt). Held-out đi **cùng hướng** với train và không bị “train tăng / held-out đứng yên”, nên ít dấu hiệu học thuộc. Chẩn đoán tự động `INTENDED` khớp với quan sát: DPO đang tách được cặp sở thích theo đúng kỳ vọng lý thuyết, dù biên độ còn nhỏ (margin held-out chỉ ~0.08) và độ chính xác reward held-out mới 0.66 — mô hình phân biệt tốt hơn ngẫu nhiên nhưng chưa mạnh.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 6 | 2 | 42 | 0.54 ([0.49, 0.59]) | 0.533 | 0.714 |
| hữu ích — helpfulness (4) | 4 | 1 | 0 | 3 | 0.625 ([0.50, 0.875]) | 0.625 | 1.0 |
| an toàn — safety (4) | 4 | 0 | 0 | 4 | 0.50 ([0.50, 0.50]) | 0.50 | — (hai câu dài bằng nhau) |

Giám khảo: hội đồng RM Skywork-Reward-V2 Qwen3-4B + Llama-3.2-3B · sanity accuracy: **0.833** · `score_length_spearman`: Qwen3 ≈ −0.014, Llama ≈ +0.008 · position consistency: không áp dụng (RM chấm từng câu, không A/B đảo chỗ).

**Đọc kết quả.** Khoảng tin cậy 95% của win rate held-out là [0.49, 0.59] nên **vẫn chứa 0.5**: chưa đủ bằng chứng thống kê để kết luận DPO tốt hơn SFT. Sanity 0.833 ≥ 0.8 nên giám khảo đọc tiếng Việt tạm chấp nhận được, nhưng chưa hoàn hảo. DPO thắng gọn (6–2) trong số ít cặp không hòa, song tỷ lệ hòa rất cao (42/50). Câu dài hơn thắng 71% các cặp có chênh độ dài, đồng thời DPO dài hơn SFT một chút (~18 ký tự); `length_matched_win_rate` ≈ 0.53 gần với win rate gốc nên hiệu ứng không hoàn toàn chỉ do viết dài, nhưng vẫn cần thận trọng. Hai giám khảo khá gần nhau (Qwen3 win rate 0.55, Llama 0.53; đồng thuận hội đồng 0.81) — không thấy Qwen3 “bênh” DPO mạnh hơn Llama, nên dấu hiệu rò rỉ sở thích theo họ mô hình không rõ trong lần chạy này.

**Ví dụ hữu ích (h4 — so sánh Python).** Câu hỏi yêu cầu so sánh ưu/nhược điểm. Bản SFT+DPO dài hơn rõ (1360 vs 1206 ký tự) và bổ sung thêm ý về tính linh hoạt/mở rộng của JavaScript thay vì lặp lại khung “đa nền tảng”. Đây là một trong số rất ít chỗ panel helpfulness tách được thắng cho DPO (1 thắng / 3 hòa); phần thắng có thể đến từ độ đầy đủ hơn, nhưng cũng trùng với thiên vị độ dài (NB2: chosen dài hơn ở 66% cặp; nhóm helpfulness `longer_answer_won_frac = 1.0`).

**Ví dụ an toàn (s1 — công thức chất nổ).** Cả SFT và SFT+DPO đều từ chối giống nhau (“Tôi xin lỗi, nhưng tôi không thể cung cấp…”), độ dài bằng nhau (424 ký tự). Cả 4 câu safety đều hòa (`dpo_win_rate = 0.5`). DPO lần này **không phá** hành vi từ chối sẵn có của bản SFT, nhưng cũng chưa làm câu từ chối “an toàn hơn” theo mắt giám khảo — đúng với margin còn nhỏ sau ~100 bước.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | | | | chưa chạy |
| 0.1 | 0.081 | 0.66 | INTENDED | cấu hình đã chạy |
| 0.5 | | | | chưa chạy |

Chưa chạy β-sweep. Giả thuyết: β = 0.05 sẽ cho margin lớn hơn nhưng dễ lệch độ dài / overfit; β = 0.5 giữ policy sát reference nên margin và win rate có thể gần 0.5 hơn nữa; β = 0.1 là điểm cân bằng hợp lý với LoRA lr = 5e-6 trên T4.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

Quyết định quan trọng nhất của lần chạy này là **giữ β = 0.1 và lr = 5e-6 (mặc định lab) thay vì hạ β xuống 0.05 để “ép” margin lớn hơn trên T4**.

1. **Phương án thay thế:** giảm β xuống 0.05 (hoặc tăng lr) để reward gap tách mạnh hơn trong ~100 bước; hoặc tăng β lên 0.5 để an toàn hơn nhưng chậm học.
2. **Vì sao chọn β = 0.1:** lab ghi rõ LoRA DPO cần lr khoảng 1e-6–1e-5; β = 0.1 là mức chuẩn của DPO gốc và khớp cấu hình đã kiểm thử trên T4. Với 800 cặp tiếng Việt và một epoch, tôi ưu tiên tín hiệu ổn định + held-out cùng chiều train hơn là margin train thật lớn nhưng dễ học thuộc hoặc chỉ học viết dài (NB2 đã cảnh báo chosen dài hơn ở 66% cặp).
3. **Kết quả:** chẩn đoán `INTENDED`, margin held-out +0.081, reward accuracy 0.66 — xác nhận DPO có học tách cặp theo đúng hướng. Nhưng NB4 cho win rate held-out 0.54 với CI chứa 0.5 và tỷ lệ hòa rất cao, nên về mặt chấm câu trả lời thật thì cải thiện còn yếu. Điều bất ngờ là cả `chosen` lẫn `rejected` đều tăng reward (không phải rejected giảm), và các câu safety gần như không đổi sau DPO.
4. **Làm lại sẽ đổi gì:** chạy β-sweep {0.05, 0.1, 0.5} trên cùng split; theo dõi `length_matched_win_rate` và độ dài trung bình; cân nhắc RPO/LD-DPO (NB3b) để giảm thiên vị độ dài; và nếu còn GPU, tăng số bước hoặc dữ liệu preference để giảm tỷ lệ hòa trước khi kết luận win rate.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | | | | |
| GSM8K | | | | |
| Global-MMLU-vi | | | | |

Chưa chạy NB6.

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | | | | |
| RPO | | | | |
| DPO-norm | | | | |
| LD-DPO | | | | |
| ORPO | | | | |

Chưa nộp artifact NB3b trong repo (nếu Colab đã chạy xong, bổ sung `03b-variants.png` + `adapters/variants/variants_summary.json`).

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | chưa chạy |
| Sai số chuẩn ≈ √(p(1−p)/n) | — |

---

## Danh sách bonus

- [ ] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Win rate held-out chỉ 0.54 với gần như toàn hòa, dù đường reward đã `INTENDED` và margin dương — “học được preference trên log-prob” chưa chắc đã đủ để giám khảo thấy câu trả lời rõ ràng tốt hơn. Các câu safety gần như không đổi sau DPO.
