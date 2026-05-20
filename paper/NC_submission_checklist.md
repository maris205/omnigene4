# Nature Communications 投稿步骤清单

## 投稿入口

https://mts-ncomms.nature.com (Manuscript Tracking System)

如果没有账号，先用 wangliang.f@gmail.com 注册，然后激活邮箱链接登录。

---

## 投稿前检查（5 分钟）

- [ ] 主文 PDF: `paper/omnigene4.pdf` (25 页, 776KB) ✓ 已编译
- [ ] Supplementary PDF: `paper/supplementary.pdf` (15 页, 370KB) ✓ 已编译
- [ ] Cover letter ASCII 文本: `paper/NC_cover_letter_ASCII.txt` ✓ 准备好
- [ ] Abstract 纯 ASCII: `paper/Abstract.md` ✓ 准备好
- [ ] LaTeX 源文件 + 图: 备份用，NC 投稿不强制要求
- [ ] bioRxiv 链接: https://doi.org/10.1101/2026.01.03.697478 ✓ 已上线
- [ ] GitHub repo public: https://github.com/maris205/omnigene4 ✓ 已 push 最新版

---

## NC 投稿表单填写步骤

### Step 1: Article Type
- 选择 "Article" (主类别，长文)
- 不选 "Brief Communication" / "Comment" / "Matters Arising"

### Step 2: Title and Abstract
- **Title**: `OmniGene-4: A Unified Bio-Language MoE Model with Router-Level Interpretability`
- **Running Title** (短版，~50 chars): `OmniGene-4: bio-MoE with router interpretability`
- **Abstract**: 复制 `Abstract.md` 的整段（已纯 ASCII，约 180 词）
- **Keywords**: bioinformatics, foundation model, mixture-of-experts, interpretability, protein homology, continued pretraining, instruction tuning

### Step 3: Authors and Affiliations
- 单作者: Liang Wang
- Affiliation: School of Artificial Intelligence and Automation, Huazhong University of Science and Technology, Wuhan 430074, China
- Corresponding author: 勾选
- ORCID: 如果有先填

### Step 4: Manuscript Files
按 NC 要求顺序上传:
1. **Main Manuscript** = `omnigene4.pdf` (file type: "Manuscript")
2. **Supplementary Information** = `supplementary.pdf` (file type: "Supplementary Information")
3. (可选) **LaTeX source** = `omnigene4.tex` + `refs.bib` + `figures/*.pdf` 打包成 zip (file type: "Source File")

### Step 5: Cover Letter
直接粘贴 `NC_cover_letter_ASCII.txt` 中 "COVER LETTER" 段。

### Step 6: Editorial Importance Statement (60 words)
粘贴 `NC_cover_letter_ASCII.txt` 中 "EDITORIAL IMPORTANCE STATEMENT" 段。

### Step 7: Suggested Reviewers (need at least 4)
逐一填入 `NC_cover_letter_ASCII.txt` 中 "RECOMMENDED REVIEWERS" 列表:
- Bonnie Berger (bab@mit.edu)
- Burkhard Rost (assistant@rostlab.org)
- Sergey Ovchinnikov (so3@mit.edu)
- Mohammed AlQuraishi (ma4129@columbia.edu)

### Step 8: Non-preferred Reviewers
留空（无非偏好审稿人）

### Step 9: Conflict of Interest / Competing Interests
选 "The author declares no competing interests"

### Step 10: Funding
填入 io.net 致谢:
"GPU compute supported by io.net (~160 GPU-hours). No additional grant funding."

### Step 11: Data Availability Statement
"All data, code, and model weights are publicly available:
- Source code: https://github.com/maris205/omnigene4
- Models: https://huggingface.co/dnagpt
- Dataset: https://huggingface.co/datasets/dnagpt/biopaws
- bioRxiv preprint: https://doi.org/10.1101/2026.01.03.697478"

### Step 12: Code Availability Statement
"Code at https://github.com/maris205/omnigene4 under Apache 2.0 license.
Six pre-trained model variants and the BioPAWS dataset are released on Hugging Face under https://huggingface.co/dnagpt"

### Step 13: Submission Statement / Originality
- [ ] 勾选 "The work has not been submitted elsewhere"
- [ ] 勾选 "All authors have approved this version"
- [ ] bioRxiv preprint disclosure: 填 DOI 10.1101/2026.01.03.697478

### Step 14: Final Review and Submit
- 系统会自动生成一个 merged PDF（主文+supp+cover letter）供你 proof-check
- 仔细看一遍 PDF，确认所有图都嵌入正确、引用完整
- 点 "Approve" 提交

---

## 提交后

### 立即处理:
- 收到 confirmation email 后，记录 submission ID（格式如 NCOMMS-26-12345）
- 把 submission ID 记到这个文档里:
  - Submission ID: __________
  - Submission Date: __________

### 7-10 天内可能收到:
- **Reject without review** (~40%): 编辑桌拒，立即用 NC 内部 transfer 工具转 Communications Biology
- **Send for review** (~50%): 进入审稿，等待 6-8 周
- **Editorial query** (~10%): 编辑问问题，2 天内回复

### 审稿进入后:
- 通常给 3 个外审，6-8 周回复
- 收到 reviewer comments 后通常给 30-60 天 revision 期

---

## 如果被拒（最坏情况）

NC 系统右上角有 "Manuscript Transfer" 按钮，可以直接 transfer 到:
1. **Communications Biology** (推荐, IF 5+, 接受率 50%+)
2. **Scientific Reports** (备选, IF 4+, 接受率 60%+)

Transfer 时审稿意见会一并传过去，新刊编辑会基于 NC 审稿意见决策，省去重新排队 + 重新审稿的时间。

---

## 重要提醒

- **不要**在 submission 期间修改 GitHub / HF repo（保持引用稳定）
- **不要**同时投其他期刊（NC 严格执行 single submission policy）
- **不要**主动联系审稿人或编辑（会被记 bad faith）
- 如果 4 周后还没消息，可以礼貌地发一封 status query 邮件给编辑
