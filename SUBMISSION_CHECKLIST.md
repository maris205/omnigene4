# OmniGene-4 投稿材料清单

## 📄 核心文件

### 1. 主论文
- **文件**: `paper/omnigene4.pdf` (642 KB, 21 页)
- **LaTeX 源文件**: `paper/omnigene4.tex`
- **参考文献**: `paper/refs.bib` (65 篇)
- **状态**: ✅ 已完成，包含所有 P0/P1/P2 修复

### 2. Cover Letter
- **文件**: `paper/cover_letter.pdf` (92 KB)
- **LaTeX 源文件**: `paper/cover_letter.tex`
- **状态**: ✅ 已完成，包含作者信息和建议审稿人

### 3. 作者信息
```
Liang Wang (王亮)
School of Artificial Intelligence and Automation
Huazhong University of Science and Technology
Wuhan, Hubei 430070, P.R. China
Email: wangliang.f@gmail.com
```

---

## 🎯 投稿目标期刊

### 推荐顺序

#### 1. BMC Bioinformatics (首选)
- **影响因子**: ~3.0 (Q2)
- **审稿周期**: 2-3 个月
- **接受率**: 60-70%
- **投稿网址**: https://www.biomedcentral.com/bmcbioinformatics
- **优势**: 
  - 方法论导向
  - 重视可复现性
  - 开放获取
- **准备材料**:
  - ✅ 主论文 PDF
  - ✅ Cover Letter
  - ✅ LaTeX 源文件
  - ✅ 图表源文件（已嵌入 LaTeX）

#### 2. PLOS Computational Biology (备选)
- **影响因子**: ~4.0 (Q1)
- **审稿周期**: 3-4 个月
- **接受率**: 50-60%
- **投稿网址**: https://journals.plos.org/ploscompbiol/
- **优势**:
  - 重视统计严谨性
  - 开放获取
  - 高可见度
- **准备材料**: 同上

#### 3. Bioinformatics (挑战)
- **影响因子**: ~5.8 (Q1)
- **审稿周期**: 2-3 个月
- **接受率**: 30-40%
- **投稿网址**: https://academic.oup.com/bioinformatics
- **优势**:
  - 顶级期刊
  - 快速发表
- **风险**: 要求更高的 benchmark 数字

---

## 📊 论文核心数据

### 主要结果
- **Standard Homology**: 99.95% (6,000 pairs)
- **Remote Homology**: 59.50% (2,000 pairs)
- **BixBench Knowledge**: 93.66%

### 核心贡献
1. **CPT/SFT 分解**: 96.2% [95.9%, 96.6%] / 3.7% [3.4%, 4.1%]
2. **Format Control**: 90.2% retention ratio
3. **ESM-2 对比**: OmniGene-4 59.5% vs ESM-2 50.5% (+9 pp)

### 统计严谨性
- Bootstrap CI (1,000 iterations)
- Format-matched routing control
- Head-to-head external baseline

---

## 💾 代码和数据

### GitHub 仓库
- **地址**: https://github.com/maris205/omnigene4
- **状态**: ✅ 公开，包含所有代码

### 关键脚本
1. **训练**:
   - `biopaws/cpt/1-prepare_cpt_data_mp.py` (数据准备)
   - `biopaws/cpt/2-run_cpt.py` (CPT 训练)
   - `biopaws/sft_data/17-train_bio_sft_v3_remote.py` (SFT 训练)

2. **评测**:
   - `biopaws/cpt/18-eval_v3_sft.py` (主评测)
   - `biopaws/cpt/27-esm2_remote_headtohead.py` (ESM-2 对比)

3. **分析**:
   - `biopaws/cpt/20-collect_moe_activations.py` (路由收集)
   - `biopaws/cpt/26-format_matched_routing.py` (Format control)
   - `biopaws/cpt/28-bootstrap_ci.py` (Bootstrap CI)

### 数据文件
- **位置**: `outputs/moe_analysis/`
- **内容**:
  - `moe_counts_baseline.npz` (Baseline 路由)
  - `moe_counts_cpt.npz` (CPT 路由)
  - `moe_counts_v3.npz` (v3 路由)
  - `format_matched_report.json` (Format control 结果)
  - `esm2_remote_2k_eval.json` (ESM-2 对比结果)
  - `bootstrap_ci_report.json` (Bootstrap CI 结果)

---

## ✅ 投稿前检查清单

### 论文内容
- [x] Abstract 包含所有关键结果
- [x] Introduction 清晰阐述动机
- [x] Methods 详细可复现
- [x] Results 包含所有表格和图表
- [x] Discussion 诚实披露限制
- [x] References 格式正确（65 篇）

### 统计严谨性
- [x] Bootstrap CI (1,000 iterations)
- [x] Format control (90.2% retention)
- [x] External baseline (ESM-2 head-to-head)
- [x] 所有 claim 有数据支撑

### 可复现性
- [x] 代码公开（GitHub）
- [x] 数据可获取（outputs/moe_analysis/）
- [x] 训练脚本完整
- [x] 评测脚本完整

### 作者信息
- [x] 姓名正确
- [x] 单位正确（华中科技大学 人工智能与自动化学院）
- [x] 邮箱正确（wangliang.f@gmail.com）
- [x] 通讯作者标注

### Cover Letter
- [x] 阐述核心贡献
- [x] 说明适合期刊原因
- [x] 建议审稿人（4 位）
- [x] 声明无利益冲突
- [x] 数据和代码可获取性

---

## 🚀 投稿步骤

### BMC Bioinformatics 投稿流程

1. **注册账号**: https://www.editorialmanager.com/bmbi/
2. **准备材料**:
   - 主论文 PDF (`omnigene4.pdf`)
   - Cover Letter PDF (`cover_letter.pdf`)
   - LaTeX 源文件（可选，建议上传）
   - 图表源文件（已嵌入 LaTeX）
3. **填写信息**:
   - 文章类型: Research Article
   - 主题分类: Machine Learning, Protein Structure, Sequence Analysis
   - 关键词: Mixture-of-Experts, Protein Homology, Foundation Models, Interpretability
4. **建议审稿人**:
   - Zeming Lin (MIT)
   - Sergey Ovchinnikov (MIT)
   - Jian Peng (UIUC)
   - Jinbo Xu (TTIC)
5. **提交**: 点击 Submit

### 预期时间线
- **投稿**: 2026-05-12
- **初审**: 1-2 周
- **外审**: 2-3 个月
- **修改**: 2-4 周
- **接受**: 2026-08-12 ~ 2026-09-12
- **发表**: 2026-09-12 ~ 2026-10-12

---

## 📝 审稿人可能的问题

### 预期问题 1: "Remote homology 只有 59.5%，太低了"
**回答**: 
- 我们在 Limitations 中诚实披露了这个差距
- 提供了 ESM-2 head-to-head 对比（+9 pp）
- 说明这是 representation-level 限制，需要更多 CPT 数据
- 强调我们的贡献是方法论（MoE 路由分析），不是 SOTA benchmark

### 预期问题 2: "只有一个模型家族（Gemma-4），泛化性如何？"
**回答**:
- 我们在 Limitations 中承认了这个限制
- 提议 future work 在 Mixtral/DeepSeek 上复现
- 强调我们的方法论框架是通用的，可以应用到任何 MoE 模型

### 预期问题 3: "Router.proj LoRA 的重要性没有 ablation 验证"
**回答**:
- 我们在 Limitations 中承认了这个限制
- 如果审稿人坚持，可以补充实验（~50 GPU-hours）
- 提供 pilot observation 的初步证据

### 预期问题 4: "BioPAWS benchmark 是作者自己的，有偏见吗？"
**回答**:
- 我们在 Limitations 中披露了 benchmark ownership
- 使用 held-out splits，避免 SFT 数据污染
- 提供了外部 baseline（ESM-2）验证
- 欢迎外部实验室在 BioPAWS 上复现

---

## 🎉 总结

**论文状态**: ✅ 完成，可投稿

**核心优势**:
- 统计严谨（Bootstrap CI）
- 方法论创新（MoE 路由分析）
- 诚实披露限制
- 代码完全可复现

**推荐行动**:
1. 投稿 BMC Bioinformatics（首选）
2. 同步发 arXiv 预印本
3. 准备好应对审稿人的 router.proj ablation 要求

**预期结果**: 60-70% 接受率，2026-08 ~ 2026-09 接受
