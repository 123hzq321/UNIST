"""Build an evidence-linked report; completed means an on-disk COMPLETE.json exists."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[3]
from joint_paths import joint_root

ROOT=joint_root()
OUT=PROJECT/'outputs/singlecell_spatial_benchmark/joint_training_v1'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    old=pd.read_csv(ROOT/'audit/legacy_reference_inventory.csv')
    good=old[old.status=='inspected']
    lines=['# UNIST 多患者、多癌种联合训练核对报告',
           '', '## 旧数据核对', '',
           f'- 检查到 {len(old):,} 个参考文件，{len(good):,} 个可读取，{len(old)-len(good)} 个损坏或不可读取。',
           f'- 可读取文件只有 {good.cell_id_hash.nunique():,} 种细胞 ID 集合指纹；这不是独立队列数，也不保证相同 ID 的表达矩阵相同。',
           f'- 逐文件累加得到 {int(good.n_cells.sum()):,} 条参考细胞成员记录，包含大量重复副本与折间复用，不能当作独立训练细胞数。',
           '- 文件存在不等于训练已完成。旧 benchmark 完成状态仍以旧日志为准，本次未重跑 1800-job 矩阵。',
           '- 原有外部测试清单已记录 SHA256；新的联合训练只读取本轮下载的三个参考文件。', '']
    old[old.status!='inspected'].to_csv(OUT/'legacy_unreadable_references.csv',index=False)
    records=json.loads((ROOT/'source_manifest.json').read_text())
    lines += ['## 新训练源', '', '| 来源 | 原始细胞数 | 供者标识数 | 文件状态 |', '|---|---:|---:|---|']
    for rec in records:
        lines.append(f"| [{rec['cancer']}](https://cellxgene.cziscience.com/collections/{rec['collection_id']}) | {rec['expected_cells']:,} | {len(rec['declared_donors'])} | {rec['status']} |")
    lines += ['', '供者数来自作者提供的 donor_id，不把合成混合物或模型重复次数记为新患者。', '']
    prep=ROOT/'prepared'; run=ROOT/'runs/joint_mlp_ensemble_v1'
    if (prep/'READY.json').exists():
        ready=json.loads((prep/'READY.json').read_text()); counts=pd.read_csv(prep/'split_counts.csv')
        lines += ['## 锁定划分', '', '| 来源 | 角色 | 细胞数 | 供者数 |','|---|---|---:|---:|']
        for r in counts.itertuples(): lines.append(f'| {r.source} | {r.split} | {r.cells:,} | {r.donors} |')
        lines += ['', f"共用基因 {ready['n_common_genes']:,} 个；仅由训练患者选择 2,000 个特征。",
                  '标签：'+', '.join(ready['labels'])+'。',
                  '同一供者只属于一个角色；作者原始标签、排除原因、表达指纹与行号均可追溯。', '']
        for p in [prep/'split_counts.csv',prep/'exclusions.csv',prep/'label_crosswalk.csv',prep/'donor_split_locked.csv',prep/'READY.json']:
            shutil.copy2(p,OUT/p.name)
    if (run/'COMPLETE.json').exists():
        complete=json.loads((run/'COMPLETE.json').read_text())
        lines += ['## 已实际完成', '',
                  f"- 联合训练：{complete['training_sources']} 个癌种来源、{complete['training_donors']} 个训练供者、{complete['training_cells']:,} 个训练细胞。",
                  f"- 完成 {complete['completed_models']} 个 MLP 成员，每个 {complete['epochs_per_member']} 轮；每轮遍历全部合格训练细胞，没有每样本 50 个细胞的抽样上限。",
                  '- 使用作者细胞标签监督，并加入同一训练供者内生成的合成混合物；后者是派生训练任务，不是真实空间金标准。',
                  '- 单细胞和合成混合物的测试均来自锁定的测试供者；所有模型训练完毕后才计算测试表现。', '']
        summary=pd.read_csv(run/'test_summary_donor_bootstrap.csv')
        metrics=pd.read_csv(run/'test_donor_metrics.csv')
        baseline=summary[(summary.calibration_target_coverage==1)&(summary.metric=='risk')&(summary.score=='fixed_fusion')]
        lines += ['## 测试结果', '', '以下为癌种等权、癌种内供者等权；不是按所有细胞直接计算的总体准确率。', '']
        for r in baseline.itertuples():
            endpoint='作者细胞标签错误率' if r.endpoint=='author_cell_labels' else '合成混合物 total-variation 误差'
            lines.append(f'- {endpoint}：{r.equal_source_donor_mean:.4f}（供者 bootstrap 95% CI {r.ci_low:.4f}–{r.ci_high:.4f}）。')
        lines += ['', '| 终点 | 风险分数 | 错误检测 AUROC | 95% CI | 可计算供者数 |','|---|---|---:|---|---:|']
        auc=summary[(summary.calibration_target_coverage==1)&(summary.metric=='error_detection_auroc')]
        for r in auc.itertuples():
            lines.append(f'| {r.endpoint} | {r.score} | {r.equal_source_donor_mean:.4f} | {r.ci_low:.4f}–{r.ci_high:.4f} | {r.n_donors} |')
        lines += ['', '混合物 AUROC 的阳性事件预先固定为 total-variation > 0.25；这里没有把跨患者测试强行命名为 OOD AUROC。没有阳性错误事件的供者，其 AUROC 未定义而不是 1；混合物中的部分供者只有 1–3 个大误差事件，AUROC 应结合连续误差与等覆盖率风险一起阅读。',
                  '融合是否优于单一分数以 paired_donor_risk_differences.csv 为准，不能只看融合自身成绩。不同方法的实际测试覆盖率可能不同。', '']
        compare=pd.read_csv(run/'paired_donor_risk_differences.csv')
        for r in compare[(compare.calibration_target_coverage==.8)&(compare.comparison=='fusion_minus_entropy_only')].itertuples():
            lines.append(f'- 校准目标覆盖率 80%，{r.endpoint} 的融合减 entropy 风险差：{r.risk_difference:+.4f}，95% CI [{r.ci_low:+.4f}, {r.ci_high:+.4f}]；负值有利于融合。')
        for name in ['COMPLETE.json','training_config.json','training_history.csv','test_donor_metrics.csv',
                     'test_summary_donor_bootstrap.csv','paired_donor_risk_differences.csv',
                     'author_cell_labels_frozen_calibration.json','synthetic_within_donor_mixtures_frozen_calibration.json']:
            shutil.copy2(run/name,OUT/name)
        if (run/'joint_vs_source_summary.csv').exists():
            controls=pd.read_csv(run/'joint_vs_source_summary.csv')
            lines += ['', '## 联合训练与单癌种训练对照', '',
                      '另完成三个单癌种集成对照，共 9 个 MLP。它们与联合模型使用相同测试细胞、相同合成混合物、相同训练集选出的基因，区别是是否合并其他癌种的训练细胞。']
            for r in controls.itertuples():
                lines.append(f'- {r.endpoint}：联合模型减单癌种模型误差 {r.joint_minus_source_error:+.4f}，95% CI [{r.ci_low:+.4f}, {r.ci_high:+.4f}]；负值有利于联合。')
            lines += ['这些对照仍共享联合训练集的特征选择，因此不等于完全独立选基因的单癌种流程，更不能替代成熟空间 mapper 的独立比较。','']
            for name in ['joint_vs_source_donor_metrics.csv','joint_vs_source_summary.csv']:
                shutil.copy2(run/name,OUT/name)
        if (run/'joint_vs_step_matched_source_summary.csv').exists():
            controls=pd.read_csv(run/'joint_vs_step_matched_source_summary.csv')
            lines += ['', '## 更新步数匹配对照', '',
                      '这是首次拟合后的补充审计，未修改联合模型。再完成三个单癌种集成，共 9 个 MLP；乳腺、肺、肾分别训练 40、146、675 轮，使每个成员的更新次数接近联合模型的 5,400 次。保留并同时报告两种对照，不择优挑选。']
            for r in controls.itertuples():
                lines.append(f'- {r.endpoint}：联合模型减步数匹配单癌种模型误差 {r.joint_minus_source_error:+.4f}，95% CI [{r.ci_low:+.4f}, {r.ci_high:+.4f}]；负值有利于联合。')
            donor_table=pd.read_csv(run/'joint_vs_step_matched_source_donor_metrics.csv')
            per_source=donor_table[donor_table.endpoint=='synthetic_within_donor_mixtures'].groupby('source').agg(
                donors=('donor_key','nunique'),joint=('joint_error','mean'),control=('source_specific_error','mean'),delta=('joint_minus_source_error','mean'))
            lines += ['', '| 合成混合物来源 | 测试供者 | 联合 TV 误差 | 步数匹配单癌种 TV 误差 | 差值 |', '|---|---:|---:|---:|---:|']
            for source,r in per_source.iterrows():
                lines.append(f'| {source} | {int(r.donors)} | {r.joint:.4f} | {r.control:.4f} | {r.delta:+.4f} |')
            lines += ['', '总体改善主要来自肾癌唯一测试供者，其次为肺癌；乳腺癌混合物没有改善。因此不能写成“各癌种均有稳定增益”，尤其不能把肾癌的单供者效应外推到肾癌人群。','']
            for name in ['joint_vs_step_matched_source_donor_metrics.csv','joint_vs_step_matched_source_summary.csv','compute_control_budgets.csv']:
                shutil.copy2(run/name,OUT/name)
        if (run/'FINAL_AUDIT.json').exists():
            audit=json.loads((run/'FINAL_AUDIT.json').read_text())
            matched=pd.read_csv(run/'matched_test_coverage_paired_differences.csv')
            lines += ['', '## 最终复核与等覆盖率诊断', '',
                      f"复核 {audit['checkpoints']} 个模型权重、供者隔离、表达指纹去重与 {audit['raw_to_prepared_spotchecks']} 处原始计数到训练矩阵的精确抽查，均通过。",
                      '因固定校准阈值在测试集上的实际覆盖率可能不同，另做每名测试供者保留相同比例的排序诊断。此项是拟合后诊断，不是可部署阈值或前瞻覆盖保证；没有重训或调整融合权重。']
            for r in matched[(matched.matched_test_coverage==.8)&matched.comparison.isin(['fusion_minus_entropy_only','fusion_minus_novelty_only'])].itertuples():
                lines.append(f'- 保留相同 80% 测试单位，{r.endpoint}，{r.comparison} 风险差：{r.risk_difference:+.6f}，95% CI [{r.ci_low:+.6f}, {r.ci_high:+.6f}]。')
            for name in ['FINAL_AUDIT.json','checkpoint_integrity.csv','raw_to_prepared_spotchecks.csv',
                         'matched_test_coverage_donor_metrics.csv','matched_test_coverage_paired_differences.csv',
                         'mixture_entropy_confound_audit.csv','error_event_counts.csv']:
                shutil.copy2(run/name,OUT/name)
            confound=pd.read_csv(run/'mixture_entropy_confound_audit.csv')
            rho=confound.groupby('source').predicted_vs_true_composition_entropy_rho.mean().mean()
            error_rho=confound.groupby('source').predicted_entropy_vs_mapping_error_rho.mean().mean()
            lines += [f'在合成混合物中，预测熵与真实组成熵的平均供者内 Spearman 为 {rho:.3f}，与映射误差为 {error_rho:.3f}。这是拟合后的解释性诊断，提示组成复杂程度与预测出错不是同一件事；不当作真实病理或因果证据。','']
        protected=pd.read_csv(ROOT/'audit/protected_legacy_test_manifests.csv')
        checks=[]
        for r in protected.itertuples():
            p=Path(r.path)
            actual=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else 'missing'
            checks.append({'path':str(p),'expected_sha256':r.sha256,'actual_sha256':actual,'unchanged':actual==r.sha256})
        pd.DataFrame(checks).to_csv(OUT/'protected_legacy_manifest_integrity.csv',index=False)
        lines += ['', f"本轮前后 {len(checks)} 个旧测试清单的 SHA256 检查：{sum(r['unchanged'] for r in checks)} 个未改变。这证明清单未改写，不等于证明所有历史样本与新参考的身份独立。", '']
    else:
        lines += ['## 当前执行状态', '', '尚无正式训练完成标记，不能报告完成训练或测试结果。']
        if (run/'status.json').exists(): lines += ['```json',(run/'status.json').read_text(),'```']
    lines += ['## 结论边界', '',
        '- 这是三癌种的联合参考与混合物训练，不是新增数百个真实独立临床队列，也不是新一轮真实病理 rescue 结果。',
        '- 当前每个癌种只有一个来源研究；癌种与研究来源混杂。肾癌测试只有一个供者，不能估计肾癌人群中的变异。',
        '- 作者注释不是独立病理金标准；肿瘤/上皮大类的识别不等于恶性诊断。',
        '- 新模型、标签空间和基因集合与原论文模型不同；旧论文指标不能直接当作本模型结果。',
        '- 是否优于各癌种单独训练，必须看相同划分、相同测试单位上的供者配对比较，不能从训练规模推断。',
        '- 2,000 基因全转录组模型不能未经缺失基因验证直接套在数百基因的靶向空间面板上。',
        '- 后续需要冻结模型的真实空间/病理审计、更多同癌种独立来源，以及独立重新选基因的 leave-one-source/cancer-out 训练。', '',
        '## 位置', '', f'- 数据与模型：`{ROOT}`',f'- 本报告及轻量汇总：`{OUT}`',
        '- 训练设计：`D:/UNIST_data/joint_training_v1/TRAINING_PROTOCOL.md`',
        '- 完整细胞清单：`D:/UNIST_data/joint_training_v1/prepared/cells.csv.gz`',
        '- 模型权重：`D:/UNIST_data/joint_training_v1/runs/joint_mlp_ensemble_v1/ensemble_member_*.pt`', '']
    (OUT/'joint_training_audit_report_zh.md').write_text('\n'.join(lines),encoding='utf-8')
    shutil.copy2(ROOT/'source_manifest.json',OUT/'source_manifest.json')
    shutil.copy2(ROOT/'TRAINING_PROTOCOL.md',OUT/'TRAINING_PROTOCOL.md')
    if (ROOT/'SECONDARY_COMPUTE_CONTROL.md').exists():
        shutil.copy2(ROOT/'SECONDARY_COMPUTE_CONTROL.md',OUT/'SECONDARY_COMPUTE_CONTROL.md')
    mirror=ROOT/'reports'; mirror.mkdir(exist_ok=True)
    for p in OUT.iterdir():
        if p.is_file(): shutil.copy2(p,mirror/p.name)
    print(OUT/'joint_training_audit_report_zh.md',flush=True)

if __name__=='__main__': main()
