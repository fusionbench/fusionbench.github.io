# -*- coding: utf-8 -*-
"""
AmbiguousEntityEvaluator：统一一对一 + 匈牙利算法（无 text 模式）
- 评估口径：AR / ER / EAR（J = ER * AR）
- 汇总口径：同时返回 Macro（样本等权）与 Micro（按 gold 数加权，近似“总体召回”）
- 调试：verbose 打印实体/答案分词、矩阵、匹配对与均值（前 N 条）

增强：
- 支持 mode=both：同时评估 grouped 与 ungrouped，并各自落盘
- 保存与打印均四舍五入至 3 位小数
- tokenizer 钩子（中文分词可接入），去歧义表面词可开关
- 匈牙利无 SciPy 回退，数值裁剪到 [0,1]
"""
import collections
import csv
import os
from pprint import pprint
import re
import string
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from utils import load_json, logger, save_json


# ================== 停用词与分词 ==================

def _load_stopwords(language: str = 'english') -> set:
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words(language))
    except Exception as e:
        logger.warning(f'error loading stopwords: {e}')
        # 兜底：简易英文停用词
        return {
            "a","an","the","and","or","but","if","then","than","so","because",
            "is","am","are","was","were","be","been","being",
            "of","to","in","on","for","with","as","by","at","from","into","about",
            "this","that","these","those","it","its","he","she","they","them","his","her","their",
        }

def _default_tokenizer(text: str) -> List[str]:
    """
    默认英文分词器：小写、去标点、去冠词、按空格切分。
    中文请注入外部分词器：lambda s: list(jieba.cut(s))
    """
    if not isinstance(text, str):
        text = str(text)

    def remove_articles(t):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE | re.IGNORECASE)
        return re.sub(regex, ' ', t)

    def white_space_fix(t):
        return ' '.join(t.split())

    def remove_punc(t):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in t if ch not in exclude)

    def lower(t):
        return t.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text)))).split()


# ================== 评估器主体 ==================

class AmbiguousEntityEvaluator:
    def __init__(
        self,
        language: str = 'english',
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        remove_surface: bool = True,
    ):
        """
        tokenizer: 可注入自定义分词器。若为中文，建议传入 jieba.cut 的包装：lambda s: list(jieba.cut(s))
        remove_surface: 是否移除主体中的歧义表面词（由 ambiguous_name 提供）
        """
        self.stop_words = _load_stopwords(language)
        self.tokenizer = tokenizer or _default_tokenizer
        self.remove_surface = remove_surface

    # ======== 基础文本处理 ========

    def get_tokens(self, text: str) -> List[str]:
        if not text:
            return []
        try:
            return self.tokenizer(text)
        except Exception:
            # 回退到默认英文 tokenizer
            return _default_tokenizer(text)

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [w for w in tokens if w not in self.stop_words and len(w) > 0]

    def get_meaningful_tokens(self, text: str) -> List[str]:
        return self.remove_stopwords(self.get_tokens(text))

    # ======== 实体键 token 化（保留括号内判别词） ========

    def entity_key_tokens(self, name: str, ambiguous_name: str) -> List[str]:
        """
        用于实体键一致性的 token：
        - 括号内判别词：仅去停用词（保留判别信息）
        - 主体（括号外）：去停用词，且（可选）移除歧义表面词 tokens
        - 合并两者；若为空则回退到“主体仅去停用词”
        """
        name = str(name) if not isinstance(name, str) else name

        # 括号内
        qualifiers = re.findall(r'\(([^)]*)\)', name)
        qual_tokens = []
        for q in qualifiers:
            qual_tokens.extend(self.get_meaningful_tokens(q))  # 仅去停用词

        # 主体（括号外）
        main_part = re.sub(r'\s*\([^)]*\)', '', name)
        main_tokens = self.get_meaningful_tokens(main_part)

        # 去歧义表面词（来自 ambiguous_name）
        surface_tokens = set(self.get_meaningful_tokens(ambiguous_name))
        if self.remove_surface:
            main_core = [t for t in main_tokens if t not in surface_tokens]
        else:
            main_core = main_tokens

        merged = main_core + qual_tokens
        if not merged:  # 全被删空则回退
            merged = main_tokens
        return merged  # 可能仍为空

    # ======== Token 级召回（答案用） ========

    def compute_token_recall(self, gold_text: str, pred_text: str) -> float:
        """
        Token 级召回：交集 / gold_tokens（多重计数）
        注意：gold 或 pred 为空 -> 0.0；gold_tokens 为空 -> 0.0（保守）
        """
        if not gold_text or not pred_text:
            return 0.0

        gold_tokens = self.get_meaningful_tokens(gold_text)
        pred_tokens = self.get_meaningful_tokens(pred_text)

        if not gold_tokens or not pred_tokens:
            return 0.0

        gold_counter = collections.Counter(gold_tokens)
        pred_counter = collections.Counter(pred_tokens)
        inter = gold_counter & pred_counter
        denom = sum(gold_counter.values())
        return (sum(inter.values()) / denom) if denom else 0.0

    def compute_token_precision(self, gold_text: str, pred_text: str) -> float:
        """
        Token 级精确率：交集 / pred_tokens（多重计数）
        注意：gold 或 pred 为空 -> 0.0；pred_tokens 为空 -> 0.0（保守）
        """
        if not gold_text or not pred_text:
            return 0.0

        gold_tokens = self.get_meaningful_tokens(gold_text)
        pred_tokens = self.get_meaningful_tokens(pred_text)

        if not gold_tokens or not pred_tokens:
            return 0.0

        gold_counter = collections.Counter(gold_tokens)
        pred_counter = collections.Counter(pred_tokens)
        inter = gold_counter & pred_counter
        denom = sum(pred_counter.values())
        return (sum(inter.values()) / denom) if denom else 0.0

    # ======== 匈牙利封装（最大化版本，返回总分与 G） ========

    def _hungarian_max(self, A: np.ndarray, metric_type: str = 'recall') -> Tuple[float, List[Tuple[int, int]], float, int, int]:
        """
        对矩阵 A 做"最大化的一对一匹配"，返回:
        - avg: 根据metric_type归一化后的平均分（recall用G，precision用P）
        - matches: 匹配对列表 (gi, pj)
        - total_sum: 选中的配对得分之和（分子）
        - G: gold 行数
        - P: pred 数量
        Args:
            A: 得分矩阵
            metric_type: 'recall' 或 'precision'，决定分母选择
        """
        G, P = A.shape if A is not None else (0, 0)
        if G == 0 or P == 0:
            return 0.0, [], 0.0, G, P

        # 稳健性：裁剪到 [0,1]，并使用 cost = 1 - sim
        A = np.clip(A, 0.0, 1.0)
        cost = 1.0 - A

        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost)
            total = float(A[row_ind, col_ind].sum())
            
            # 根据指标类型选择正确的分母
            if metric_type == 'precision':
                avg = total / float(P) if P else 0.0
            else:  # 'recall' 或其他情况默认使用G
                avg = total / float(G) if G else 0.0
                
            matches = list(zip(row_ind.tolist(), col_ind.tolist()))
            return avg, matches, total, G, P
        except Exception as e:
            logger.error(f"发生错误: {e}")
            # 回退：贪心最大匹配
            candidates = [(A[gi, pj], gi, pj) for gi in range(G) for pj in range(P) if A[gi, pj] > 0.0]
            candidates.sort(key=lambda x: x[0], reverse=True)
            used_g, used_p, total = set(), set(), 0.0
            matches = []
            for score, gi, pj in candidates:
                if gi in used_g or pj in used_p:
                    continue
                used_g.add(gi); used_p.add(pj)
                total += float(score); matches.append((gi, pj))
                if len(used_g) == G:
                    break
            
            # 根据指标类型选择正确的分母
            if metric_type == 'precision':
                avg = total / float(P) if P else 0.0
            else:  # 'recall' 或其他情况默认使用G
                avg = total / float(G) if G else 0.0
                
            return avg, matches, total, G, P

    # ======== 矩阵构建 ========

    def _build_all_score_matrices(self, example: Dict[str, Any]):
        gold_answers = example.get("gold_answers", {})
        pred_answers = example.get("generated_answers", {})
        ambiguous_name = example.get("ambiguous_name", "")

        gold_entities = list(gold_answers.keys())
        pred_entities = list(pred_answers.keys())
        gold_ans_list = [gold_answers[g] for g in gold_entities]
        pred_ans_list = [pred_answers[p] for p in pred_entities]

        gold_tok_list = [self.entity_key_tokens(g, ambiguous_name) for g in gold_entities]
        pred_tok_list = [self.entity_key_tokens(p, ambiguous_name) for p in pred_entities]

        G, P = len(gold_entities), len(pred_entities)
        
        AR = np.zeros((G, P), dtype=float)
        AP = np.zeros((G, P), dtype=float)
        ER = np.zeros((G, P), dtype=float)
        EP = np.zeros((G, P), dtype=float)

        for gi in range(G):
            gtok = gold_tok_list[gi]
            gans = gold_ans_list[gi]
            
            for pj in range(P):
                ptok = pred_tok_list[pj]
                pans = pred_ans_list[pj]

                # Entity scores
                if gtok and ptok:
                    gc = collections.Counter(gtok)
                    pc = collections.Counter(ptok)
                    inter = gc & pc
                    inter_sum = float(sum(inter.values()))
                    
                    denom_er = float(len(gtok))
                    ER[gi, pj] = inter_sum / denom_er if denom_er > 0 else 0.0
                    
                    denom_ep = float(len(ptok))
                    EP[gi, pj] = inter_sum / denom_ep if denom_ep > 0 else 0.0

                # Answer scores
                if gans and pans:
                    AR[gi, pj] = self.compute_token_recall(gans, pans)
                    AP[gi, pj] = self.compute_token_precision(gans, pans)

        EAR = ER * AR
        EAP = EP * AP
        
        # Pack for debugging
        pack = (gold_entities, pred_entities, gold_tok_list, pred_tok_list, gold_ans_list, pred_ans_list)

        return {
            "AR": AR, "AP": AP, "ER": ER, "EP": EP, "EAR": EAR, "EAP": EAP, "pack": pack
        }

    # ======== 单样本统计（返回 avg,total,G） ========

    def _get_all_stats(self, example):
        matrices = self._build_all_score_matrices(example)
        stats = {}
        
        # 依次对 AR, AP, ER, EP, EAR, EAP 矩阵做匈牙利匹配
        # 区分 Recall 和 Precision 指标，使用正确的分母
        metric_types = {
            "AR": "recall",     # Answer Recall
            "AP": "precision",  # Answer Precision  
            "ER": "recall",     # Entity Recall
            "EP": "precision",  # Entity Precision
            "EAR": "recall",    # Entity Answer Recall
            "EAP": "precision"  # Entity Answer Precision
        }
        
        for key in ["AR", "AP", "ER", "EP", "EAR", "EAP"]:
            metric_type = metric_types[key]
            avg, matches, total_sum, G, P = self._hungarian_max(matrices[key], metric_type)
            stats[key] = {'avg': avg, 'matches': matches, 'sum': total_sum, 'G': G, 'P': P}
        
        return stats, matrices

    # ======== 对外简洁接口（仅均值） ========

    def evaluate_single_example(self, example: Dict[str, Any]) -> Dict[str, float]:
        stats, _ = self._get_all_stats(example)
        
        def _f1(p, r):
            return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        ar = stats['AR']['avg']
        ap = stats['AP']['avg']
        er = stats['ER']['avg']
        ep = stats['EP']['avg']
        ear = stats['EAR']['avg']
        eap = stats['EAP']['avg']

        return {
            "answer_recall": ar,
            "answer_precision": ap,
            "answer_f1": _f1(ap, ar),
            "entity_recall": er,
            "entity_precision": ep,
            "entity_f1": _f1(ep, er),
            "entity_answer_recall": ear,
            "entity_answer_precision": eap,
            "entity_answer_f1": _f1(eap, ear),
        }

    # ======== 数据集汇总：Macro + Micro ========

    def evaluate(self, data_path: str, verbose: bool = False, sample_n: int = 10) -> Dict[str, float]:
        """
        评测单个文件
        """
        logger.info(f"开始计算文件 {data_path} 的评估指标")
        dataset = load_json(data_path)

        # macro trackers
        ars, aps, ers, eps, ears, eaps = [], [], [], [], [], []
        
        # micro trackers
        ar_total_sum, ap_total_sum, er_total_sum, ep_total_sum, ear_total_sum, eap_total_sum = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        gold_total = 0
        pred_total = 0

        for i, ex in enumerate(dataset):
            try:
                stats, matrices = self._get_all_stats(ex)
              
                # append for macro
                ars.append(stats['AR']['avg'])
                aps.append(stats['AP']['avg'])
                ers.append(stats['ER']['avg'])
                eps.append(stats['EP']['avg'])
                ears.append(stats['EAR']['avg'])
                eaps.append(stats['EAP']['avg'])

                # accumulate for micro
                G = stats['AR']['G'] # any 'G' would work
                P = stats['AR']['P'] # any 'P' would work
                gold_total += G
                pred_total += P
                ar_total_sum += stats['AR']['sum']
                ap_total_sum += stats['AP']['sum']
                er_total_sum += stats['ER']['sum']
                ep_total_sum += stats['EP']['sum']
                ear_total_sum += stats['EAR']['sum']
                eap_total_sum += stats['EAP']['sum']

                if verbose and i < sample_n:
                    print("\n" + "="*80)
                    print(f"[样例 {i}] ambiguous_name: {ex.get('ambiguous_name','')}")
                    self._debug_example(ex, stats, matrices)

            except Exception as e:
                logger.error(f"处理 {ex.get('ambiguous_name')} 时发生错误: {e}.")
                ars.append(0.0); aps.append(0.0); ers.append(0.0); eps.append(0.0); ears.append(0.0); eaps.append(0.0)

        def _get_scores(recall_list, precision_list, recall_sum, precision_sum, total_g, total_p):
            macro_r = float(np.mean(recall_list)) if recall_list else 0.0
            macro_p = float(np.mean(precision_list)) if precision_list else 0.0
            macro_f1 = 2 * macro_r * macro_p / (macro_r + macro_p) if (macro_r + macro_p) > 0 else 0.0
            
            micro_r = (recall_sum / total_g) if total_g > 0 else 0.0
            micro_p = (precision_sum / total_p) if total_p > 0 else 0.0
            micro_f1 = 2 * micro_r * micro_p / (micro_r + micro_p) if (micro_r + micro_p) > 0 else 0.0
            
            return {
                "macro_recall": macro_r, "macro_precision": macro_p, "macro_f1": macro_f1,
                "micro_recall": micro_r, "micro_precision": micro_p, "micro_f1": micro_f1,
            }

        answer_scores = _get_scores(ars, aps, ar_total_sum, ap_total_sum, gold_total, pred_total)
        entity_scores = _get_scores(ers, eps, er_total_sum, ep_total_sum, gold_total, pred_total)
        joint_scores = _get_scores(ears, eaps, ear_total_sum, eap_total_sum, gold_total, pred_total)
        
        scores = {
            # Answer
            "answer_recall": answer_scores["macro_recall"],
            "answer_precision": answer_scores["macro_precision"],
            "answer_f1": answer_scores["macro_f1"],

            # Entity
            "entity_recall": entity_scores["macro_recall"],
            "entity_precision": entity_scores["macro_precision"],
            "entity_f1": entity_scores["macro_f1"],

            # Entity-answer
            "entity_answer_recall": joint_scores["macro_recall"],
            "entity_answer_precision": joint_scores["macro_precision"],
            "entity_answer_f1": joint_scores["macro_f1"],
        }
        scores = {k: round(v, 3) for k, v in scores.items()}
        return {
            "data_path": data_path,
            **scores
        }

    def batch_evaluate(self, data_dir: str):
        """
        批量计算指标

        Args:
            data_dir: 待评测数据(.json, .jsonl) 的根目录, 注意: 不评测子目录下的数据文件
        """
        logger.info(f"开始计算根目录 {data_dir} 下结果文件的指标")
        results = []
        for fname in os.listdir(data_dir):
            path = os.path.join(data_dir, fname)
            if os.path.isfile(path) and fname.lower().endswith(('.json', '.jsonl')):
                r = self.evaluate(path)
                logger.info(f"评测结果: {r}")
                results.append(r)
        return results


    # ======== 调试输出 ========

    def _print_matrix_with_headers(self, M: np.ndarray, row_headers: List[str], col_headers: List[str],
                                   title: str, max_rows: int = 10, max_cols: int = 10):
        print(f"\n--- {title} ---")
        if M is None or M.size == 0:
            print("(empty)")
            return
        r = min(M.shape[0], max_rows)
        c = min(M.shape[1], max_cols)
        hdr = ["(g\\p)"] + [f"{col_headers[j][:48]}" for j in range(c)]
        print("\t".join(hdr))
        for i in range(r):
            row = [f"{row_headers[i][:48]}"]
            for j in range(c):
                row.append(f"{M[i,j]:.3f}")
            print("\t".join(row))
        if M.shape[0] > r or M.shape[1] > c:
            print(f"... (显示前 {r}×{c}，实际 {M.shape[0]}×{M.shape[1]})")

    def _debug_example(self, example: Dict[str, Any], stats: Dict, matrices: Dict):
        ambiguous_name = example.get("ambiguous_name", "")
        (gold_entities, pred_entities, gold_tok, pred_tok, gold_ans_list, pred_ans_list) = matrices['pack']
        
        # 实体 token 展示
        print("\n[实体分词]")
        for i, g in enumerate(gold_entities):
            print(f"  gold[{i}] {g} -> {gold_tok[i] if i < len(gold_tok) else []}")
        for j, p in enumerate(pred_entities):
            print(f"  pred[{j}] {p} -> {pred_tok[j] if j < len(pred_tok) else []}")

        # 答案 token 展示
        print("\n[答案分词]")
        for i, ga in enumerate(gold_ans_list):
            print(f"  gold[{i}] ans: {ga} -> {self.get_meaningful_tokens(ga)}")
        for j, pa in enumerate(pred_ans_list):
            print(f"  pred[{j}] ans: {pa} -> {self.get_meaningful_tokens(pa)}")

        def _print_metric_debug(metric_key_short: str):
            M = matrices[metric_key_short]
            metric_stats = stats[metric_key_short]
            matches = metric_stats['matches']
            total = metric_stats['sum']
            G = metric_stats['G']
            avg = metric_stats['avg']

            if 'A' in metric_key_short: # Answer metrics
                row_headers = [f"{i}:{gold_entities[i]} :: {str(gold_ans_list[i])[:32]}" for i in range(len(gold_ans_list))]
                col_headers = [f"{j}:{pred_entities[j]} :: {str(pred_ans_list[j])[:32]}" for j in range(len(pred_ans_list))]
            else: # Entity metrics
                row_headers = [f"{i}:{gold_entities[i]}" for i in range(len(gold_entities))]
                col_headers = [f"{j}:{pred_entities[j]}" for j in range(len(pred_entities))]
            
            self._print_matrix_with_headers(M, row_headers, col_headers, title=f"{metric_key_short} 矩阵")
            print(f"\n[{metric_key_short}] 匹配对（gi->pj，score）:")
            for gi, pj in matches:
                sc = M[gi, pj]
                print(f"  {gi} -> {pj} | {gold_entities[gi]} ~~ {pred_entities[pj]} | score={sc:.3f}")
            print(f"[{metric_key_short}] Avg = {total / max(1, G):.3f} (与返回值一致：{avg:.3f})")

        _print_metric_debug("ER")
        _print_metric_debug("EP")
        _print_metric_debug("AR")
        _print_metric_debug("AP")
        _print_metric_debug("EAR")
        _print_metric_debug("EAP")

        # 歧义表面词
        surf = self.get_meaningful_tokens(ambiguous_name)
        print(f"\n[歧义表面词 tokens from ambiguous_name='{ambiguous_name}'] -> {surf if surf else '(empty)'}")


def save_listdict_to_csv(result_list, csv_path):
    if not result_list:
        logger.warning("结果列表为空，未生成 CSV。")
        return
    fieldnames = list(result_list[0].keys())
    with open(csv_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_list)


def main():
    """
    用法示例：
    1) 单文件模式：python evaluate.py <data_path>
    2) 批量模式：python evaluate.py <data_dir>
    自动检测输入类型（文件或目录）
    """
    if len(sys.argv) < 2:
        logger.error("用法: python evaluate.py <data_path|data_dir>")
        sys.exit(1)

    input_path = sys.argv[1]
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    evaluator = AmbiguousEntityEvaluator()

    output_dir = 'output/evaluations'
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isfile(input_path):
        result = evaluator.evaluate(input_path, verbose=verbose)
        pprint(result)
        output_path = os.path.join(output_dir, time.strftime("%Y%m%d_%H%M%S") + '.json')
        save_json(result, output_path)

    elif os.path.isdir(input_path):
        results = evaluator.batch_evaluate(input_path)
        output_path = os.path.join(output_dir, time.strftime("%Y%m%d_%H%M%S") + '.csv')
        save_listdict_to_csv(results, output_path)
        
    else:
        logger.error(f"路径不存在: {input_path}")
        sys.exit(1)

    logger.info(f"结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
