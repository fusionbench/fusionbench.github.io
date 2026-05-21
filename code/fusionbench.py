from string import Template
from textwrap import dedent
import json
from typing import Any, Dict, List, Optional

from llms import get_llm_chat
from utils import logger


# 知识分组
KNOWLEDGE_GROUPING = Template(dedent("""\
    You are given knowledge about several entities, which will be given in the forms of text, tables, triples.
    Your mode: Classify texts, tables, and triples, concerning the same entity into one group.

    Example 1
    Input:
    Anna Mae -- occupation --> Actress
    A. M. G. R. (1876–1939), was a prominent British astronomer who founded the astronomical observatory at Cambridge.
    Anna Mae (1905–1990) was an American actress, singer, and dancer. She was a prominent figure in the Golden Age of Hollywood.
    A. M. G. R. -- field of work --> Astronomy
    Anna Mae -- date of birth --> 1905

    Output:
    Group A. M. G. R.: A. M. G. R. (1876–1939), was a prominent British astronomer who founded the astronomical observatory at Cambridge. A. M. G. R. -- field of work --> Astronomy
    Group Anna Mae: Anna Mae (1905–1990) was an American actress, singer, and dancer. She was a prominent figure in the Golden Age of Hollywood. Anna Mae -- occupation --> Actress. Anna Mae -- date of birth --> 1905

    Example 2
    Input:
    Table:
    | Name | Year of Birth | Profession |
    |---|---|---|
    | Charles Dickens | 1812 | Novelist |
    | Charles Darwin | 1809 | Naturalist |
    Charles Darwin was best known for his contributions to the science of evolution.
    Charles Dickens -- notable work --> A Christmas Carol

    Output:
    Group Charles Darwin: Table row: Charles Darwin, 1809, Naturalist. Charles Darwin was best known for his contributions to the science of evolution.
    Group Charles Dickens: Table row: Charles Dickens, 1812, Novelist. Charles Dickens -- notable work --> A Christmas Carol

    Here is the knowledge:
    ${context}
    Your output:
"""))


# 歧义检测：从上下文中找出可能能回答问题的实体候选
AMBIGUITY_DETECTION = Template(dedent("""\
    You are an expert entity disambiguation assistant.
    Your task: Given the user's question and the provided context (which may contain multiple entities with similar names),
    identify which entities in the context are plausible candidates for answering the question.

    Rules:
    - Only propose entities that appear in the context.
    - If only one entity is clearly relevant, return a single candidate.
    - If none can answer, return an empty list.
    - Keep candidate names exactly as they appear in the context.

    Return STRICT JSON:
    {
      "candidates": [
        {"entity": "<entity name>", "reason": "<short reason>", "confidence": 0.0}
      ]
    }

    Question: ${query}
    Context:
    ${context}
"""))


# 针对单一实体：从原始上下文中检索该实体相关证据
ENTITY_EVIDENCE_RETRIEVAL = Template(dedent("""\
    You are an evidence retrieval assistant.
    Given the user's question, a target entity, and the raw context, extract ONLY the evidence relevant to answering the question for that entity.

    Rules:
    - Only include information that is about the target entity.
    - Only include information that helps answer the question.
    - If there is no relevant evidence for the target entity, output "Nothing".
    - Do not add new facts.

    Question: ${query}
    Target entity: ${entity}
    Context:
    ${context}

    Relevant evidence:
"""))


# 针对单一实体：构建推理链并尝试回答（允许判定无法回答）
ENTITY_REASONING_AND_ANSWER = Template(dedent("""\
    You are a careful question answering assistant.
    Given a question, a target entity, and evidence about that entity, decide whether the evidence is sufficient to answer the question.

    Rules:
    - Use ONLY the provided evidence.
    - If the evidence is insufficient or irrelevant, set "can_answer": false and set "answer": "N/A".
    - Provide a concise reasoning chain (steps) that explicitly links evidence to the answer when can_answer is true.
    - Keep the answer short and factual.

    Return STRICT JSON:
    {
      "entity": "<entity>",
      "can_answer": true,
      "answer": "<answer or N/A>",
      "reasoning_chain": ["step1", "step2"],
      "used_evidence": ["evidence snippet 1", "evidence snippet 2"]
    }

    Question: ${query}
    Target entity: ${entity}
    Evidence:
    ${evidence}
"""))


# 歧义消解：综合所有候选实体结果，决定最终回复
DISAMBIGUATION_AND_FINAL = Template(dedent("""\
    You are an expert entity disambiguation judge.
    You are given the user's question and multiple candidate-entity analysis results (each includes can_answer, reasoning_chain, answer).
    Decide the final response to the user.

    Rules:
    - Prefer candidates with can_answer=true and with reasoning grounded in evidence.
    - If multiple entities yield valid different answers and the question itself is ambiguous, present multiple answers with entity names.
    - If none can answer, reply "N/A" and briefly explain that the provided context lacks sufficient evidence.
    - Do not invent facts beyond the provided analyses.

    Return STRICT JSON:
    {
      "final_answer": {"<entity name>": "<answer>",},
      "notes": "<short note about ambiguity or insufficiency>"
    }

    Question: ${query}
    Candidate analyses (JSON list):
    ${analyses_json}
"""))


def _safe_json_loads(text: Optional[str]) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _truncate(text: Optional[str], max_len: int = 240) -> str:
    if not text:
        return ""
    t = str(text).replace("\n", " ").strip()
    return t if len(t) <= max_len else (t[: max_len - 3] + "...")


class FusionPipeline:
    def __init__(self, model: str, enable_thinking: bool=False):
        logger.info(f"Initializing FusionPipeline with model: {model}, enable_thinking: {enable_thinking}")

        self.model = model
        self.enable_thinking = enable_thinking


    def run(self, query: str, context: str) -> str:
        """
        Run the FusionPipeline.
        新逻辑（面向歧义实体问答）：
        1. 输入用户查询和上下文
        2. 歧义检测：调用大模型检查有可能有几个实体能回答用户查询，输出候选实体列表
        3. 针对每个候选实体：从原始上下文中检索相关证据 -> 构建推理链 -> 生成该实体答案（允许无法回答）
        4. 歧义消解：综合所有候选实体的推理链、答案，再次检查并给出最终回复
        5. 回复用户
        """
        try:
            logger.info(
                f"[FusionPipeline] start model={self.model} query={_truncate(query, 160)} ctx_len={len(context)}"
            )

            # Step 2: 歧义检测（直接基于原始上下文）
            logger.debug("Step 2: Ambiguity detection")
            amb_prompt = AMBIGUITY_DETECTION.safe_substitute(query=query, context=context)
            amb_resp = get_llm_chat(amb_prompt, self.model, json_format=True, enable_thinking=self.enable_thinking)
            amb_obj = _safe_json_loads(amb_resp) or {}
            candidates_raw = amb_obj.get("candidates", []) if isinstance(amb_obj, dict) else []

            # 规范化候选实体列表（去重/过滤空值）
            candidates: List[str] = []
            for item in candidates_raw if isinstance(candidates_raw, list) else []:
                if isinstance(item, dict):
                    ent = (item.get("entity") or "").strip()
                else:
                    ent = str(item).strip()
                if ent and ent not in candidates:
                    candidates.append(ent)
            logger.info(f"[FusionPipeline] candidates_from_detection n={len(candidates)} {candidates}")

            # 兜底：若歧义检测失败/为空，则先分组再抽取实体名
            if not candidates:
                logger.warning("[FusionPipeline] empty candidates; fallback to grouping extraction")
                grouped_ctx = self._group(context)
                candidates = self._extract_entities_from_grouped_context(grouped_ctx)
                logger.info(f"[FusionPipeline] candidates_from_grouping n={len(candidates)} {candidates}")

            # Step 3: 针对每个候选实体，检索证据 + 推理链 + 答案
            logger.debug("Step 3: Per-entity evidence -> reasoning -> answer")
            analyses: List[Dict[str, Any]] = []
            for ent in candidates:
                ev_prompt = ENTITY_EVIDENCE_RETRIEVAL.safe_substitute(query=query, entity=ent, context=context)
                evidence = get_llm_chat(ev_prompt, self.model, json_format=False, enable_thinking=self.enable_thinking)
                evidence = (evidence or "").strip()
                if (not evidence) or (evidence.lower() == "nothing"):
                    evidence = "Nothing"

                ra_prompt = ENTITY_REASONING_AND_ANSWER.safe_substitute(query=query, entity=ent, evidence=evidence)
                ra_resp = get_llm_chat(ra_prompt, self.model, json_format=True, enable_thinking=self.enable_thinking)
                ra_obj = _safe_json_loads(ra_resp)
                if not isinstance(ra_obj, dict):
                    ra_obj = {}

                # 强制补齐关键字段，避免模型输出缺字段导致后续崩溃
                ra_obj.setdefault("entity", ent)
                ra_obj.setdefault("can_answer", False)
                ra_obj.setdefault("answer", "N/A")
                ra_obj.setdefault("reasoning_chain", [])
                ra_obj.setdefault("used_evidence", [])

                analyses.append(ra_obj)
                logger.info(
                    "[FusionPipeline] entity=%s can_answer=%s answer=%s evidence_len=%s",
                    ent,
                    ra_obj.get("can_answer"),
                    _truncate(ra_obj.get("answer"), 160),
                    0 if evidence == "Nothing" else len(evidence),
                )

            # Step 4: 歧义消解，综合所有候选结果
            logger.debug("Step 4: Disambiguation and final response")
            analyses_json = json.dumps(analyses, ensure_ascii=False)
            dis_prompt = DISAMBIGUATION_AND_FINAL.safe_substitute(query=query, analyses_json=analyses_json)
            final_resp = get_llm_chat(dis_prompt, self.model, json_format=True, enable_thinking=self.enable_thinking)

            # Step 5: 回复用户（返回 JSON 或兜底为 JSON 文本）
            if final_resp:
                final_answer = final_resp
            else:
                final_answer = json.dumps(
                    {
                        "final_answer": "N/A",
                        "selected_entities": [],
                        "notes": "LLM call failed during disambiguation; insufficient evidence in provided context.",
                    },
                    ensure_ascii=False,
                )

            final_obj = _safe_json_loads(final_answer) or {}
            if isinstance(final_obj, dict):
                logger.info(
                    "[FusionPipeline] done selected=%s notes=%s",
                    final_obj.get("selected_entities", []),
                    _truncate(final_obj.get("notes"), 200),
                )
            else:
                logger.info("[FusionPipeline] done final_len=%s", len(final_answer))
            return final_answer
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise e


    def _group(self, context: str) -> str:
        """
        Group the evidence.
        """
        logger.debug("Executing knowledge grouping")
        prompt = KNOWLEDGE_GROUPING.safe_substitute(context=context)
        logger.debug(f"Generated grouping prompt length: {len(prompt)} characters")
        response = get_llm_chat(prompt, self.model, json_format=False, enable_thinking=self.enable_thinking)
        logger.debug("Grouping response received")
        return response


    def _extract_entities_from_grouped_context(self, grouped_context: str) -> List[str]:
        """
        从 _group 输出中抽取实体名。
        约定：分组输出通常包含形如 "Group <Entity Name>:" 的行。
        """
        if not grouped_context:
            return []
        entities: List[str] = []
        for line in grouped_context.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("group ") and ":" in line:
                head = line.split(":", 1)[0].strip()
                ent = head[6:].strip()  # len("Group ") == 6
                if ent and ent not in entities:
                    entities.append(ent)
        return entities


if __name__ == "__main__":
    pipeline = FusionPipeline(model="qwen-plus-latest")
    response = pipeline.run(query="Where was Edward Andrews born?",
                 context= "Edward Deming Andrews\n:Edward Deming Andrews(March 6, 1894 – June 6, 1964) was an American historian, educator, curator, and preeminent authority on theUnited Society of Believers in Christ's Second Appearing, best known as theShakers.[1]\n| Field | Value |\n|-------|-------|\n| Born | ( 1894-03-06 ) March 6, 1894 Pittsfield, Massachusetts , U.S. |\n| Died | June 6, 1964 (1964-06-06) (aged 70) Pittsfield, Massachusetts, U.S. |\n| Education | Amherst College ( BA ) Yale University ( PhD ) |\n| Occupation(s) | Historian, educator, curator |\n| Employer | Scarborough Day School |\n| Known for | Authority on Shakerism |\n| Spouse | Faith Young ​ ( m. 1921) ​ |\n| Children | 2 |\n\nEdward Deming Andrews -- sex or gender --> male\nEdward Deming Andrews -- country of citizenship --> United States\nEdward Deming Andrews -- instance of --> human\nEdward Deming Andrews -- educated at --> Yale University\nEdward Deming Andrews -- educated at --> Amherst College\nEdward Deming Andrews -- place of burial --> Center Cemetery\nEdward Deming Andrews -- award received --> Guggenheim Fellowship\nEdward Deming Andrews -- member of --> Hancock Shaker Village\nEdward Deming Andrews -- academic degree --> Doctor of Philosophy\nEdward Deming Andrews -- date of birth --> 1894-03-06T00:00:00Z\nEdward Deming Andrews -- date of death --> 1964-06-13T00:00:00Z\nEdward Deming Andrews -- family name --> Andrews\nEdward Deming Andrews -- given name --> Edward\nEdward Deming Andrews -- languages spoken, written or signed --> English\nEdward Deming Andrews -- rdf-schema#label --> Edward Deming Andrews\n\nEdward Gayer Andrews\n:Edward Gayer Andrews(7 August 1825 – 31 December 1907) was abishopof theMethodist Episcopal Church, elected in 1872.[1]\n| Field | Value |\n|-------|-------|\n| Born | ( 1825-08-07 ) August 7, 1825 New Hartford, New York , U.S. |\n| Died | December 31, 1907 (1907-12-31) (aged 82) New York City, U.S. |\n| Burial place | Oakwood Cemetery |\n| Alma mater | Cazenovia College Wesleyan University |\n| Occupation | Bishop |\n| Family | Grace Andrews (daughter) |\n\nEdward Gayer Andrews -- place of birth --> New Hartford\nEdward Gayer Andrews -- sex or gender --> male\nEdward Gayer Andrews -- country of citizenship --> United States\nEdward Gayer Andrews -- instance of --> human\nEdward Gayer Andrews -- educated at --> Wesleyan University\nEdward Gayer Andrews -- educated at --> Cazenovia College\nEdward Gayer Andrews -- educated at --> Mystical Seven\nEdward Gayer Andrews -- place of burial --> Oakwood Cemetery\nEdward Gayer Andrews -- date of birth --> 1825-08-07T00:00:00Z\nEdward Gayer Andrews -- date of death --> 1907-12-31T00:00:00Z\nEdward Gayer Andrews -- family name --> Andrews\nEdward Gayer Andrews -- given name --> Edward\nEdward Gayer Andrews -- given name --> Gayer\nEdward Gayer Andrews -- described by source --> The Biographical Dictionary of America\nEdward Gayer Andrews -- languages spoken, written or signed --> English\nEdward Gayer Andrews -- rdf-schema#label --> Edward Gayer Andrews\n\nEddie Andrews\n:Edwin Peter Andrews(born 18 March 1977) is aSouth Africanpolitician serving as theDeputy Mayor of Cape Townsince November 2021. A formerrugby unionfootballer, his usual position wasprop, and he played for theSpringboks.[1]He played for theStormersin theSuper 14between 2003 and 2007.[2]\n| Field | Value |\n|-------|-------|\n| Mayor | Geordin Hill-Lewis |\n| Preceded by | Ian Neilson |\n| Born | Edwin Peter Andrews ( 1977-03-18 ) 18 March 1977 (age 48) Cape Town , South Africa |\n| Political party | Democratic Alliance |\n| Education | Steenberg High School |\n| Height | 6.1 ft (1.9 m) |\n| Weight | 253 lb (115 kg) |\n| Position(s) | Prop |\n| Years | Team |\n| 2000–2006 | Western Province |\n| 2003–2007 | Stormers |\n| 2004–2007 | South Africa |\n\nEddie Andrews -- place of birth --> Cape Town\nEddie Andrews -- sex or gender --> male\nEddie Andrews -- country of citizenship --> South Africa\nEddie Andrews -- instance of --> human\nEddie Andrews -- member of sports team --> South Africa national rugby union team\nEddie Andrews -- member of sports team --> Western Province\nEddie Andrews -- position played on team / speciality --> prop\nEddie Andrews -- date of birth --> 1977-03-18T00:00:00Z\nEddie Andrews -- sport --> rugby union\nEddie Andrews -- family name --> Andrews\nEddie Andrews -- given name --> Eddie\nEddie Andrews -- languages spoken, written or signed --> English\nEddie Andrews -- rdf-schema#label --> Eddie Andrews\n\nE. Wyllys Andrews IV\n:Edward Wyllys Andrews IV(December 11, 1916 – July 3, 1971) was an American archaeologist noted for research intoMaya civilization. During his career withTulane University'sMiddle American Research Institute, Andrews focused on Mayan ruins, rediscovering several sites and leading investigations intoBalankanche,Kulubá,Coba, and more.\n| Field | Value |\n|-------|-------|\n| Born | ( 1916-12-11 ) December 11, 1916 Chicago, Illinois |\n| Died | July 3, 1971 (1971-07-03) (aged 54) New Orleans, Louisiana , US |\n| Alma mater | Harvard University |\n| Discipline | Archeology |\n| Sub-discipline | Maya civilization |\n| Institutions | Tulane University Middle American Research Institute |\n\nE. Wyllys Andrews IV -- languages spoken, written or signed --> English\nE. Wyllys Andrews IV -- different from --> E. Wyllys Andrews V\nE. Wyllys Andrews IV -- place of birth --> Chicago\nE. Wyllys Andrews IV -- sex or gender --> male\nE. Wyllys Andrews IV -- father --> Edmund Andrews\nE. Wyllys Andrews IV -- mother --> Irene Dwen Andrews\nE. Wyllys Andrews IV -- country of citizenship --> United States\nE. Wyllys Andrews IV -- instance of --> human\nE. Wyllys Andrews IV -- child --> E. Wyllys Andrews V\nE. Wyllys Andrews IV -- educated at --> Harvard University\nE. Wyllys Andrews IV -- educated at --> University of Chicago\nE. Wyllys Andrews IV -- award received --> Legion of Merit\nE. Wyllys Andrews IV -- award received --> Guggenheim Fellowship\nE. Wyllys Andrews IV -- military branch --> United States Navy\nE. Wyllys Andrews IV -- military or police rank --> lieutenant (junor grade)\nE. Wyllys Andrews IV -- date of birth --> 1916-12-11T00:00:00Z\nE. Wyllys Andrews IV -- date of death --> 1971-07-03T00:00:00Z\nE. Wyllys Andrews IV -- family name --> Andrews\nE. Wyllys Andrews IV -- given name --> Edward\nE. Wyllys Andrews IV -- given name --> Wyllys\nE. Wyllys Andrews IV -- rdf-schema#label --> Wyllys Andrews",
)
    print(response)
