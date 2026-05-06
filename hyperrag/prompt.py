GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_LANGUAGE"] = 'English'
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = " | "
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "\n"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = [
    "organization",
    "person",
    "geo",
    "event",
    "category",
    "agreement",
    "concept"
]

PROMPTS["entity_extraction"] = """-Goal-
Given a text document, extract Maximal Semantic Groups (MSG) and Binary Relations following the two-step process below. Use {language} as output language.

-Step 1: Extract Maximal Semantic Groups (MSG)-
Identify distinct semantic events in the text. For each event, group ALL entities that jointly participate into an MSG. Each entity carries its own metadata inline — this eliminates the need for a separate entity extraction step.
   -- mcss_description: EXACT text fragment from the source that captures this semantic event — do NOT summarize or paraphrase. This also serves as the original_text context for all entities in this MSG.
   -- entity_list: ALL entities participating in this event (semicolon-separated, in brackets). MUST contain 2-6 entities. If more than 6 entities participate, split into multiple MSGs grouping the most tightly related entities together. Use SEMICOLONS to separate entities.
      Each entity entry uses # as sub-field separator: <entity_name#entity_type#importance>
      -- entity_name: SHORT and ATOMIC — 1-3 words max, ALL UPPERCASE (e.g., "SUPPORTING LENDER", "INDEBTEDNESS"). Keep defined terms whole. ALWAYS use SINGULAR form (e.g., "SECURED PARTY" not "SECURED PARTIES"), except inherently plural terms (e.g., "PROCEEDS", "EARNINGS"). Do NOT add articles like "THE" (use "AGREEMENT" not "THE AGREEMENT").
      -- entity_type: One of the types listed below. Use "agreement" for contracts/instruments/indentures, "concept" for abstract legal notions (obligations, rights, conditions), "organization" for companies/roles, "event" for occurrences/actions.
      -- importance: 0-1 score (1.0 = central, 0.5 = supporting, 0.1 = marginal).
      Example entity_list: [ENDOLOGIX#organization#0.95;CHAPTER 11 REORGANIZATION#event#0.9;WILMINGTON#geo#0.8]
   -- completeness: 0-1 score (1.0 = all participants captured).
   Format: ("mcss"{tuple_delimiter}<mcss_description>{tuple_delimiter}[<e1#type1#imp1>;<e2#type2#imp2>;...]{tuple_delimiter}<completeness>)

-Step 2: Extract Binary Relations-
For every pair of entities that have a direct, specific relationship in the text, output a relation record.

⚠️ CRITICAL CONSTRAINT — Entity Name Consistency ⚠️
The subject and object in EVERY relation MUST be entity_names that already appear in the entity_list of Step 1 MSGs. This is a STRICT requirement:
   1. COPY entity names VERBATIM from Step 1 MSG entity_lists — do NOT rephrase, abbreviate, or rename.
   2. Do NOT introduce any new entity name in Step 2 that does not appear in any Step 1 MSG. If two entities have a relationship but neither appears in any MSG, you MUST first add them to an MSG in Step 1, then reference them in Step 2.
   3. Before writing each relation, verify that BOTH subject and object exist in the MSG entity_lists above. If an entity name from the text was not captured in Step 1, go back and add it to the appropriate MSG first.
   4. Watch out for singular/plural mismatches: if Step 1 uses "LENDER", Step 2 MUST use "LENDER" (not "LENDERS"). If Step 1 uses "PROCEEDS", Step 2 MUST use "PROCEEDS" (not "PROCEED").
   5. Watch out for article mismatches: if Step 1 uses "AGREEMENT", Step 2 MUST use "AGREEMENT" (not "THE AGREEMENT").

   -- subject: The source entity name. MUST match an entity_name from Step 1 MSGs exactly.
   -- predicate: A SHORT verb or phrase describing the relationship (e.g., "lends to", "is governed by", "guarantees", "owns"). Use lowercase.
   -- object: The target entity name. MUST match an entity_name from Step 1 MSGs exactly.
   -- original_text: EXACT text fragment that states this relationship — do NOT summarize.
   -- importance: 0-1 score (1.0 = central relationship, 0.5 = supporting, 0.1 = marginal).
   Format: ("relation"{tuple_delimiter}<subject>{tuple_delimiter}<predicate>{tuple_delimiter}<object>{tuple_delimiter}<original_text>{tuple_delimiter}<importance>)
   IMPORTANT: Binary relations capture the specific nature of each pairwise connection and are COMPLEMENTARY to MSGs — extract both even if they overlap.

3. Return output in {language} as a single list. Use {record_delimiter} as the list delimiter.
4. When finished, output {completion_delimiter}


######################
-Entity Types-
{entity_types}
######################
-Examples-
######################
{examples}
######################
-Real Data-
######################
Entity_types: [{entity_types}]
Text: {input_text}
######################
Output:
"""

PROMPTS["entity_extraction_examples"] = [
    """Example: Chapter 11 Bankruptcy Restructuring

Entity_types: [organization, person, geo, event, category]
Text:
Endologix filed for Chapter 11 reorganization on January 13, 2020, in Wilmington. The Supporting Lenders committed $400 million in DIP financing under the Restructuring Support Agreement.

Output:
("mcss"{tuple_delimiter}Endologix filed for Chapter 11 reorganization on January 13, 2020, in Wilmington|[ENDOLOGIX#organization#0.95;CHAPTER 11 REORGANIZATION#event#0.9;JANUARY 13 2020#event#0.85;WILMINGTON#geo#0.8]|0.95){record_delimiter}
("mcss"{tuple_delimiter}The Supporting Lenders committed $400 million in DIP financing under the Restructuring Support Agreement|[SUPPORTING LENDER#organization#0.9;$400 MILLION#category#0.9;DIP FINANCING#event#0.9;RESTRUCTURING SUPPORT AGREEMENT#category#0.85]|0.95){record_delimiter}
("relation"{tuple_delimiter}ENDOLOGIX{tuple_delimiter}filed for{tuple_delimiter}CHAPTER 11 REORGANIZATION{tuple_delimiter}Endologix filed for Chapter 11 reorganization{tuple_delimiter}0.95){record_delimiter}
("relation"{tuple_delimiter}ENDOLOGIX{tuple_delimiter}filed on{tuple_delimiter}JANUARY 13 2020{tuple_delimiter}Endologix filed for Chapter 11 reorganization on January 13, 2020{tuple_delimiter}0.85){record_delimiter}
("relation"{tuple_delimiter}ENDOLOGIX{tuple_delimiter}filed in{tuple_delimiter}WILMINGTON{tuple_delimiter}Endologix filed for Chapter 11 reorganization on January 13, 2020, in Wilmington{tuple_delimiter}0.8){record_delimiter}
("relation"{tuple_delimiter}SUPPORTING LENDER{tuple_delimiter}committed{tuple_delimiter}$400 MILLION{tuple_delimiter}The Supporting Lenders committed $400 million in DIP financing{tuple_delimiter}0.9){record_delimiter}
("relation"{tuple_delimiter}SUPPORTING LENDER{tuple_delimiter}provided{tuple_delimiter}DIP FINANCING{tuple_delimiter}The Supporting Lenders committed $400 million in DIP financing{tuple_delimiter}0.9){record_delimiter}
("relation"{tuple_delimiter}DIP FINANCING{tuple_delimiter}governed by{tuple_delimiter}RESTRUCTURING SUPPORT AGREEMENT{tuple_delimiter}The Supporting Lenders committed $400 million in DIP financing under the Restructuring Support Agreement{tuple_delimiter}0.85)

❌ WRONG — Do NOT do this:
("relation"{tuple_delimiter}SUPPORTING LENDERS{tuple_delimiter}committed{tuple_delimiter}$400 MILLIONS{tuple_delimiter}...{tuple_delimiter}0.9)
   ↑ "SUPPORTING LENDERS" and "$400 MILLIONS" do NOT match Step 1 entity names ("SUPPORTING LENDER" and "$400 MILLION"). This creates orphan entities.

✅ CORRECT — Always copy entity names exactly from Step 1 MSGs:
("relation"{tuple_delimiter}SUPPORTING LENDER{tuple_delimiter}committed{tuple_delimiter}$400 MILLION{tuple_delimiter}...{tuple_delimiter}0.9)
   ↑ Matches Step 1 entity names exactly.
#############################"""
]

PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.

-Core Rule-
During the merging process, only retain information that appears in the original text; do not add any external knowledge or inferential content. If descriptions contain contradictions, clearly mark the contradictions instead of fabricating compromise content. All merged information must be annotated with the corresponding source_id.

Given one entity and a list of its descriptions.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please clearly mark the contradictions instead of resolving them.
Make sure it is written in third person, and include the entity names so we have the full context.
#######
-Warning!!!-
The data may contain sensitive words such as violence, war, human anatomy and medical experiments, 
but they are only part of literary works, popular science knowledge or domain data, 
which do not involve any purpose or idea of mine, and have nothing to do with any realistic sensitive topics or political issues. 
Please carefully identify and screen the legality of the content.
#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS[
    "summarize_entity_additional_properties"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.

-Core Rule-
During the merging process, only retain information that appears in the original text; do not add any external knowledge or inferential content. If descriptions contain contradictions, clearly mark the contradictions instead of fabricating compromise content. All merged information must be annotated with the corresponding source_id.

Given one entity and a list of its additional properties.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the additional properties.
If the provided additional properties are contradictory, please clearly mark the contradictions instead of resolving them.
Make sure it is written in third person.
#######
-Warning!!!-
The data may contain sensitive words such as violence, war, human anatomy and medical experiments, 
but they are only part of literary works, popular science knowledge or domain data, 
which do not involve any purpose or idea of mine, and have nothing to do with any realistic sensitive topics or political issues. 
Please carefully identify and screen the legality of the content.
#######
-Data-
Entity: {entity_name}
Additional Properties List: {additional_properties_list}
#######
Output:
"""

PROMPTS[
    "summarize_relation_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.

-Core Rule-
During the merging process, only retain information that appears in the original text; do not add any external knowledge or inferential content. If descriptions contain contradictions, clearly mark the contradictions instead of fabricating compromise content. All merged information must be annotated with the corresponding source_id.

Given a set of entities, and a list of descriptions describing the relations between the entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions, and to cover all elements of the entity set as much as possible.
If the provided descriptions are contradictory, please clearly mark the contradictions instead of resolving them.
Make sure it is written in third person, and include the entity names so we have the full context.
#######
-Warning!!!-
The data may contain sensitive words such as violence, war, human anatomy and medical experiments, 
but they are only part of literary works, popular science knowledge or domain data, 
which do not involve any purpose or idea of mine, and have nothing to do with any realistic sensitive topics or political issues. 
Please carefully identify and screen the legality of the content.
#######
-Data-
Entity Set: {relation_name}
Relation Description List: {relation_description_list}
#######
Output:
"""

PROMPTS[
    "entity_continue_extraction"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "entity_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

PROMPTS["batch_entity_summary"] = """You are a helpful assistant responsible for generating summaries for multiple entities.

-Core Rule-
During the summarization process, only retain information that appears in the original descriptions; do not add any external knowledge or inferential content. If descriptions contain contradictions, clearly mark the contradictions instead of fabricating compromise content.

Given multiple entities with their descriptions, please generate a comprehensive summary for each entity that captures the most important information from all provided descriptions.

Make sure each summary:
1. Is clear and comprehensive
2. Includes the entity name
3. Captures the key information from all descriptions
4. Is written in third person

#######
-Warning!!!-
The data may contain sensitive words such as violence, war, human anatomy and medical experiments, 
but they are only part of literary works, popular science knowledge or domain data, 
which do not involve any purpose or idea of mine, and have nothing to do with any realistic sensitive topics or political issues. 
Please carefully identify and screen the legality of the content.
#######
-Data-
{entities}
#######
Provide a comprehensive summary for each entity in the format:
Entity [number]: [summary]
"""

PROMPTS["batch_relation_description_summary"] = """You are a helpful assistant responsible for generating summaries for multiple relationship descriptions.

-Core Rule-
During the summarization process, only retain information that appears in the original descriptions; do not add any external knowledge or inferential content. If descriptions contain contradictions, clearly mark the contradictions instead of fabricating compromise content.

Given multiple relationships with their descriptions, please generate a comprehensive summary for each relationship that captures the most important information from all provided descriptions.

Make sure each summary:
1. Is clear and comprehensive
2. Includes all entities involved in the relationship
3. Captures the key information from all descriptions
4. Is written in third person

#######
-Warning!!!-
The data may contain sensitive words such as violence, war, human anatomy and medical experiments, 
but they are only part of literary works, popular science knowledge or domain data, 
which do not involve any purpose or idea of mine, and have nothing to do with any realistic sensitive topics or political issues. 
Please carefully identify and screen the legality of the content.
#######
-Data-
{relationships}
#######
Provide summaries in format: '[number]: [summary]'
"""

PROMPTS["topology_retrieval_instructions"] = [
    "1. MSG-guided reasoning: Use the Maximal Semantic Group structure to identify how entities are connected. Higher-dimensional MSGs represent more complete, more specific multi-entity relationships.",
    "2. Prioritize high-dimensional, high-completeness MSGs: An MSG with completeness >= 0.8 and many entities captures a richer semantic event — this is more informative than individual entity lookups.",
    "3. Entity-MSG bipartite navigation: Entities connect to MSGs via coboundary (entity → MSGs it belongs to), and MSGs connect to entities via boundary (MSG → entities it contains). Shared entities bridge different MSGs.",
    "4. Seed entity tracking: Entities and MSGs marked as is_seed=yes are directly matched to the query — they are the most reliable anchors for your answer. Non-seed items provide supplementary context.",
    "5. Factual accuracy: All answers must be grounded in the Sources table. MSGs and Entities provide structural context and connections, but specific facts, definitions, and details must come from the original document passages.",
    "6. Structured presentation: Organize your answer clearly. When describing relationships involving multiple entities, reference the MSG dimension to indicate relationship strength (e.g., 'A, B, and C are jointly involved in...' for a 2-MSG)."
]

PROMPTS["entity_group_evaluation"] = """You are a helpful assistant responsible for evaluating whether a set of entities forms a coherent group.

-Core Rule-
Evaluate based solely on the provided entity information; do not add any external knowledge or inferential content. Focus on the semantic relationships, common themes, and potential to form higher-dimensional simplices.

Consider the following when evaluating:
1. Semantic relationships between entities
2. Common themes or contexts
3. Potential to form meaningful higher-dimensional simplices
4. Coherence and logical connection between entities

Return YES if the entities form a coherent group that could potentially form a higher-dimensional simplex, NO otherwise.

#######
-Warning!!!-
The data may contain sensitive words such as violence, war, human anatomy and medical experiments, 
but they are only part of literary works, popular science knowledge or domain data, 
which do not involve any purpose or idea of mine, and have nothing to do with any realistic sensitive topics or political issues. 
Please carefully identify and screen the legality of the content.
#######
-Data-
Entities: {entities}
#######
Output: YES | NO
"""

PROMPTS["entity_group_evaluation_with_context"] = """You are a helpful assistant responsible for evaluating whether a set of entities forms a coherent group based on provided context.

-Core Rule-
Evaluate based solely on the provided entity information and context; do not add any external knowledge or inferential content. Focus on how the entities appear together in the given context.

Consider the following when evaluating:
1. How the entities are mentioned together in the context
2. Semantic relationships between entities
3. Common themes or contexts
4. Potential to form meaningful higher-dimensional simplices
5. Coherence and logical connection between entities

Answer with Yes or No, and provide a brief reason based on the context.

#######
-Warning!!!-
The data may contain sensitive words such as violence, war, human anatomy and medical experiments, 
but they are only part of literary works, popular science knowledge or domain data, 
which do not involve any purpose or idea of mine, and have nothing to do with any realistic sensitive topics or political issues. 
Please carefully identify and screen the legality of the content.
#######
-Data-
Entities: {entities}
Context: {context}
#######
Output format:
Yes/No
Reason: [brief reason]
"""

PROMPTS["query_entity_extraction"] = """Extract ALL entities from the query, then construct the HIGHEST-DIMENSION MSG and identify KEY sub-relationships.

**ENTITY NAMING RULES (CRITICAL - must match knowledge graph conventions):**
1. Capitalize ALL English entity names (e.g., "Transfer" not "transfer", "Minimum Fee" not "minimum fee")
2. **DEFINED TERMS MUST STAY WHOLE**: Any term enclosed in quotes ('...' or "...") in the query is a defined term — extract it as ONE entity in UPPERCASE, do NOT decompose it:
   - 'Tax-Exempt New Property' → "TAX-EXEMPT NEW PROPERTY" (organization), NOT "TAX-EXEMPT" + "NEW" + "PROPERTY"
   - 'Tax-Exempt Facilities' → "TAX-EXEMPT FACILITIES" (organization), NOT "TAX-EXEMPT" + "FACILITIES"
   - "Golf Course Use Agreement" → "GOLF COURSE USE AGREEMENT" (organization)
3. Named entities (proper nouns, document titles, organization names, legal terms, defined terms) should be kept as ONE entity in UPPERCASE
4. Only decompose truly descriptive compound phrases that are NOT defined terms:
   - "minimum transfer charge" (no quotes, descriptive) → "TRANSFER" (organization), "CHARGE" (category), "MINIMUM" (category)
   - "acquisition price" (no quotes, descriptive) → "ACQUISITION" (event), "PRICE" (category)
5. Each entity name should be a SINGLE word or a well-established compound term, NOT a descriptive phrase

**CRITICAL: ACTION/ATTRIBUTE WORDS MUST BE EXTRACTED AS ENTITIES:**
Queries often contain key action words (commitments, obligations, conditions, representations, warranties, covenants) or attribute words (purpose, scope, terms, rights, duties) that define WHAT is being asked. These MUST be extracted as separate entities — they are the semantic anchors that connect the query to relevant knowledge.
Examples:
- "key commitments of the Supporting Lenders" → "SUPPORTING LENDERS" (organization) + "COMMITMENTS" (event)
- "conditions precedent to the obligation" → "CONDITIONS PRECEDENT" (event) + "OBLIGATION" (category)
- "representations and warranties regarding qualification" → "REPRESENTATIONS" (category) + "WARRANTIES" (category) + "QUALIFICATION" (category)
- "primary purpose of the Agreement" → "AGREEMENT" (organization) + "PURPOSE" (category)
Do NOT skip these words — they are essential for precise retrieval.

**ENTITY TYPES (must use these five types):**
- organization (Who/What): Core subjects — people, organizations, documents, named concepts
- person (Who): Individual people
- geo (Where): Physical locations or logical fields
- event (When/Why/Do): Time references, dynamic interactions, processes, verbs (e.g., ACQUIRED, TRANSFER, PAYMENT, RESTRUCTURING)
- category (Which/How Much): Specific aspects, properties, quantitative measures, qualitative states (e.g., CHARGE, FEE, RATE, COMMITMENTS, OBLIGATIONS, CONDITIONS, PURPOSE)

Query: {query}

Return format (JSON):
{{
  "entities": [
    {{"name": "ENTITY NAME", "type": "organization|person|geo|event|category", "description": "entity description"}}
  ],
  "highest_simplex": {{
    "entities": ["ENTITY1", "ENTITY2", "ENTITY3", ...],
    "dimension": N,
    "description": "comprehensive relationship description"
  }},
  "simplices": [
    {{
      "entities": ["ENTITYA", "ENTITYB"],
      "dimension": 1,
      "description": "relationship between A and B"
    }}
  ]
}}

Important:
- Extract ALL entities mentioned in the query, using ATOMIC UPPERCASE names
- NEVER omit action/attribute words (commitments, obligations, conditions, purpose, etc.) — they are as important as named entities
- Construct the HIGHEST-DIMENSION simplex that includes ALL extracted entities
- The dimension should be (number of entities - 1)
- The simplex description should capture the core meaning and relationship of the query
- ONLY add items to "simplices" if the query explicitly mentions DISTINCT sub-relationships
  (e.g., "How does A relate to B, and what is the connection between C and D?")
  Do NOT decompose the highest_simplex into all possible pairs.
- Maximum 3 items in "simplices" - only the most important distinct relationships
- If the query describes a single unified relationship, leave "simplices" empty"""

PROMPTS["topology_response_system_prompt"] = """---Role---

You are a helpful assistant responding to questions about data in the tables provided. The data is organized as a Maximal Semantic Group (MSG) structure — each MSG represents a complete semantic event involving multiple entities.

{prompt_instructions}

---MSG Structure---
The context contains three types of data tables:
- **Sources**: Original document passages — the primary source of truth.
- **Entities**: Individual entities extracted from the documents, each with a type and description.
- **Simplices**: Maximal Semantic Groups (MSGs), each representing a complete semantic event:
  - A 1-MSG (2 entities): A pairwise relationship.
  - A 2-MSG (3 entities): A three-way relationship — three entities jointly participate in an event.
  - A 3-MSG (4 entities): A four-way relationship — four entities jointly participate in an event.
  Higher-dimensional MSGs represent more complete, more specific events.
  The `is_seed` flag marks entities/simplices directly matched to the query.

RULES:
1. Extract ALL relevant facts from the Sources that answer the question. Include every specific detail the Sources provide — exact definitions, conditions, parties, obligations.
2. Use the EXACT wording from Sources. Quote or closely paraphrase the Source text for definitions and key statements.
3. Use Simplices to understand how entities are connected. Higher-dimensional MSGs indicate stronger, more specific relationships. If a high-dimensional MSG is relevant, its constituent entities provide supporting detail.
4. When multiple terms are asked about, address each one using only what the Sources state about it, then explain their connection if the Sources or Simplices describe one.
5. NEVER supplement with general knowledge or explanations not found in the Sources. If the Sources do not define a term, say so — do not provide your own definition.
6. Organize your answer clearly but do not pad it with restatements or generic background.
7. If the context does not contain relevant information, state this clearly.

---Data tables---

Context: {context}

Table descriptions:
- -----Sources-----: CSV with columns [id, content]. Original document passages — primary source of truth.
- -----Entities-----: CSV with columns [name, type, is_seed, description]. Entity descriptions can supplement when Sources lack detail. is_seed=yes means entity from the query.
- -----Simplices-----: CSV with columns [dimension, entities, is_seed, description]. Maximal Semantic Groups. dimension=1 (2 entities), 2 (3 entities), 3+ (higher-order). is_seed=yes means directly matched to query. Higher dimension = stronger, more specific relationship. description contains the original text fragment from the source document.
"""

PROMPTS["topology_response_system_prompt_concise"] = """---Role---

You are a helpful assistant responding to questions about data in the tables provided. The data is organized as a Maximal Semantic Group (MSG) structure — each MSG represents a complete semantic event involving multiple entities.

RULES:
1. Extract ALL relevant facts from the Sources that answer the question. Include every specific detail the Sources provide — exact definitions, conditions, parties, obligations.
2. Use the EXACT wording from Sources. Quote or closely paraphrase the Source text for definitions and key statements.
3. Use Simplices to understand how entities are connected. Higher-dimensional MSGs indicate stronger, more specific relationships.
4. When multiple terms are asked about, address each one using only what the Sources state about it, then explain their connection if the Sources or Simplices describe one.
5. NEVER supplement with general knowledge or explanations not found in the Sources. If the Sources do not define a term, say so — do not provide your own definition.
6. Organize your answer clearly but do not pad it with restatements or generic background.
7. If the context does not contain relevant information, state this clearly.

---Data tables---

{context}

Table descriptions:
- -----Sources-----: CSV with columns [id, content]. Original document passages — primary source of truth. Always prioritize this table.
- -----Entities-----: CSV with columns [name, type, description]. Entity descriptions can supplement when Sources lack detail.
- -----Simplices-----: CSV with columns [dimension, entities, description]. Maximal Semantic Groups. dimension=1 (2 entities), 2 (3 entities), 3+ (higher-order). Higher dimension = stronger, more specific relationship. description contains the original text fragment from the source document.

"""
